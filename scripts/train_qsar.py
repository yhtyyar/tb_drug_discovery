#!/usr/bin/env python
"""Train QSAR model for TB drug activity prediction.

Production-quality QSAR training pipeline with:
- MoleculeStandardizer (tautomers/salts) before descriptor calculation
- Scaffold split (default) or random split
- Imputer + RobustScaler fitted only on X_train (no leakage)
- Optional Optuna hyperparameter optimisation (--n-trials)
- QSARModel training with cross-validation
- Probability calibration for classification (QSARCalibrator)
- Learning curve analysis (sklearn learning_curve)
- Git hash + dataset hash in saved metrics for provenance
- MLflow tracking (--use-mlflow)

Saved artefacts:
  qsar_rf_model.joblib, preprocessor.joblib, calibrator.joblib,
  qsar_metrics.json, feature_importance.csv, learning_curve.json

Usage:
    python scripts/train_qsar.py --data data/raw/chembl_inhA.csv --output models/
    python scripts/train_qsar.py --split scaffold --n-trials 30 --use-mlflow
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.model_selection import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from data.chembl_loader import ChEMBLLoader
from data.descriptor_calculator import DescriptorCalculator
from data.mol_standardizer import MoleculeStandardizer
from data.scaffold_split import scaffold_split_df
from evaluation.cross_validation import cross_validate_model
from models.qsar_model import QSARModel, _git_commit_hash
from utils.config import Config
from utils.logger import setup_logger

# --------------------------------------------------------------------------- #
# Optional dependencies
# --------------------------------------------------------------------------- #
try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from models.calibration import QSARCalibrator
    HAS_CALIBRATOR = True
except ImportError:
    HAS_CALIBRATOR = False


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Train QSAR model for TB drug discovery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data", default="data/raw/chembl_inhA.csv",
                   help="Path to input CSV file")
    p.add_argument("--output", default="models",
                   help="Output directory for model and artefacts")
    p.add_argument("--task", choices=["regression", "classification"],
                   default="classification",
                   help="Task type")
    p.add_argument("--threshold", type=float, default=6.0,
                   help="pIC50 threshold for activity classification")
    p.add_argument("--n-estimators", type=int, default=100,
                   help="Number of trees in Random Forest")
    p.add_argument("--n-folds", type=int, default=5,
                   help="Number of cross-validation folds")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--split", choices=["random", "scaffold"], default="scaffold",
                   help="Data splitting strategy")
    p.add_argument("--config", default=None,
                   help="Path to YAML configuration file")
    p.add_argument("--n-trials", type=int, default=0,
                   help="Optuna hyperopt trials (0 = disabled)")
    p.add_argument("--use-mlflow", action="store_true", default=False,
                   help="Enable MLflow experiment tracking")
    p.add_argument("--experiment-name", default="tb-qsar",
                   help="MLflow experiment name")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Provenance helpers
# --------------------------------------------------------------------------- #
def _dataset_hash(df: pd.DataFrame) -> str:
    """MD5 hash of the raw dataframe bytes for provenance tracking."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


def _provenance(df_raw: pd.DataFrame, args: argparse.Namespace) -> dict:
    return {
        "git_commit": _git_commit_hash(),
        "dataset_hash": _dataset_hash(df_raw),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data_path": str(args.data),
        "split": args.split,
        "task": args.task,
        "seed": args.seed,
    }


# --------------------------------------------------------------------------- #
# Preprocessing (no-leakage pipeline)
# --------------------------------------------------------------------------- #
def build_preprocessor() -> Pipeline:
    """Return an unfitted Imputer + RobustScaler pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])


# --------------------------------------------------------------------------- #
# Data splitting
# --------------------------------------------------------------------------- #
def split_data(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    args: argparse.Namespace,
) -> tuple:
    """Return (X_train, X_test, y_train, y_test, smiles_test)."""
    X = df[feature_cols].values
    y = df[target_col].values
    smiles = df["smiles"].values

    if args.split == "scaffold":
        logger.info("Using scaffold-based splitting (70/10/20)...")
        split_df = pd.DataFrame({"smiles": smiles})
        split_df[feature_cols] = X
        split_df["__y__"] = y

        train_df, val_df, test_df = scaffold_split_df(
            split_df,
            smiles_col="smiles",
            frac_train=0.7,
            frac_val=0.1,
            frac_test=0.2,
            random_seed=args.seed,
        )
        X_train = train_df[feature_cols].values
        y_train = train_df["__y__"].values
        # merge val into test for final evaluation (common practice)
        X_test = pd.concat([val_df, test_df])[feature_cols].values
        y_test = pd.concat([val_df, test_df])["__y__"].values
        smiles_test = pd.concat([val_df, test_df])["smiles"].values
    else:
        logger.info("Using random splitting (80/20)...")
        from sklearn.model_selection import train_test_split
        stratify = y if args.task == "classification" else None
        X_train, X_test, y_train, y_test, smiles_train, smiles_test = (
            train_test_split(X, y, smiles,
                             test_size=0.2,
                             random_state=args.seed,
                             stratify=stratify)
        )

    logger.info(f"Split sizes — train: {len(X_train)}, test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# Optuna hyperopt
# --------------------------------------------------------------------------- #
def _optuna_objective(trial, X_train, y_train, args):
    """Optuna objective: maximise CV ROC-AUC (clf) or R² (reg)."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features",
                                                   ["sqrt", "log2", 0.5]),
    }
    model = QSARModel(task=args.task, random_seed=args.seed, **params)
    cv = cross_validate_model(
        model.model, X_train, y_train,
        n_folds=3, task=args.task, random_seed=args.seed,
    )
    return cv["roc_auc_mean"] if args.task == "classification" else cv["r2_mean"]


def run_optuna(X_train, y_train, args) -> dict:
    """Run Optuna study and return best hyperparameters."""
    if not HAS_OPTUNA:
        logger.warning("Optuna not installed — skipping hyperopt.")
        return {}
    logger.info(f"Running Optuna hyperopt with {args.n_trials} trials...")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(
        lambda trial: _optuna_objective(trial, X_train, y_train, args),
        n_trials=args.n_trials,
        show_progress_bar=False,
    )
    logger.info(f"Best params: {study.best_params}  "
                f"value={study.best_value:.4f}")
    return study.best_params


# --------------------------------------------------------------------------- #
# Learning curves
# --------------------------------------------------------------------------- #
def compute_learning_curves(estimator, X, y, args) -> dict:
    """Compute sklearn learning curves and return serialisable dict."""
    logger.info("Computing learning curves...")
    scoring = "roc_auc" if args.task == "classification" else "r2"
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        cv=min(args.n_folds, 5),
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 8),
        random_state=args.seed,
        n_jobs=-1,
    )
    return {
        "train_sizes": train_sizes.tolist(),
        "train_scores_mean": train_scores.mean(axis=1).tolist(),
        "train_scores_std": train_scores.std(axis=1).tolist(),
        "val_scores_mean": val_scores.mean(axis=1).tolist(),
        "val_scores_std": val_scores.std(axis=1).tolist(),
        "scoring": scoring,
    }


# --------------------------------------------------------------------------- #
# MLflow helpers
# --------------------------------------------------------------------------- #
def _start_mlflow_run(args) -> object:
    if not (args.use_mlflow and HAS_MLFLOW):
        return None
    mlflow.set_experiment(args.experiment_name)
    run = mlflow.start_run(
        run_name=f"{args.task}-{args.split}-{datetime.now():%Y%m%d-%H%M%S}"
    )
    mlflow.log_params({
        "task": args.task,
        "split_strategy": args.split,
        "n_estimators": args.n_estimators,
        "threshold": args.threshold,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "n_optuna_trials": args.n_trials,
    })
    logger.info(f"MLflow run started: {mlflow.get_tracking_uri()}")
    return run


def _log_mlflow_results(test_metrics, cv_results, model, paths, run):
    if run is None or not HAS_MLFLOW:
        return
    for k, v in test_metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(f"test_{k}", v)
    for k, v in cv_results.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(f"cv_{k}", v)
    mlflow.sklearn.log_model(model.model, "qsar_model")
    for p in paths:
        if Path(p).exists():
            mlflow.log_artifact(str(p))
    mlflow.end_run()


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    setup_logger(level="INFO")
    logger.info("=" * 60)
    logger.info("TB Drug Discovery — QSAR Training Pipeline")
    logger.info("=" * 60)

    mlflow_run = _start_mlflow_run(args)

    config = Config(args.config)
    config.set("random_seed", args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Step 1: Load raw data
    # ------------------------------------------------------------------ #
    logger.info("Step 1: Loading data...")
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    loader = ChEMBLLoader(random_seed=args.seed)
    df_raw = loader.load_from_csv(str(data_path))
    prov = _provenance(df_raw, args)
    logger.info(f"Loaded {len(df_raw)} records  (dataset hash: {prov['dataset_hash'][:8]})")

    # ------------------------------------------------------------------ #
    # Step 2: Standardise molecules (tautomers / salt stripping)
    # ------------------------------------------------------------------ #
    logger.info("Step 2: Standardising molecules...")
    standardizer = MoleculeStandardizer()
    df_raw["smiles"] = standardizer.standardize_smiles_series(df_raw["smiles"])
    df_raw = df_raw.dropna(subset=["smiles"])
    logger.info(f"After standardisation: {len(df_raw)} valid SMILES")

    # ------------------------------------------------------------------ #
    # Step 3: Preprocess labels
    # ------------------------------------------------------------------ #
    logger.info("Step 3: Preprocessing labels...")
    df_clean = loader.preprocess(df_raw)
    if args.task == "classification":
        df_clean = loader.create_activity_labels(df_clean, threshold=args.threshold)
    target_col = "active" if args.task == "classification" else "pIC50"

    processed_dir = output_dir.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    loader.save_processed(df_clean, str(processed_dir / "cleaned_chembl_inhA.csv"))

    # ------------------------------------------------------------------ #
    # Step 4: Calculate descriptors
    # ------------------------------------------------------------------ #
    logger.info("Step 4: Calculating molecular descriptors...")
    calculator = DescriptorCalculator(lipinski=True, topological=True, extended=True)
    df_desc = calculator.calculate_from_dataframe(df_clean, smiles_col="smiles")
    feature_cols = calculator.descriptor_names
    logger.info(f"Calculated {len(feature_cols)} descriptors")
    df_desc.to_csv(processed_dir / "descriptors.csv", index=False)

    df_model = df_desc.dropna(subset=[target_col]).copy()
    logger.info(f"Samples with valid target: {len(df_model)}")

    # ------------------------------------------------------------------ #
    # Step 5: Split data (scaffold or random)
    # ------------------------------------------------------------------ #
    logger.info("Step 5: Splitting data...")
    X_train, X_test, y_train, y_test = split_data(
        df_model, feature_cols, target_col, args
    )

    # ------------------------------------------------------------------ #
    # Step 6: Fit preprocessor ONLY on X_train (no leakage)
    # ------------------------------------------------------------------ #
    logger.info("Step 6: Fitting preprocessor on training data only...")
    preprocessor = build_preprocessor()
    X_train_pp = preprocessor.fit_transform(X_train)
    X_test_pp = preprocessor.transform(X_test)

    preprocessor_path = output_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Preprocessor saved: {preprocessor_path}")

    # ------------------------------------------------------------------ #
    # Step 7: Optional Optuna hyperopt
    # ------------------------------------------------------------------ #
    best_params = {}
    if args.n_trials > 0:
        best_params = run_optuna(X_train_pp, y_train, args)

    n_estimators = best_params.pop("n_estimators", args.n_estimators)
    model = QSARModel(
        task=args.task,
        n_estimators=n_estimators,
        random_seed=args.seed,
        **best_params,
    )

    # ------------------------------------------------------------------ #
    # Step 8: Train model
    # ------------------------------------------------------------------ #
    logger.info("Step 8: Training QSAR model...")
    model.fit(X_train_pp, y_train, feature_names=feature_cols)

    # ------------------------------------------------------------------ #
    # Step 9: Cross-validation
    # ------------------------------------------------------------------ #
    logger.info(f"Step 9: {args.n_folds}-fold cross-validation...")
    cv_results = cross_validate_model(
        model.model, X_train_pp, y_train,
        n_folds=args.n_folds, task=args.task, random_seed=args.seed,
    )

    # ------------------------------------------------------------------ #
    # Step 10: Test-set evaluation
    # ------------------------------------------------------------------ #
    logger.info("Step 10: Evaluating on held-out test set...")
    test_metrics = model.evaluate(X_test_pp, y_test)
    logger.info(f"Test metrics: {test_metrics}")

    # ------------------------------------------------------------------ #
    # Step 11: Probability calibration (classification only)
    # ------------------------------------------------------------------ #
    calibrator_path = output_dir / "calibrator.joblib"
    if args.task == "classification" and HAS_CALIBRATOR:
        logger.info("Step 11: Calibrating probabilities...")
        calibrator = QSARCalibrator(method="isotonic")
        calibrator.fit(model, X_train_pp, y_train)
        joblib.dump(calibrator, calibrator_path)
        logger.info(f"Calibrator saved: {calibrator_path}")
    else:
        if args.task == "classification":
            logger.warning("QSARCalibrator not available — skipping calibration.")

    # ------------------------------------------------------------------ #
    # Step 12: Learning curves
    # ------------------------------------------------------------------ #
    lc_data = compute_learning_curves(model.model, X_train_pp, y_train, args)
    lc_path = output_dir / "learning_curve.json"
    with open(lc_path, "w") as fh:
        json.dump(lc_data, fh, indent=2)
    logger.info(f"Learning curves saved: {lc_path}")

    # ------------------------------------------------------------------ #
    # Step 13: Save model and artefacts
    # ------------------------------------------------------------------ #
    logger.info("Step 13: Saving model and artefacts...")

    model_path = output_dir / "qsar_rf_model.joblib"
    model.save(str(model_path))

    # Compile full metrics with provenance
    all_metrics = {
        "provenance": prov,
        "task": args.task,
        "split_strategy": args.split,
        "n_samples_train": int(len(X_train)),
        "n_samples_test": int(len(X_test)),
        "n_features": len(feature_cols),
        "hyperparams": {
            "n_estimators": n_estimators,
            "threshold": args.threshold,
            "n_folds": args.n_folds,
            "seed": args.seed,
            **best_params,
        },
        "test_metrics": {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in test_metrics.items()
        },
        "cv_results": {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in cv_results.items()
            if not isinstance(v, (list, np.ndarray))
        },
    }

    metrics_path = output_dir / "qsar_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(all_metrics, fh, indent=2)
    logger.info(f"Metrics saved: {metrics_path}")

    importance = model.get_feature_importance(top_n=50)
    importance_path = output_dir / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    logger.info(f"Feature importance saved: {importance_path}")

    # ------------------------------------------------------------------ #
    # Step 14: MLflow logging
    # ------------------------------------------------------------------ #
    _log_mlflow_results(
        test_metrics, all_metrics["cv_results"], model,
        [model_path, metrics_path, importance_path, lc_path, preprocessor_path],
        mlflow_run,
    )

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    if args.task == "classification":
        roc = test_metrics.get("roc_auc", float("nan"))
        cv_roc = all_metrics["cv_results"].get("roc_auc_mean", float("nan"))
        cv_std = all_metrics["cv_results"].get("roc_auc_std", float("nan"))
        logger.info(f"Test ROC-AUC : {roc:.4f}")
        logger.info(f"CV  ROC-AUC  : {cv_roc:.4f} +/- {cv_std:.4f}")
        status = "PASSED" if roc >= 0.75 else "NOT MET"
        logger.info(f"Target (>= 0.75): {status}")
    else:
        r2 = test_metrics.get("r2", float("nan"))
        cv_r2 = all_metrics["cv_results"].get("r2_mean", float("nan"))
        cv_std = all_metrics["cv_results"].get("r2_std", float("nan"))
        logger.info(f"Test R2  : {r2:.4f}")
        logger.info(f"CV   R2  : {cv_r2:.4f} +/- {cv_std:.4f}")

    logger.info(f"Git commit : {prov['git_commit']}")
    logger.info(f"Dataset MD5: {prov['dataset_hash']}")
    logger.info(f"Artefacts  : {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
