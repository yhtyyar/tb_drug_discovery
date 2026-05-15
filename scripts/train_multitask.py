"""Train the Multi-task QSAR model across five TB targets.

Usage
-----
python scripts/train_multitask.py --data data/multitask.csv --targets InhA KatG rpoB

The CSV should have columns: smiles, InhA, KatG, rpoB, DprE1, MmpL3 (any subset).
Values are pIC50; missing = NaN. Compounds tested in ≥1 target are included.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multi-task QSAR model")
    p.add_argument("--data", default="data/multitask.csv", help="Path to multi-target CSV")
    p.add_argument(
        "--targets",
        nargs="+",
        default=["InhA", "KatG", "rpoB", "DprE1", "MmpL3"],
        help="Targets to include",
    )
    p.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 256, 128])
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="models/multitask", help="Output directory")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data for testing")
    p.add_argument("--mlflow", action="store_true", help="Log metrics to MLflow")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ Data
    from src.data.multitask_loader import (
        MultiTaskDataset,
        compute_descriptors,
        load_multitask_csv,
        make_synthetic_multitask_dataset,
    )

    if args.synthetic or not Path(args.data).exists():
        logger.info("Using synthetic dataset (500 compounds)")
        dataset = make_synthetic_multitask_dataset(
            n_compounds=500, targets=args.targets, seed=args.seed
        )
    else:
        dataset = load_multitask_csv(args.data, target_cols=args.targets)
        dataset = dataset.filter_min_targets(min_targets=1)

    logger.info("Dataset: %d compounds × %d targets", len(dataset), dataset.n_targets)
    cov_df = dataset.coverage()
    logger.info("\nData coverage:\n%s", cov_df.to_string())

    # -------------------------------------------------------- Scaffold split
    train_ds, val_ds, test_ds = dataset.train_val_test_split(
        frac_train=0.70, frac_val=0.10, frac_test=0.20,
        scaffold_split=True, seed=args.seed,
    )
    logger.info(
        "Split: train=%d  val=%d  test=%d",
        len(train_ds), len(val_ds), len(test_ds),
    )

    # --------------------------------------------------- Descriptor computation
    logger.info("Computing Morgan fingerprints (r=2, 2048 bits)…")
    X_train = compute_descriptors(train_ds)
    X_val = compute_descriptors(val_ds)
    X_test = compute_descriptors(test_ds)

    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    import joblib

    joblib.dump(scaler, output_dir / "scaler.joblib")

    # --------------------------------------------------------- MLflow setup
    run = None
    if args.mlflow:
        try:
            import mlflow

            mlflow.set_experiment("multitask_qsar")
            run = mlflow.start_run()
            mlflow.log_params(
                {
                    "targets": ",".join(args.targets),
                    "hidden_dims": str(args.hidden_dims),
                    "dropout": args.dropout,
                    "epochs": args.epochs,
                    "seed": args.seed,
                    "n_train": len(train_ds),
                    "n_val": len(val_ds),
                    "n_test": len(test_ds),
                }
            )
        except ImportError:
            logger.warning("MLflow not installed — skipping logging")

    # ------------------------------------------------------- Model training
    from src.models.multitask_qsar import MultiTaskConfig, MultiTaskQSAR, evaluate_multitask

    config = MultiTaskConfig(
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_seed=args.seed,
    )

    model = MultiTaskQSAR(targets=args.targets, config=config)
    logger.info("Training model…")
    model.fit(
        X_train, train_ds.y_dict,
        X_val=X_val, y_val_dict=val_ds.y_dict,
    )

    # ---------------------------------------------------------- Evaluation
    logger.info("\n=== Validation metrics ===")
    val_metrics_df = evaluate_multitask(model, X_val, val_ds.y_dict)
    if not val_metrics_df.empty:
        logger.info("\n%s", val_metrics_df.to_string())

    logger.info("\n=== Test metrics ===")
    test_metrics_df = evaluate_multitask(model, X_test, test_ds.y_dict)
    if not test_metrics_df.empty:
        logger.info("\n%s", test_metrics_df.to_string())

    # ------------------------------------------------- Uncertainty on test set
    uncertainty = model.predict_uncertainty(X_test)
    mean_unc = {t: float(v.mean()) for t, v in uncertainty.items()}
    logger.info("Mean epistemic uncertainty: %s", mean_unc)

    # -------------------------------------------------------- Save artifacts
    model.save(str(output_dir / "multitask_qsar.joblib"))

    metrics_out: dict = {
        "validation": val_metrics_df.to_dict(orient="index") if not val_metrics_df.empty else {},
        "test": test_metrics_df.to_dict(orient="index") if not test_metrics_df.empty else {},
        "mean_uncertainty": mean_unc,
    }
    with open(output_dir / "multitask_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # Log to MLflow
    if run is not None:
        for target, row in test_metrics_df.iterrows():
            for metric, val in row.items():
                if isinstance(val, float):
                    mlflow.log_metric(f"test_{target}_{metric}", val)
        mlflow.end_run()

    logger.info("Artifacts saved to %s", output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
