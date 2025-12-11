#!/usr/bin/env python
"""Train QSAR model for TB drug activity prediction.

This script implements the complete QSAR training pipeline:
1. Load and preprocess ChEMBL data
2. Calculate molecular descriptors
3. Train Random Forest model
4. Evaluate with cross-validation
5. Save model and metrics

Usage:
    python scripts/train_qsar.py --data data/raw/chembl_inhA.csv --output models/

Example:
    python scripts/train_qsar.py \\
        --data data/raw/chembl_inhA.csv \\
        --task classification \\
        --threshold 6.0 \\
        --n-folds 5
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from loguru import logger

from data.chembl_loader import ChEMBLLoader
from data.descriptor_calculator import DescriptorCalculator
from data.data_preprocessor import DataPreprocessor
from models.qsar_model import QSARModel
from evaluation.cross_validation import cross_validate_model
from utils.config import Config
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train QSAR model for TB drug discovery"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/chembl_inhA.csv",
        help="Path to input CSV file",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Output directory for model and metrics",
    )
    
    parser.add_argument(
        "--task",
        type=str,
        choices=["regression", "classification"],
        default="classification",
        help="Task type (regression for pIC50, classification for active/inactive)",
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=6.0,
        help="pIC50 threshold for activity classification",
    )
    
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in Random Forest",
    )
    
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Setup logging
    setup_logger(level="INFO")
    logger.info("=" * 60)
    logger.info("TB Drug Discovery - QSAR Training Pipeline")
    logger.info("=" * 60)
    
    # Load configuration
    config = Config(args.config)
    config.set("random_seed", args.seed)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== Step 1: Load Data =====
    logger.info("Step 1: Loading data...")
    
    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        logger.info("Please download ChEMBL InhA data first.")
        logger.info("See notebooks/01_data_loading.ipynb for instructions.")
        sys.exit(1)
    
    loader = ChEMBLLoader(random_seed=args.seed)
    df_raw = loader.load_from_csv(args.data)
    logger.info(f"Loaded {len(df_raw)} records")
    
    # ===== Step 2: Preprocess Data =====
    logger.info("Step 2: Preprocessing data...")
    
    df_clean = loader.preprocess(df_raw)
    
    if args.task == "classification":
        df_clean = loader.create_activity_labels(df_clean, threshold=args.threshold)
    
    stats = loader.get_statistics(df_clean)
    logger.info(f"Dataset statistics: {stats}")
    
    # Save processed data
    processed_path = output_dir.parent / "data" / "processed" / "cleaned_chembl_inhA.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    loader.save_processed(df_clean, str(processed_path))
    
    # ===== Step 3: Calculate Descriptors =====
    logger.info("Step 3: Calculating molecular descriptors...")
    
    calculator = DescriptorCalculator(
        lipinski=True,
        topological=True,
        extended=True,
    )
    
    df_with_desc = calculator.calculate_from_dataframe(df_clean, smiles_col="smiles")
    logger.info(f"Calculated {len(calculator.descriptor_names)} descriptors")
    
    # Save descriptors
    desc_path = output_dir.parent / "data" / "processed" / "descriptors.csv"
    df_with_desc.to_csv(desc_path, index=False)
    
    # ===== Step 4: Prepare for Training =====
    logger.info("Step 4: Preparing data for training...")
    
    feature_cols = calculator.descriptor_names
    target_col = "active" if args.task == "classification" else "pIC50"
    
    # Drop rows with missing values
    df_train = df_with_desc.dropna(subset=feature_cols + [target_col])
    logger.info(f"Training samples: {len(df_train)}")
    
    X = df_train[feature_cols].values
    y = df_train[target_col].values
    
    # Split data
    preprocessor = DataPreprocessor(random_seed=args.seed)
    X_train, X_test, y_train, y_test = preprocessor.split_data_simple(
        X, y, test_size=0.2, stratify=(args.task == "classification")
    )
    
    # Scale features
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Save scaler
    scaler_path = output_dir / "qsar_scaler.pkl"
    preprocessor.save(str(scaler_path))
    
    # ===== Step 5: Train Model =====
    logger.info("Step 5: Training QSAR model...")
    
    model = QSARModel(
        task=args.task,
        n_estimators=args.n_estimators,
        random_seed=args.seed,
    )
    
    model.fit(X_train_scaled, y_train, feature_names=feature_cols)
    
    # ===== Step 6: Evaluate =====
    logger.info("Step 6: Evaluating model...")
    
    # Test set evaluation
    test_metrics = model.evaluate(X_test_scaled, y_test)
    logger.info(f"Test set metrics: {test_metrics}")
    
    # Cross-validation
    logger.info(f"Running {args.n_folds}-fold cross-validation...")
    cv_results = cross_validate_model(
        model.model,
        X_train_scaled,
        y_train,
        n_folds=args.n_folds,
        task=args.task,
        random_seed=args.seed,
    )
    
    # ===== Step 7: Save Results =====
    logger.info("Step 7: Saving results...")
    
    # Save model
    model_path = output_dir / "qsar_rf_model.pkl"
    model.save(str(model_path))
    
    # Save metrics
    all_metrics = {
        "task": args.task,
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test),
        "n_features": len(feature_cols),
        "test_metrics": test_metrics,
        "cv_results": {
            k: v for k, v in cv_results.items()
            if not isinstance(v, (list, np.ndarray))
        },
        "config": {
            "n_estimators": args.n_estimators,
            "n_folds": args.n_folds,
            "threshold": args.threshold,
            "seed": args.seed,
        },
    }
    
    metrics_path = output_dir / "qsar_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    # Save feature importance
    importance = model.get_feature_importance(top_n=20)
    importance_path = output_dir / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    
    # ===== Summary =====
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    
    if args.task == "classification":
        logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"CV ROC-AUC: {cv_results['roc_auc_mean']:.4f} ± {cv_results['roc_auc_std']:.4f}")
        
        # Check target metric
        target_met = test_metrics['roc_auc'] >= 0.75
        logger.info(f"Target (ROC-AUC >= 0.75): {'✅ PASSED' if target_met else '❌ NOT MET'}")
    else:
        logger.info(f"Test R²: {test_metrics['r2']:.4f}")
        logger.info(f"CV R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
    
    logger.info(f"\nArtifacts saved to: {output_dir}")
    logger.info(f"  - Model: {model_path.name}")
    logger.info(f"  - Metrics: {metrics_path.name}")
    logger.info(f"  - Feature importance: {importance_path.name}")
    logger.info(f"  - Scaler: {scaler_path.name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
