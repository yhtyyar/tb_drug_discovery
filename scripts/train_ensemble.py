#!/usr/bin/env python
"""Train ensemble QSAR models with hyperparameter optimization.

This script trains and evaluates ensemble models for QSAR prediction,
optionally using Optuna for hyperparameter optimization.

Usage:
    python scripts/train_ensemble.py --data data/processed/descriptors.csv
    python scripts/train_ensemble.py --data data.csv --optimize --n-trials 50
    python scripts/train_ensemble.py --ensemble stacking --task classification

Examples:
    # Train with default settings
    python scripts/train_ensemble.py --data data/processed/descriptors.csv

    # Train with hyperparameter optimization
    python scripts/train_ensemble.py --data data.csv --optimize --n-trials 100

    # Train specific ensemble type
    python scripts/train_ensemble.py --data data.csv --ensemble voting
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ensemble import (
    VotingEnsemble,
    StackingEnsemble,
    BlendingEnsemble,
    create_ensemble,
    train_best_ensemble,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ensemble QSAR models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to training data CSV",
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        default="pIC50",
        help="Target column name (default: pIC50)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "regression"],
        default="classification",
        help="Task type (default: classification)",
    )
    parser.add_argument(
        "--ensemble",
        type=str,
        choices=["voting", "stacking", "blending", "auto"],
        default="auto",
        help="Ensemble type (default: auto - selects best)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run hyperparameter optimization with Optuna",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (default: 50)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction (default: 0.2)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation set fraction (default: 0.15)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="models/ensemble",
        help="Output directory (default: models/ensemble)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds for stacking (default: 5)",
    )
    
    return parser.parse_args()


def load_data(data_path: str, target_col: str):
    """Load and prepare data."""
    logger.info(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {df.columns.tolist()}")
    
    # Separate features and target
    y = df[target_col].values
    
    # Drop non-numeric and target columns
    feature_cols = [c for c in df.columns if c != target_col and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    X = df[feature_cols].values
    
    logger.info(f"Loaded {len(df)} samples with {len(feature_cols)} features")
    
    return X, y, feature_cols


def run_hyperparameter_optimization(X_train, y_train, task, n_trials):
    """Run Optuna hyperparameter optimization."""
    try:
        from src.models.hyperopt import QSAROptimizer, OptimizationConfig
    except ImportError:
        logger.warning("Optuna not available, skipping optimization")
        return None
    
    logger.info(f"Running hyperparameter optimization with {n_trials} trials")
    
    config = OptimizationConfig(
        n_trials=n_trials,
        cv_folds=3,
        metric="roc_auc" if task == "classification" else "r2",
    )
    
    optimizer = QSAROptimizer(config)
    best_params = optimizer.optimize_all(X_train, y_train, task)
    
    return best_params


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    X, y, feature_names = load_data(args.data, args.target)
    
    # Convert to binary classification if needed
    if args.task == "classification" and y.dtype == 'float64':
        threshold = 6.0  # pIC50 threshold
        y = (y >= threshold).astype(int)
        logger.info(f"Converted to classification with threshold={threshold}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y if args.task == "classification" else None
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=args.val_size, random_state=args.seed, stratify=y_train if args.task == "classification" else None
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Hyperparameter optimization
    best_params = None
    if args.optimize:
        best_params = run_hyperparameter_optimization(X_train, y_train, args.task, args.n_trials)
        
        if best_params:
            with open(output_dir / "best_hyperparams.json", "w") as f:
                json.dump(best_params, f, indent=2)
            logger.info(f"Best hyperparameters saved to {output_dir / 'best_hyperparams.json'}")
    
    # Train ensemble
    if args.ensemble == "auto":
        logger.info("Selecting best ensemble type automatically...")
        ensemble, metrics = train_best_ensemble(X_train, y_train, X_val, y_val, task=args.task)
    else:
        logger.info(f"Training {args.ensemble} ensemble...")
        ensemble = create_ensemble(args.ensemble, task=args.task, cv_folds=args.cv_folds)
        ensemble.fit(X_train, y_train)
        metrics = ensemble.evaluate(X_val, y_val)
    
    # Evaluate on test set
    test_metrics = ensemble.evaluate(X_test, y_test)
    
    # Log results
    logger.info("=" * 50)
    logger.info("VALIDATION METRICS:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    logger.info("TEST METRICS:")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info("=" * 50)
    
    # Save model and results
    ensemble.save(str(output_dir / "ensemble_model.pkl"))
    
    results = {
        "ensemble_type": type(ensemble).__name__,
        "task": args.task,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "n_features": X.shape[1],
        "validation_metrics": metrics,
        "test_metrics": test_metrics,
    }
    
    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Model and results saved to {output_dir}")
    
    # Get individual model scores
    if hasattr(ensemble, 'get_model_scores'):
        model_scores = ensemble.get_model_scores(X_test, y_test)
        logger.info("\nIndividual model scores:")
        for name, score in sorted(model_scores.items(), key=lambda x: -x[1]):
            logger.info(f"  {name}: {score:.4f}")


if __name__ == "__main__":
    main()
