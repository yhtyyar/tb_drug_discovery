#!/usr/bin/env python
"""GNN training pipeline for TB drug discovery.

This script trains graph neural network models for molecular property prediction.

Usage:
    python scripts/train_gnn.py --data data/processed/cleaned_chembl_inhA.csv --model gat
    
    python scripts/train_gnn.py --data data/processed/cleaned_chembl_inhA.csv \\
        --model gcn --hidden-dim 256 --num-layers 4 --epochs 200

Requirements:
    - PyTorch
    - PyTorch Geometric
    - RDKit
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from loguru import logger

from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GNN models for molecular property prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train GCN model
    python scripts/train_gnn.py --data data/processed/cleaned_chembl_inhA.csv --model gcn
    
    # Train GAT with custom hyperparameters
    python scripts/train_gnn.py --data data/processed/cleaned_chembl_inhA.csv \\
        --model gat --hidden-dim 256 --num-heads 8 --epochs 200
    
    # Train MPNN (uses edge features)
    python scripts/train_gnn.py --data data/processed/cleaned_chembl_inhA.csv --model mpnn
    
    # Train ensemble combining QSAR + GNN
    python scripts/train_gnn.py --data data/processed/cleaned_chembl_inhA.csv \\
        --model gat --ensemble --qsar-model models/qsar_rf_model.pkl

Available models: gcn, gat, mpnn, attentivefp
        """
    )
    
    # Data
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV with SMILES and targets"
    )
    parser.add_argument(
        "--smiles-col",
        type=str,
        default="smiles",
        help="Column name for SMILES (default: smiles)"
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="active",
        help="Column name for target (default: active)"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "regression"],
        default="classification",
        help="Task type (default: classification)"
    )
    
    # Model
    parser.add_argument(
        "--model",
        type=str,
        choices=["gcn", "gat", "mpnn", "attentivefp"],
        default="gat",
        help="GNN model type (default: gat)"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden layer dimension (default: 128)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of GNN layers (default: 3)"
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads for GAT (default: 4)"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability (default: 0.2)"
    )
    
    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (default: 15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    # Ensemble
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Train ensemble with QSAR model"
    )
    parser.add_argument(
        "--qsar-model",
        type=str,
        default="models/qsar_rf_model.pkl",
        help="Path to QSAR model for ensemble"
    )
    parser.add_argument(
        "--qsar-scaler",
        type=str,
        default="models/qsar_scaler.pkl",
        help="Path to QSAR scaler for ensemble"
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="models/gnn",
        help="Output directory (default: models/gnn)"
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Setup
    setup_logger(log_file="logs/gnn_training.log")
    set_seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("GNN TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model.upper()}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Check PyTorch Geometric
    try:
        from torch_geometric.loader import DataLoader
        from src.gnn.featurizer import MolecularGraphFeaturizer, create_data_loaders
        from src.gnn.models import create_model
        from src.gnn.trainer import GNNTrainer, EarlyStopping
    except ImportError as e:
        logger.error(f"PyTorch Geometric not installed: {e}")
        logger.error("Install with: pip install torch-geometric")
        return 1
    
    # Step 1: Load data
    logger.info("\nStep 1: Loading data...")
    
    df = pd.read_csv(args.data)
    logger.info(f"Loaded {len(df)} samples from {args.data}")
    
    # Get SMILES and targets
    smiles_list = df[args.smiles_col].dropna().tolist()
    
    if args.target_col not in df.columns:
        logger.error(f"Target column '{args.target_col}' not found")
        return 1
    
    targets = df.loc[df[args.smiles_col].notna(), args.target_col].tolist()
    
    logger.info(f"Valid samples: {len(smiles_list)}")
    
    if args.task == "classification":
        class_counts = pd.Series(targets).value_counts()
        logger.info(f"Class distribution: {dict(class_counts)}")
    
    # Step 2: Create DataLoaders
    logger.info("\nStep 2: Creating molecular graphs...")
    
    train_loader, val_loader, test_loader, featurizer = create_data_loaders(
        smiles_list=smiles_list,
        y_list=targets,
        train_ratio=0.8,
        val_ratio=0.1,
        batch_size=args.batch_size,
        random_seed=args.seed,
    )
    
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    logger.info(f"Node features: {featurizer.atom_dim}, Edge features: {featurizer.bond_dim}")
    
    # Step 3: Create model
    logger.info("\nStep 3: Creating model...")
    
    model_kwargs = {
        "node_dim": featurizer.atom_dim,
        "edge_dim": featurizer.bond_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "task": args.task,
    }
    
    if args.model == "gat":
        model_kwargs["num_heads"] = args.num_heads
    
    model = create_model(model_type=args.model, **model_kwargs)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Step 4: Train
    logger.info("\nStep 4: Training...")
    
    trainer = GNNTrainer(
        model=model,
        task=args.task,
        learning_rate=args.learning_rate,
    )
    
    early_stopping = EarlyStopping(patience=args.patience, mode='min')
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping=early_stopping,
        checkpoint_dir=str(output_dir),
        verbose=1,
    )
    
    # Step 5: Evaluate
    logger.info("\nStep 5: Evaluating on test set...")
    
    test_metrics = trainer.evaluate(test_loader)
    
    # Step 6: Ensemble (optional)
    ensemble_metrics = None
    if args.ensemble:
        logger.info("\nStep 6: Creating ensemble with QSAR...")
        
        try:
            from src.gnn.ensemble import EnsembleModel
            from src.models import QSARModel
            from src.data import DataPreprocessor, DescriptorCalculator
            
            # Load QSAR components
            qsar_model = QSARModel.load(args.qsar_model)
            preprocessor = DataPreprocessor.load(args.qsar_scaler)
            calculator = DescriptorCalculator()
            
            # Create ensemble
            ensemble = EnsembleModel(
                qsar_model=qsar_model.model,
                gnn_model=model,
                featurizer=featurizer,
                preprocessor=preprocessor,
                strategy='weighted',
                task=args.task,
            )
            
            # Get test data for evaluation
            test_smiles = [data.smiles for data in test_loader.dataset]
            test_targets = [data.y.item() for data in test_loader.dataset]
            
            # Compute descriptors
            df_test = pd.DataFrame({'smiles': test_smiles})
            df_desc = calculator.calculate_from_dataframe(df_test, smiles_col='smiles')
            X_test = df_desc[calculator.descriptor_names].values
            X_test_scaled = preprocessor.transform(X_test)
            
            # Optimize weights
            best_weights, best_score = ensemble.optimize_weights(
                test_smiles, X_test_scaled, np.array(test_targets)
            )
            
            # Evaluate ensemble
            ensemble_metrics = ensemble.evaluate(
                test_smiles, X_test_scaled, np.array(test_targets)
            )
            
            # Save ensemble
            ensemble.save(str(output_dir / "ensemble"))
            
        except Exception as e:
            logger.warning(f"Ensemble creation failed: {e}")
    
    # Step 7: Save results
    logger.info("\nStep 7: Saving results...")
    
    # Save model
    trainer.save_model(str(output_dir / f"{args.model}_model.pt"))
    
    # Save metrics
    results = {
        "model_type": args.model,
        "task": args.task,
        "hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
        },
        "training": {
            "epochs_trained": len(history['train_loss']),
            "best_val_loss": min(history['val_loss']),
            "final_train_loss": history['train_loss'][-1],
        },
        "test_metrics": test_metrics,
    }
    
    if ensemble_metrics:
        results["ensemble_metrics"] = ensemble_metrics
    
    with open(output_dir / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    
    if args.task == "classification":
        logger.info(f"Test ROC-AUC: {test_metrics.get('roc_auc', 0):.4f}")
        logger.info(f"Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
        
        target_met = test_metrics.get('roc_auc', 0) >= 0.75
        logger.info(f"Target (ROC-AUC >= 0.75): {'✅ PASSED' if target_met else '❌ NOT MET'}")
        
        if ensemble_metrics:
            logger.info(f"\nEnsemble ROC-AUC: {ensemble_metrics.get('ensemble_roc_auc', 0):.4f}")
            logger.info(f"Improvement over GNN: {ensemble_metrics.get('improvement_over_gnn', 0):+.4f}")
    else:
        logger.info(f"Test R²: {test_metrics.get('r2', 0):.4f}")
        logger.info(f"Test RMSE: {test_metrics.get('rmse', 0):.4f}")
    
    logger.info(f"\nArtifacts saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
