#!/usr/bin/env python
"""Train Graph Transformer (GraphGPS) for molecular property prediction.

This script trains Graph Transformer models on molecular graph data
for QSAR/property prediction tasks.

Usage:
    python scripts/train_graph_transformer.py --data data/processed/molecules.csv
    python scripts/train_graph_transformer.py --model gps --hidden-dim 256 --epochs 100

Examples:
    # Train GraphGPS with default settings
    python scripts/train_graph_transformer.py --data data.csv --target pIC50

    # Train lightweight transformer
    python scripts/train_graph_transformer.py --data data.csv --model light --epochs 50
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Graph Transformer for molecular property prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to molecular data CSV with SMILES column",
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        default="pIC50",
        help="Target column name (default: pIC50)",
    )
    parser.add_argument(
        "--smiles-col",
        type=str,
        default="SMILES",
        help="SMILES column name (default: SMILES)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gps", "light"],
        default="gps",
        help="Model type: gps (GraphGPS) or light (default: gps)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "regression"],
        default="classification",
        help="Task type (default: classification)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension (default: 256)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer layers (default: 4)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)",
    )
    parser.add_argument(
        "--local-gnn",
        type=str,
        choices=["gine", "gat", "none"],
        default="gine",
        help="Local GNN type for GPS (default: gine)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Test set fraction (default: 0.15)",
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
        default="models/graph_transformer",
        help="Output directory (default: models/graph_transformer)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: cuda, cpu, or auto (default: auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    return parser.parse_args()


def load_and_featurize_data(data_path: str, smiles_col: str, target_col: str, task: str):
    """Load data and convert to molecular graphs."""
    from src.gnn.featurizer import MolecularGraphFeaturizer, MoleculeDataset
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    if smiles_col not in df.columns:
        # Try lowercase
        smiles_col = smiles_col.lower()
        if smiles_col not in df.columns:
            raise ValueError(f"SMILES column not found. Available: {df.columns.tolist()}")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    smiles_list = df[smiles_col].tolist()
    targets = df[target_col].values
    
    # Convert to binary for classification
    if task == "classification" and targets.dtype == 'float64':
        threshold = 6.0
        targets = (targets >= threshold).astype(float)
        logger.info(f"Converted to classification with threshold={threshold}")
    
    # Create molecular graphs
    logger.info("Featurizing molecules...")
    featurizer = MolecularGraphFeaturizer()
    
    valid_graphs = []
    valid_targets = []
    
    for i, smiles in enumerate(smiles_list):
        try:
            graph = featurizer.featurize(smiles)
            if graph is not None:
                graph.y = torch.tensor([targets[i]], dtype=torch.float)
                valid_graphs.append(graph)
                valid_targets.append(targets[i])
        except Exception as e:
            logger.debug(f"Failed to featurize {smiles}: {e}")
    
    logger.info(f"Successfully featurized {len(valid_graphs)}/{len(smiles_list)} molecules")
    
    # Get feature dimensions
    node_dim = valid_graphs[0].x.shape[1]
    edge_dim = valid_graphs[0].edge_attr.shape[1] if valid_graphs[0].edge_attr is not None else 0
    
    return valid_graphs, node_dim, edge_dim


def split_dataset(graphs, test_size: float, val_size: float, seed: int):
    """Split graphs into train/val/test sets."""
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    train_val, test = train_test_split(graphs, test_size=test_size, random_state=seed)
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=seed)
    
    return train, val, test


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and featurize data
    graphs, node_dim, edge_dim = load_and_featurize_data(
        args.data, args.smiles_col, args.target, args.task
    )
    
    # Split data
    train_graphs, val_graphs, test_graphs = split_dataset(
        graphs, args.test_size, args.val_size, args.seed
    )
    logger.info(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    # Create dataloaders
    from torch_geometric.loader import DataLoader
    
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)
    
    # Create model
    from src.gnn.graph_transformer import create_graph_transformer
    
    local_gnn = args.local_gnn if args.local_gnn != "none" else None
    
    model = create_graph_transformer(
        model_type=args.model,
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        task=args.task,
        local_gnn=local_gnn,
    )
    model.to(device)
    
    logger.info(f"Model: {args.model.upper()}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    from src.gnn.trainer import GNNTrainer, EarlyStopping
    
    trainer = GNNTrainer(
        model=model,
        task=args.task,
        learning_rate=args.lr,
        device=str(device),
    )
    
    early_stopping = EarlyStopping(patience=args.patience, mode="min")
    
    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping=early_stopping,
        checkpoint_dir=str(output_dir),
    )
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader)
    
    # Log results
    logger.info("=" * 50)
    logger.info("TEST METRICS:")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info("=" * 50)
    
    # Save final model
    trainer.save_model(str(output_dir / "final_model.pt"))
    
    # Save results
    results = {
        "model_type": args.model,
        "config": {
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "local_gnn": args.local_gnn,
        },
        "training": {
            "epochs_trained": len(history["train_loss"]),
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "train_samples": len(train_graphs),
            "val_samples": len(val_graphs),
            "test_samples": len(test_graphs),
        },
        "test_metrics": test_metrics,
        "best_val_loss": min(history["val_loss"]) if history["val_loss"] else None,
    }
    
    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
