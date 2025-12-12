#!/usr/bin/env python
"""VAE training script for molecular generation.

Train a Variational Autoencoder on SMILES data for de novo drug design.

Usage:
    python scripts/train_vae.py --data data/processed/cleaned_chembl_inhA.csv
    
    python scripts/train_vae.py --data data/raw/chembl_1849_ic50.csv \\
        --latent-dim 256 --epochs 100 --batch-size 64

Requirements:
    - PyTorch
    - RDKit
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from loguru import logger

from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SMILES VAE for molecular generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python scripts/train_vae.py --data data/processed/cleaned_chembl_inhA.csv
    
    # Custom architecture
    python scripts/train_vae.py --data data.csv --latent-dim 512 --hidden-dim 1024
    
    # Generate molecules after training
    python scripts/train_vae.py --data data.csv --generate 1000
        """
    )
    
    # Data
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV with SMILES"
    )
    parser.add_argument(
        "--smiles-col",
        type=str,
        default="smiles",
        help="Column name for SMILES (default: smiles)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=120,
        help="Maximum SMILES length (default: 120)"
    )
    
    # Model architecture
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="Latent space dimension (default: 256)"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="RNN hidden dimension (default: 512)"
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=128,
        help="Embedding dimension (default: 128)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of RNN layers (default: 2)"
    )
    
    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--kl-annealing",
        action="store_true",
        default=True,
        help="Use KL annealing (default: True)"
    )
    parser.add_argument(
        "--kl-warmup",
        type=int,
        default=10,
        help="KL annealing warmup epochs (default: 10)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    # Generation
    parser.add_argument(
        "--generate",
        type=int,
        default=100,
        help="Number of molecules to generate after training (default: 100)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="models/vae",
        help="Output directory (default: models/vae)"
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Setup
    setup_logger(log_file="logs/vae_training.log")
    set_seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("SMILES VAE TRAINING")
    logger.info("=" * 60)
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Import modules
    from src.generation.tokenizer import SmilesTokenizer, get_smiles_statistics
    from src.generation.vae import SmilesVAE
    from src.generation.generator import MoleculeGenerator, validate_smiles
    
    # Step 1: Load data
    logger.info("\nStep 1: Loading data...")
    
    df = pd.read_csv(args.data)
    smiles_list = df[args.smiles_col].dropna().tolist()
    
    # Filter valid SMILES
    valid_smiles = [s for s in smiles_list if validate_smiles(s)]
    
    # Filter by length
    valid_smiles = [s for s in valid_smiles if len(s) <= args.max_length * 2]
    
    logger.info(f"Loaded {len(smiles_list)} SMILES, {len(valid_smiles)} valid")
    
    # Statistics
    stats = get_smiles_statistics(valid_smiles)
    logger.info(f"Average length: {stats['avg_length']:.1f} tokens")
    logger.info(f"Max length: {stats['max_length']} tokens")
    logger.info(f"Unique tokens: {stats['unique_tokens']}")
    
    # Step 2: Create tokenizer
    logger.info("\nStep 2: Creating tokenizer...")
    
    tokenizer = SmilesTokenizer(max_length=args.max_length)
    tokenizer.fit(valid_smiles)
    
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Step 3: Create model
    logger.info("\nStep 3: Creating VAE model...")
    
    vae = SmilesVAE(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        max_length=args.max_length,
    )
    
    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Step 4: Train
    logger.info("\nStep 4: Training...")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = MoleculeGenerator(tokenizer, vae)
    
    history = generator.train(
        smiles_list=valid_smiles,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        kl_annealing=args.kl_annealing,
        kl_warmup=args.kl_warmup,
        checkpoint_dir=str(output_dir),
        verbose=1,
    )
    
    # Step 5: Evaluate reconstruction
    logger.info("\nStep 5: Evaluating reconstruction...")
    
    sample_smiles = valid_smiles[:100]
    reconstructed, recon_accuracy = generator.reconstruct(sample_smiles)
    logger.info(f"Reconstruction accuracy: {recon_accuracy:.2%}")
    
    # Step 6: Generate molecules
    logger.info("\nStep 6: Generating molecules...")
    
    generated = generator.generate(
        num_samples=args.generate,
        temperature=args.temperature,
        unique=True,
        valid_only=True,
    )
    
    # Evaluate generation
    metrics = generator.evaluate(generated, reference_smiles=valid_smiles)
    
    logger.info(f"\nGeneration metrics:")
    logger.info(f"  Validity:   {metrics.get('validity', 0):.2%}")
    logger.info(f"  Uniqueness: {metrics.get('uniqueness', 0):.2%}")
    logger.info(f"  Novelty:    {metrics.get('novelty', 0):.2%}")
    logger.info(f"  Avg QED:    {metrics.get('avg_qed', 0):.3f}")
    
    # Step 7: Save results
    logger.info("\nStep 7: Saving results...")
    
    # Save tokenizer
    tokenizer.save(str(output_dir / "tokenizer.json"))
    
    # Save model
    generator.save(str(output_dir / "vae_model.pt"))
    
    # Save generated molecules
    gen_df = pd.DataFrame({'smiles': generated})
    gen_df.to_csv(output_dir / "generated_molecules.csv", index=False)
    
    # Save metrics
    results = {
        'training': {
            'epochs': args.epochs,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1],
        },
        'reconstruction_accuracy': recon_accuracy,
        'generation_metrics': metrics,
        'hyperparameters': {
            'latent_dim': args.latent_dim,
            'hidden_dim': args.hidden_dim,
            'embed_dim': args.embed_dim,
            'num_layers': args.num_layers,
            'max_length': args.max_length,
        },
    }
    
    with open(output_dir / "vae_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Generated {len(generated)} valid molecules")
    
    target_validity = metrics.get('validity', 0) >= 0.90
    logger.info(f"Target (validity >= 90%): {'✅ PASSED' if target_validity else '❌ NOT MET'}")
    
    logger.info(f"\nArtifacts saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
