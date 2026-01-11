#!/usr/bin/env python
"""Train Molecular Diffusion model for de novo drug design.

This script trains a diffusion model on molecular latent representations
for generating novel drug-like molecules.

Usage:
    python scripts/train_diffusion.py --data data/processed/smiles.csv
    python scripts/train_diffusion.py --vae-path models/vae/best_model.pt --epochs 100

Examples:
    # Train with default settings
    python scripts/train_diffusion.py --data data/processed/smiles.csv

    # Train with custom configuration
    python scripts/train_diffusion.py --data data.csv --latent-dim 256 --epochs 200
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Molecular Diffusion model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--data", "-d",
        type=str,
        help="Path to SMILES data CSV (requires VAE for encoding)",
    )
    parser.add_argument(
        "--latent-data",
        type=str,
        help="Path to pre-computed latent vectors (.npy)",
    )
    parser.add_argument(
        "--vae-path",
        type=str,
        help="Path to trained VAE model (for encoding SMILES)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="Latent dimension (default: 256)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension (default: 512)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of denoising layers (default: 4)",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps (default: 1000)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["linear", "cosine", "sigmoid"],
        default="cosine",
        help="Noise scheduler type (default: cosine)",
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
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="models/diffusion",
        help="Output directory (default: models/diffusion)",
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
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Use exponential moving average",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate after training (default: 100)",
    )
    
    return parser.parse_args()


def load_latent_data(latent_path: str) -> np.ndarray:
    """Load pre-computed latent vectors."""
    logger.info(f"Loading latent vectors from {latent_path}")
    return np.load(latent_path)


def encode_smiles_to_latent(smiles_list, vae_path: str, device: torch.device) -> np.ndarray:
    """Encode SMILES to latent vectors using VAE."""
    from src.generation.vae import SmilesVAE
    from src.generation.tokenizer import SmilesTokenizer
    
    logger.info(f"Loading VAE from {vae_path}")
    vae = SmilesVAE.load(vae_path)
    vae.to(device)
    vae.eval()
    
    # Create tokenizer and encode
    tokenizer = SmilesTokenizer()
    tokenizer.fit(smiles_list)
    
    latents = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i+batch_size]
            encoded = tokenizer.encode_batch(batch_smiles)
            encoded = torch.LongTensor(encoded).to(device)
            
            mu, _ = vae.encode(encoded)
            latents.append(mu.cpu().numpy())
    
    return np.concatenate(latents, axis=0)


def create_synthetic_latents(n_samples: int, latent_dim: int) -> np.ndarray:
    """Create synthetic latent data for demo/testing."""
    logger.info(f"Creating {n_samples} synthetic latent vectors")
    return np.random.randn(n_samples, latent_dim).astype(np.float32)


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
    
    # Load or create latent data
    if args.latent_data:
        latents = load_latent_data(args.latent_data)
    elif args.data and args.vae_path:
        df = pd.read_csv(args.data)
        smiles_col = "SMILES" if "SMILES" in df.columns else "smiles"
        smiles_list = df[smiles_col].tolist()
        latents = encode_smiles_to_latent(smiles_list, args.vae_path, device)
    else:
        logger.warning("No data provided, using synthetic latent data for demo")
        latents = create_synthetic_latents(1000, args.latent_dim)
    
    latent_dim = latents.shape[1]
    logger.info(f"Loaded {len(latents)} latent vectors of dimension {latent_dim}")
    
    # Create dataloader
    train_data = torch.FloatTensor(latents)
    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Validation split
    val_size = int(0.1 * len(latents))
    val_data = train_data[-val_size:]
    train_data = train_data[:-val_size]
    val_dataset = TensorDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model
    from src.generation.diffusion import DiffusionConfig, MolecularDiffusion
    from src.generation.diffusion.mol_diffusion import train_diffusion_model
    
    config = DiffusionConfig(
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_timesteps=args.num_timesteps,
        scheduler_type=args.scheduler,
        learning_rate=args.lr,
    )
    
    model = MolecularDiffusion(config)
    model.to(device)
    
    logger.info(f"Model config: {config}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    history = train_diffusion_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=str(output_dir),
        use_ema=args.use_ema,
    )
    
    # Generate samples
    logger.info(f"Generating {args.num_samples} samples...")
    model.eval()
    
    samples = model.generate(
        num_samples=args.num_samples,
        num_inference_steps=50,
        sampler_type="ddim",
    )
    
    # Save samples
    np.save(output_dir / "generated_latents.npy", samples.cpu().numpy())
    
    # Save training results
    results = {
        "config": {
            "latent_dim": config.latent_dim,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "num_timesteps": config.num_timesteps,
            "scheduler_type": config.scheduler_type,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "train_samples": len(train_data),
            "val_samples": len(val_data),
        },
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
    }
    
    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Training complete. Model saved to {output_dir}")
    logger.info(f"Final train loss: {results['final_train_loss']:.4f}")
    if results['final_val_loss']:
        logger.info(f"Final val loss: {results['final_val_loss']:.4f}")


if __name__ == "__main__":
    main()
