"""Molecular generation modules for de novo drug design.

This package provides:
- SMILES VAE for molecular generation
- Diffusion models for molecular generation
- Latent space optimization
- Molecular validity checking
- Property-guided generation

Classes:
    SmilesVAE: Variational Autoencoder for SMILES
    SmilesTokenizer: SMILES string tokenization
    LatentOptimizer: Optimize molecules in latent space
    MoleculeGenerator: High-level generation interface
    MolecularDiffusion: Diffusion model for molecules
"""

from src.generation.tokenizer import SmilesTokenizer, create_vocabulary
from src.generation.vae import SmilesVAE, VAEEncoder, VAEDecoder
from src.generation.optimizer import LatentOptimizer, PropertyPredictor
from src.generation.generator import MoleculeGenerator, validate_smiles, calculate_properties

# Diffusion models (may not be available if dependencies missing)
try:
    from src.generation.diffusion import (
        MolecularDiffusion,
        DiffusionConfig,
        NoiseScheduler,
        CosineScheduler,
        DDPMSampler,
        DDIMSampler,
    )
    HAS_DIFFUSION = True
except ImportError:
    HAS_DIFFUSION = False

__all__ = [
    "SmilesTokenizer",
    "create_vocabulary",
    "SmilesVAE",
    "VAEEncoder",
    "VAEDecoder",
    "LatentOptimizer",
    "PropertyPredictor",
    "MoleculeGenerator",
    "validate_smiles",
    "calculate_properties",
]

if HAS_DIFFUSION:
    __all__.extend([
        "MolecularDiffusion",
        "DiffusionConfig",
        "NoiseScheduler",
        "CosineScheduler",
        "DDPMSampler",
        "DDIMSampler",
    ])
