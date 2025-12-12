"""Molecular generation modules for de novo drug design.

This package provides:
- SMILES VAE for molecular generation
- Latent space optimization
- Molecular validity checking
- Property-guided generation

Classes:
    SmilesVAE: Variational Autoencoder for SMILES
    SmilesTokenizer: SMILES string tokenization
    LatentOptimizer: Optimize molecules in latent space
    MoleculeGenerator: High-level generation interface
"""

from src.generation.tokenizer import SmilesTokenizer, create_vocabulary
from src.generation.vae import SmilesVAE, VAEEncoder, VAEDecoder
from src.generation.optimizer import LatentOptimizer, PropertyPredictor
from src.generation.generator import MoleculeGenerator, validate_smiles, calculate_properties

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
