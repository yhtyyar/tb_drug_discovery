"""Diffusion Models for Molecular Generation.

This module implements diffusion-based generative models for 
de novo molecular design, including:
- DDPM: Denoising Diffusion Probabilistic Models
- DDIM: Denoising Diffusion Implicit Models  
- EDM: Equivariant Diffusion Models for 3D molecules

Example:
    >>> from src.generation.diffusion import MolecularDiffusion
    >>> model = MolecularDiffusion(latent_dim=256)
    >>> molecules = model.sample(num_samples=100)
"""

from .scheduler import NoiseScheduler, CosineScheduler, LinearScheduler
from .mol_diffusion import MolecularDiffusion, DiffusionConfig
from .sampler import DDPMSampler, DDIMSampler

__all__ = [
    "NoiseScheduler",
    "CosineScheduler", 
    "LinearScheduler",
    "MolecularDiffusion",
    "DiffusionConfig",
    "DDPMSampler",
    "DDIMSampler",
]
