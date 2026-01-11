"""Molecular Diffusion Model for de novo drug design.

This module implements a complete diffusion-based generative model
for molecular generation, supporting both latent space diffusion
and direct SMILES generation.

Features:
- Conditional generation with target properties
- Classifier-free guidance
- Multiple denoising architectures
- Integration with molecular scoring functions

Example:
    >>> config = DiffusionConfig(latent_dim=256, num_timesteps=1000)
    >>> model = MolecularDiffusion(config)
    >>> model.train(train_loader, epochs=100)
    >>> molecules = model.generate(num_samples=100, condition=target_activity)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm

from .scheduler import NoiseScheduler, CosineScheduler, get_scheduler
from .sampler import DDPMSampler, DDIMSampler, get_sampler


@dataclass
class DiffusionConfig:
    """Configuration for Molecular Diffusion model.
    
    Args:
        latent_dim: Dimension of latent space.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of denoising network layers.
        num_timesteps: Number of diffusion timesteps.
        scheduler_type: Type of noise scheduler.
        condition_dim: Dimension of conditioning (0 = unconditional).
        dropout: Dropout probability.
        use_self_condition: Use self-conditioning.
        prediction_type: 'noise', 'x0', or 'velocity'.
    """
    latent_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 4
    num_timesteps: int = 1000
    scheduler_type: str = "cosine"
    condition_dim: int = 0
    dropout: float = 0.1
    use_self_condition: bool = True
    prediction_type: str = "noise"  # 'noise', 'x0', 'velocity'
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    ema_decay: float = 0.9999
    gradient_clip: float = 1.0
    
    # Sampling
    sampler_type: str = "ddim"
    num_inference_steps: int = 50
    guidance_scale: float = 2.0


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional embeddings for timestep encoding.
    
    Uses sinusoidal functions to encode timesteps into continuous
    representations, following the transformer architecture.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        return embeddings


class MLPBlock(nn.Module):
    """MLP block with residual connection and normalization."""
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        time_emb_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Time embedding projection
        self.time_mlp = None
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, dim * 2),
            )
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        
        # Add time embedding
        if self.time_mlp is not None and time_emb is not None:
            time_emb = self.time_mlp(time_emb)
            scale, shift = time_emb.chunk(2, dim=-1)
            h = h * (1 + scale) + shift
        
        h = self.fc1(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        
        return x + h


class DenoisingNetwork(nn.Module):
    """Neural network for noise prediction in diffusion models.
    
    Architecture: MLP with time embeddings and optional conditioning.
    
    Args:
        config: DiffusionConfig with model parameters.
    """
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        
        self.config = config
        latent_dim = config.latent_dim
        hidden_dim = config.hidden_dim
        
        # Time embedding
        time_dim = hidden_dim * 4
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Condition embedding (if conditional)
        self.condition_embedding = None
        if config.condition_dim > 0:
            self.condition_embedding = nn.Sequential(
                nn.Linear(config.condition_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, time_dim),
            )
        
        # Self-conditioning
        input_dim = latent_dim
        if config.use_self_condition:
            input_dim = latent_dim * 2
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Denoising blocks
        self.blocks = nn.ModuleList([
            MLPBlock(hidden_dim, hidden_dim * 2, config.dropout, time_dim)
            for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        self_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise (or x0/velocity) from noisy input.
        
        Args:
            x: Noisy latent vectors (batch, latent_dim).
            timesteps: Diffusion timesteps (batch,).
            condition: Optional conditioning (batch, condition_dim).
            self_condition: Previous prediction for self-conditioning.
            
        Returns:
            Predicted noise/x0/velocity of shape (batch, latent_dim).
        """
        # Time embedding
        t_emb = self.time_embedding(timesteps)
        
        # Add condition embedding
        if condition is not None and self.condition_embedding is not None:
            c_emb = self.condition_embedding(condition)
            t_emb = t_emb + c_emb
        
        # Self-conditioning
        if self.config.use_self_condition:
            if self_condition is None:
                self_condition = torch.zeros_like(x)
            x = torch.cat([x, self_condition], dim=-1)
        
        # Forward pass
        h = self.input_proj(x)
        
        for block in self.blocks:
            h = block(h, t_emb)
        
        out = self.output_proj(h)
        
        return out


class EMA:
    """Exponential Moving Average for model parameters.
    
    Maintains a shadow copy of model parameters that is updated
    with exponential moving average, typically used for stable sampling.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        self.register()
    
    def register(self):
        """Register model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters with EMA."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class MolecularDiffusion(nn.Module):
    """Complete Molecular Diffusion model for drug discovery.
    
    Combines the denoising network with noise scheduler and samplers
    for end-to-end molecular generation.
    
    Args:
        config: DiffusionConfig with all parameters.
        vae: Optional VAE for latent space diffusion.
        
    Example:
        >>> config = DiffusionConfig(latent_dim=256)
        >>> model = MolecularDiffusion(config)
        >>> 
        >>> # Training
        >>> loss = model.compute_loss(x_batch, condition)
        >>> 
        >>> # Generation
        >>> samples = model.generate(num_samples=100, condition=target)
    """
    
    def __init__(
        self,
        config: DiffusionConfig,
        vae: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.config = config
        self.vae = vae
        
        # Denoising network
        self.denoiser = DenoisingNetwork(config)
        
        # Noise scheduler
        self.scheduler = get_scheduler(
            config.scheduler_type,
            num_timesteps=config.num_timesteps,
        )
        
        # EMA for stable sampling
        self.ema = None
        
        logger.info(f"MolecularDiffusion initialized: latent_dim={config.latent_dim}, "
                   f"timesteps={config.num_timesteps}, scheduler={config.scheduler_type}")
    
    def compute_loss(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss.
        
        Args:
            x: Clean latent vectors (batch, latent_dim).
            condition: Optional conditioning (batch, condition_dim).
            noise: Optional pre-generated noise.
            
        Returns:
            Dictionary with 'loss' and optional auxiliary losses.
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.config.num_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )
        
        # Sample noise
        if noise is None:
            noise = torch.randn_like(x)
        
        # Add noise to data
        x_noisy = self.scheduler.add_noise(x, noise, timesteps)
        
        # Self-conditioning (with 50% probability during training)
        self_condition = None
        if self.config.use_self_condition and torch.rand(1).item() > 0.5:
            with torch.no_grad():
                self_condition = self.denoiser(x_noisy, timesteps, condition)
        
        # Predict target
        prediction = self.denoiser(x_noisy, timesteps, condition, self_condition)
        
        # Compute loss based on prediction type
        if self.config.prediction_type == "noise":
            target = noise
        elif self.config.prediction_type == "x0":
            target = x
        elif self.config.prediction_type == "velocity":
            target = self.scheduler.get_velocity(x, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
        
        # MSE loss
        loss = F.mse_loss(prediction, target)
        
        return {"loss": loss}
    
    @torch.no_grad()
    def generate(
        self,
        num_samples: int,
        condition: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        sampler_type: Optional[str] = None,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Generate molecular latent vectors.
        
        Args:
            num_samples: Number of molecules to generate.
            condition: Optional conditioning tensor.
            guidance_scale: Classifier-free guidance scale.
            num_inference_steps: Number of sampling steps (for DDIM).
            sampler_type: Override default sampler type.
            return_trajectory: Return intermediate samples.
            
        Returns:
            Generated latent vectors (and trajectory if requested).
        """
        device = next(self.parameters()).device
        
        # Use config defaults if not specified
        if guidance_scale is None:
            guidance_scale = self.config.guidance_scale
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_steps
        if sampler_type is None:
            sampler_type = self.config.sampler_type
        
        # Apply EMA weights for sampling
        if self.ema is not None:
            self.ema.apply_shadow()
        
        self.eval()
        
        # Create sampler
        sampler_kwargs = {}
        if sampler_type == "ddim":
            sampler_kwargs["num_inference_steps"] = num_inference_steps
        
        sampler = get_sampler(
            sampler_type,
            self.scheduler,
            self._forward_with_cfg(condition, guidance_scale),
            **sampler_kwargs,
        )
        
        # Generate
        shape = (num_samples, self.config.latent_dim)
        
        if return_trajectory:
            samples, trajectory = sampler.sample_with_trajectory(
                shape, condition, save_every=num_inference_steps // 10
            )
        else:
            samples = sampler.sample(shape, condition, guidance_scale)
            trajectory = None
        
        # Restore original weights
        if self.ema is not None:
            self.ema.restore()
        
        if return_trajectory:
            return samples, trajectory
        return samples
    
    def _forward_with_cfg(
        self,
        condition: Optional[torch.Tensor],
        guidance_scale: float,
    ) -> Callable:
        """Create forward function with classifier-free guidance."""
        
        def forward_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            # Self-conditioning
            self_cond = None
            if self.config.use_self_condition:
                self_cond = self.denoiser(x, t, condition)
            
            # Conditional prediction
            cond_pred = self.denoiser(x, t, condition, self_cond)
            
            # Classifier-free guidance
            if guidance_scale != 1.0 and condition is not None:
                uncond_pred = self.denoiser(x, t, None, self_cond)
                cond_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            
            return cond_pred
        
        return forward_fn
    
    def decode_latents(self, z: torch.Tensor) -> List[str]:
        """Decode latent vectors to SMILES using VAE.
        
        Args:
            z: Latent vectors (batch, latent_dim).
            
        Returns:
            List of SMILES strings.
        """
        if self.vae is None:
            raise ValueError("VAE not provided for decoding")
        
        self.vae.eval()
        with torch.no_grad():
            # Decode using VAE decoder
            generated = self.vae.decoder.generate(z)
        
        # Convert to SMILES (requires tokenizer)
        return generated
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer for training."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
    
    def setup_ema(self):
        """Initialize EMA for model parameters."""
        self.ema = EMA(self.denoiser, decay=self.config.ema_decay)
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "config": self.config,
            "state_dict": self.state_dict(),
        }
        if self.ema is not None:
            checkpoint["ema_shadow"] = self.ema.shadow
        
        torch.save(checkpoint, path)
        logger.info(f"Saved diffusion model to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "MolecularDiffusion":
        """Load model from checkpoint."""
        if device is None:
            device = torch.device("cpu")
        
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        
        if "ema_shadow" in checkpoint:
            model.setup_ema()
            model.ema.shadow = checkpoint["ema_shadow"]
        
        logger.info(f"Loaded diffusion model from {path}")
        return model


def train_diffusion_model(
    model: MolecularDiffusion,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    epochs: int = 100,
    save_dir: str = "models/diffusion",
    use_ema: bool = True,
) -> Dict[str, List[float]]:
    """Train diffusion model.
    
    Args:
        model: MolecularDiffusion model.
        train_loader: Training data loader.
        val_loader: Optional validation loader.
        epochs: Number of training epochs.
        save_dir: Directory to save checkpoints.
        use_ema: Use exponential moving average.
        
    Returns:
        Training history dictionary.
    """
    device = next(model.parameters()).device
    optimizer = model.configure_optimizers()
    
    if use_ema:
        model.setup_ema()
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            if isinstance(batch, (tuple, list)):
                x, condition = batch[0].to(device), batch[1].to(device) if len(batch) > 1 else None
            else:
                x, condition = batch.to(device), None
            
            optimizer.zero_grad()
            loss_dict = model.compute_loss(x, condition)
            loss = loss_dict["loss"]
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.config.gradient_clip)
            optimizer.step()
            
            if model.ema is not None:
                model.ema.update()
            
            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (tuple, list)):
                        x, condition = batch[0].to(device), batch[1].to(device) if len(batch) > 1 else None
                    else:
                        x, condition = batch.to(device), None
                    
                    loss_dict = model.compute_loss(x, condition)
                    val_losses.append(loss_dict["loss"].item())
            
            avg_val_loss = np.mean(val_losses)
            history["val_loss"].append(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model.save(f"{save_dir}/best_model.pt")
            
            logger.info(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
        else:
            logger.info(f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}")
        
        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            model.save(f"{save_dir}/checkpoint_epoch{epoch + 1}.pt")
    
    # Save final model
    model.save(f"{save_dir}/final_model.pt")
    
    return history
