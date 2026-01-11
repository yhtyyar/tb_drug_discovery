"""Noise schedulers for diffusion models.

This module implements various noise schedules for controlling
the forward and reverse diffusion processes.

Schedulers:
- Linear: Linear interpolation of beta values
- Cosine: Cosine annealing schedule (improved for molecules)
- Sigmoid: Sigmoid-based schedule
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class NoiseScheduler(ABC):
    """Base class for noise schedulers.
    
    Defines the noise schedule for the diffusion process,
    controlling how noise is added during forward diffusion
    and removed during reverse diffusion.
    
    Args:
        num_timesteps: Number of diffusion timesteps.
        beta_start: Starting beta value.
        beta_end: Ending beta value.
        device: Computation device.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: Optional[torch.device] = None,
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device or torch.device("cpu")
        
        # Compute schedule
        self.betas = self._compute_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=self.device),
            self.alphas_cumprod[:-1]
        ])
        
        # Precompute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        logger.debug(f"Initialized {self.__class__.__name__} with {num_timesteps} steps")
    
    @abstractmethod
    def _compute_betas(self) -> torch.Tensor:
        """Compute the beta schedule."""
        pass
    
    def add_noise(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to data according to schedule.
        
        Forward diffusion process: q(x_t | x_0).
        
        Args:
            x: Original data of shape (batch, ...).
            noise: Gaussian noise of same shape as x.
            timesteps: Timestep indices of shape (batch,).
            
        Returns:
            Noisy data at timestep t.
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_alpha.shape) < len(x.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * x + sqrt_one_minus_alpha * noise
    
    def remove_noise(
        self,
        x_t: torch.Tensor,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Remove predicted noise from noisy data.
        
        Reverse diffusion step: p(x_{t-1} | x_t).
        
        Args:
            x_t: Noisy data at timestep t.
            noise_pred: Predicted noise.
            timesteps: Current timestep indices.
            
        Returns:
            Denoised data (mean of p(x_{t-1} | x_t)).
        """
        sqrt_recip_alpha = self.sqrt_recip_alphas[timesteps]
        beta = self.betas[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_recip_alpha.shape) < len(x_t.shape):
            sqrt_recip_alpha = sqrt_recip_alpha.unsqueeze(-1)
            beta = beta.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        # Predict x_0
        x_0_pred = sqrt_recip_alpha * (x_t - beta / sqrt_one_minus_alpha * noise_pred)
        
        return x_0_pred
    
    def step(
        self,
        noise_pred: torch.Tensor,
        timestep: int,
        x_t: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Perform one reverse diffusion step.
        
        Args:
            noise_pred: Predicted noise.
            timestep: Current timestep.
            x_t: Current noisy sample.
            eta: Stochasticity parameter (0 = deterministic).
            
        Returns:
            Sample at timestep t-1.
        """
        t = timestep
        prev_t = max(0, t - 1)
        
        # Current and previous alpha values
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[prev_t] if prev_t > 0 else torch.tensor(1.0)
        beta_t = self.betas[t]
        
        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        
        # Clip x_0 prediction for stability
        x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        # Compute variance
        sigma_t = eta * torch.sqrt(
            (1 - alpha_prev) / (1 - alpha_t) * beta_t
        )
        
        # Direction pointing to x_t
        pred_dir = torch.sqrt(1 - alpha_prev - sigma_t ** 2) * noise_pred
        
        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_prev) * x_0_pred + pred_dir
        
        # Add noise if not at final step
        if eta > 0 and t > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma_t * noise
        
        return x_prev
    
    def get_velocity(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute velocity target for v-prediction.
        
        v = sqrt(alpha) * noise - sqrt(1-alpha) * x
        
        Args:
            x: Original data.
            noise: Gaussian noise.
            timesteps: Timestep indices.
            
        Returns:
            Velocity target.
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        while len(sqrt_alpha.shape) < len(x.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * noise - sqrt_one_minus_alpha * x


class LinearScheduler(NoiseScheduler):
    """Linear noise schedule.
    
    Beta values increase linearly from beta_start to beta_end.
    Simple but effective for many applications.
    """
    
    def _compute_betas(self) -> torch.Tensor:
        return torch.linspace(
            self.beta_start,
            self.beta_end,
            self.num_timesteps,
            device=self.device,
        )


class CosineScheduler(NoiseScheduler):
    """Cosine noise schedule.
    
    Uses cosine annealing for smoother noise addition,
    which typically works better for molecular generation.
    
    Reference: Nichol & Dhariwal (2021)
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        s: float = 0.008,
        device: Optional[torch.device] = None,
    ):
        self.s = s
        super().__init__(
            num_timesteps=num_timesteps,
            beta_start=0,  # Not used
            beta_end=0,    # Not used
            device=device,
        )
    
    def _compute_betas(self) -> torch.Tensor:
        steps = self.num_timesteps + 1
        t = torch.linspace(0, self.num_timesteps, steps, device=self.device)
        
        # Cosine schedule
        alphas_cumprod = torch.cos(
            ((t / self.num_timesteps) + self.s) / (1 + self.s) * np.pi / 2
        ) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        
        # Compute betas from alphas_cumprod
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        
        # Clip betas to reasonable range
        return torch.clamp(betas, 0.0001, 0.9999)


class SigmoidScheduler(NoiseScheduler):
    """Sigmoid noise schedule.
    
    Uses sigmoid function for smooth transition,
    useful for fine-grained control of noise levels.
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        sigmoid_start: float = -3,
        sigmoid_end: float = 3,
        device: Optional[torch.device] = None,
    ):
        self.sigmoid_start = sigmoid_start
        self.sigmoid_end = sigmoid_end
        super().__init__(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device,
        )
    
    def _compute_betas(self) -> torch.Tensor:
        t = torch.linspace(
            self.sigmoid_start,
            self.sigmoid_end,
            self.num_timesteps,
            device=self.device,
        )
        sigmoid_values = torch.sigmoid(t)
        
        # Scale to beta range
        betas = self.beta_start + (self.beta_end - self.beta_start) * sigmoid_values
        
        return betas


def get_scheduler(
    scheduler_type: str = "cosine",
    num_timesteps: int = 1000,
    **kwargs,
) -> NoiseScheduler:
    """Factory function to create noise schedulers.
    
    Args:
        scheduler_type: Type of scheduler ('linear', 'cosine', 'sigmoid').
        num_timesteps: Number of diffusion timesteps.
        **kwargs: Additional scheduler-specific arguments.
        
    Returns:
        NoiseScheduler instance.
    """
    schedulers = {
        "linear": LinearScheduler,
        "cosine": CosineScheduler,
        "sigmoid": SigmoidScheduler,
    }
    
    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler: {scheduler_type}. Choose from: {list(schedulers.keys())}")
    
    return schedulers[scheduler_type](num_timesteps=num_timesteps, **kwargs)
