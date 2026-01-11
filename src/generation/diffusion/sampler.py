"""Samplers for diffusion models.

This module implements various sampling strategies for
generating molecules from trained diffusion models.

Samplers:
- DDPM: Original denoising diffusion sampling
- DDIM: Deterministic/accelerated sampling
- Ancestral: Ancestral sampling with noise injection
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, List

import torch
import torch.nn as nn
from tqdm import tqdm
from loguru import logger

from .scheduler import NoiseScheduler


class BaseSampler(ABC):
    """Base class for diffusion samplers.
    
    Args:
        scheduler: Noise scheduler instance.
        model: Denoising model.
    """
    
    def __init__(
        self,
        scheduler: NoiseScheduler,
        model: nn.Module,
    ):
        self.scheduler = scheduler
        self.model = model
        self.device = next(model.parameters()).device
    
    @abstractmethod
    def sample(
        self,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Generate samples from the diffusion model."""
        pass
    
    @torch.no_grad()
    def _predict_noise(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise using the model."""
        if condition is not None:
            return self.model(x_t, timestep, condition)
        return self.model(x_t, timestep)


class DDPMSampler(BaseSampler):
    """DDPM (Denoising Diffusion Probabilistic Models) sampler.
    
    Original sampling algorithm from Ho et al. (2020).
    Iteratively denoises from pure noise to data.
    
    Args:
        scheduler: Noise scheduler.
        model: Trained denoising model.
        
    Example:
        >>> sampler = DDPMSampler(scheduler, model)
        >>> samples = sampler.sample((batch_size, latent_dim))
    """
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Generate samples using DDPM.
        
        Args:
            shape: Shape of samples to generate.
            condition: Optional conditioning tensor.
            guidance_scale: Classifier-free guidance scale.
            show_progress: Show progress bar.
            
        Returns:
            Generated samples.
        """
        self.model.eval()
        
        # Start from pure noise
        x_t = torch.randn(shape, device=self.device)
        
        timesteps = range(self.scheduler.num_timesteps - 1, -1, -1)
        if show_progress:
            timesteps = tqdm(timesteps, desc="DDPM Sampling")
        
        for t in timesteps:
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self._predict_noise(x_t, t_tensor, condition)
            
            # Classifier-free guidance
            if guidance_scale != 1.0 and condition is not None:
                noise_uncond = self._predict_noise(x_t, t_tensor, None)
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
            
            # Get previous sample
            x_t = self.scheduler.step(noise_pred, t, x_t, eta=1.0)
        
        return x_t
    
    @torch.no_grad()
    def sample_with_trajectory(
        self,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        save_every: int = 100,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate samples and save intermediate states.
        
        Useful for visualization and debugging.
        
        Args:
            shape: Shape of samples.
            condition: Optional condition.
            save_every: Save trajectory every N steps.
            
        Returns:
            Tuple of (final_samples, trajectory_list).
        """
        self.model.eval()
        
        x_t = torch.randn(shape, device=self.device)
        trajectory = [x_t.clone()]
        
        for t in range(self.scheduler.num_timesteps - 1, -1, -1):
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            
            noise_pred = self._predict_noise(x_t, t_tensor, condition)
            x_t = self.scheduler.step(noise_pred, t, x_t, eta=1.0)
            
            if t % save_every == 0:
                trajectory.append(x_t.clone())
        
        trajectory.append(x_t.clone())
        return x_t, trajectory


class DDIMSampler(BaseSampler):
    """DDIM (Denoising Diffusion Implicit Models) sampler.
    
    Deterministic sampling with optional stochasticity.
    Allows faster sampling with fewer steps.
    
    Reference: Song et al. (2021)
    
    Args:
        scheduler: Noise scheduler.
        model: Trained denoising model.
        num_inference_steps: Number of steps for sampling (< num_timesteps).
        eta: Stochasticity parameter (0 = deterministic, 1 = DDPM).
    """
    
    def __init__(
        self,
        scheduler: NoiseScheduler,
        model: nn.Module,
        num_inference_steps: int = 50,
        eta: float = 0.0,
    ):
        super().__init__(scheduler, model)
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        
        # Compute timestep schedule for DDIM
        self.timesteps = self._compute_timesteps()
    
    def _compute_timesteps(self) -> torch.Tensor:
        """Compute spaced timesteps for accelerated sampling."""
        step_ratio = self.scheduler.num_timesteps // self.num_inference_steps
        timesteps = torch.arange(
            0, 
            self.scheduler.num_timesteps, 
            step_ratio,
            device=self.device
        ).flip(0)
        return timesteps
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Generate samples using DDIM.
        
        Args:
            shape: Shape of samples to generate.
            condition: Optional conditioning tensor.
            guidance_scale: Classifier-free guidance scale.
            show_progress: Show progress bar.
            
        Returns:
            Generated samples.
        """
        self.model.eval()
        
        # Start from pure noise
        x_t = torch.randn(shape, device=self.device)
        
        timesteps = self.timesteps.tolist()
        if show_progress:
            timesteps = tqdm(timesteps, desc="DDIM Sampling")
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self._predict_noise(x_t, t_tensor, condition)
            
            # Classifier-free guidance
            if guidance_scale != 1.0 and condition is not None:
                noise_uncond = self._predict_noise(x_t, t_tensor, None)
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
            
            # DDIM update step
            x_t = self._ddim_step(x_t, noise_pred, t, i)
        
        return x_t
    
    def _ddim_step(
        self,
        x_t: torch.Tensor,
        noise_pred: torch.Tensor,
        t: int,
        step_idx: int,
    ) -> torch.Tensor:
        """Perform one DDIM sampling step."""
        alpha_t = self.scheduler.alphas_cumprod[t]
        
        # Get previous timestep
        if step_idx < len(self.timesteps) - 1:
            prev_t = self.timesteps[step_idx + 1].item()
            alpha_prev = self.scheduler.alphas_cumprod[int(prev_t)]
        else:
            alpha_prev = torch.tensor(1.0, device=self.device)
        
        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        # Compute sigma for stochasticity
        sigma_t = self.eta * torch.sqrt(
            (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
        )
        
        # Direction pointing to x_t  
        pred_dir = torch.sqrt(1 - alpha_prev - sigma_t ** 2) * noise_pred
        
        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_prev) * x_0_pred + pred_dir
        
        # Add noise
        if self.eta > 0 and step_idx < len(self.timesteps) - 1:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma_t * noise
        
        return x_prev


class AncestralSampler(BaseSampler):
    """Ancestral sampler with controllable noise injection.
    
    Provides fine-grained control over the sampling process
    with adjustable noise levels at each step.
    """
    
    def __init__(
        self,
        scheduler: NoiseScheduler,
        model: nn.Module,
        temperature: float = 1.0,
    ):
        super().__init__(scheduler, model)
        self.temperature = temperature
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Generate samples with ancestral sampling.
        
        Args:
            shape: Shape of samples.
            condition: Optional condition.
            guidance_scale: Guidance scale for CFG.
            show_progress: Show progress bar.
            
        Returns:
            Generated samples.
        """
        self.model.eval()
        
        x_t = torch.randn(shape, device=self.device) * self.temperature
        
        timesteps = range(self.scheduler.num_timesteps - 1, -1, -1)
        if show_progress:
            timesteps = tqdm(timesteps, desc="Ancestral Sampling")
        
        for t in timesteps:
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self._predict_noise(x_t, t_tensor, condition)
            
            # Classifier-free guidance
            if guidance_scale != 1.0 and condition is not None:
                noise_uncond = self._predict_noise(x_t, t_tensor, None)
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
            
            # Ancestral step with temperature
            x_t = self._ancestral_step(x_t, noise_pred, t)
        
        return x_t
    
    def _ancestral_step(
        self,
        x_t: torch.Tensor,
        noise_pred: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """Perform ancestral sampling step."""
        alpha_t = self.scheduler.alphas_cumprod[t]
        alpha_prev = self.scheduler.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
        beta_t = self.scheduler.betas[t]
        
        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        
        # Compute mean
        coef1 = torch.sqrt(alpha_prev) * beta_t / (1 - alpha_t)
        coef2 = torch.sqrt(alpha_t) * (1 - alpha_prev) / (1 - alpha_t)
        mean = coef1 * x_0_pred + coef2 * x_t
        
        # Add noise
        if t > 0:
            variance = beta_t * (1 - alpha_prev) / (1 - alpha_t)
            noise = torch.randn_like(x_t) * self.temperature
            x_prev = mean + torch.sqrt(variance) * noise
        else:
            x_prev = mean
        
        return x_prev


def get_sampler(
    sampler_type: str,
    scheduler: NoiseScheduler,
    model: nn.Module,
    **kwargs,
) -> BaseSampler:
    """Factory function to create samplers.
    
    Args:
        sampler_type: Type of sampler ('ddpm', 'ddim', 'ancestral').
        scheduler: Noise scheduler.
        model: Denoising model.
        **kwargs: Additional sampler arguments.
        
    Returns:
        Sampler instance.
    """
    samplers = {
        "ddpm": DDPMSampler,
        "ddim": DDIMSampler,
        "ancestral": AncestralSampler,
    }
    
    if sampler_type not in samplers:
        raise ValueError(f"Unknown sampler: {sampler_type}")
    
    return samplers[sampler_type](scheduler, model, **kwargs)
