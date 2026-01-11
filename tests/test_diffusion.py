"""Tests for diffusion models module.

Tests cover:
- Noise schedulers (Linear, Cosine, Sigmoid)
- Samplers (DDPM, DDIM)
- MolecularDiffusion model
"""

import pytest
import numpy as np
import torch

from src.generation.diffusion.scheduler import (
    LinearScheduler,
    CosineScheduler,
    SigmoidScheduler,
    get_scheduler,
)
from src.generation.diffusion.mol_diffusion import (
    DiffusionConfig,
    MolecularDiffusion,
    DenoisingNetwork,
    SinusoidalPositionEmbedding,
)


class TestNoiseSchedulers:
    """Tests for noise scheduler implementations."""
    
    def test_linear_scheduler_init(self):
        """Test linear scheduler initialization."""
        scheduler = LinearScheduler(num_timesteps=100)
        
        assert scheduler.num_timesteps == 100
        assert len(scheduler.betas) == 100
        assert scheduler.betas[0] < scheduler.betas[-1]  # Increasing
    
    def test_cosine_scheduler_init(self):
        """Test cosine scheduler initialization."""
        scheduler = CosineScheduler(num_timesteps=1000)
        
        assert scheduler.num_timesteps == 1000
        assert len(scheduler.betas) == 1000
        # Cosine schedule should have smoother progression
        assert scheduler.alphas_cumprod[0] > scheduler.alphas_cumprod[-1]
    
    def test_sigmoid_scheduler_init(self):
        """Test sigmoid scheduler initialization."""
        scheduler = SigmoidScheduler(num_timesteps=500)
        
        assert scheduler.num_timesteps == 500
        assert len(scheduler.betas) == 500
    
    def test_add_noise(self):
        """Test adding noise to data."""
        scheduler = CosineScheduler(num_timesteps=100)
        
        x = torch.randn(4, 64)  # batch of 4, dim 64
        noise = torch.randn_like(x)
        timesteps = torch.tensor([10, 20, 30, 40])
        
        x_noisy = scheduler.add_noise(x, noise, timesteps)
        
        assert x_noisy.shape == x.shape
        # Noisy data should be different from original
        assert not torch.allclose(x_noisy, x)
    
    def test_noise_increases_with_timestep(self):
        """Test that more noise is added at higher timesteps."""
        scheduler = LinearScheduler(num_timesteps=100)
        
        x = torch.ones(2, 32)
        noise = torch.randn(2, 32)
        
        x_early = scheduler.add_noise(x, noise, torch.tensor([10, 10]))
        x_late = scheduler.add_noise(x, noise, torch.tensor([90, 90]))
        
        # Later timesteps should have more noise (less signal)
        early_signal = (x_early - noise).abs().mean()
        late_signal = (x_late - noise).abs().mean()
        
        # Early should retain more of original signal
        assert early_signal > late_signal
    
    def test_get_scheduler_factory(self):
        """Test scheduler factory function."""
        linear = get_scheduler("linear", num_timesteps=100)
        cosine = get_scheduler("cosine", num_timesteps=100)
        sigmoid = get_scheduler("sigmoid", num_timesteps=100)
        
        assert isinstance(linear, LinearScheduler)
        assert isinstance(cosine, CosineScheduler)
        assert isinstance(sigmoid, SigmoidScheduler)
    
    def test_get_scheduler_invalid(self):
        """Test scheduler factory with invalid type."""
        with pytest.raises(ValueError):
            get_scheduler("invalid_scheduler")


class TestDiffusionConfig:
    """Tests for DiffusionConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DiffusionConfig()
        
        assert config.latent_dim == 256
        assert config.hidden_dim == 512
        assert config.num_timesteps == 1000
        assert config.scheduler_type == "cosine"
        assert config.prediction_type == "noise"
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = DiffusionConfig(
            latent_dim=128,
            num_timesteps=500,
            scheduler_type="linear",
        )
        
        assert config.latent_dim == 128
        assert config.num_timesteps == 500
        assert config.scheduler_type == "linear"


class TestSinusoidalEmbedding:
    """Tests for sinusoidal position embeddings."""
    
    def test_embedding_shape(self):
        """Test embedding output shape."""
        embed = SinusoidalPositionEmbedding(dim=128)
        timesteps = torch.tensor([0, 10, 50, 100])
        
        output = embed(timesteps)
        
        assert output.shape == (4, 128)
    
    def test_embedding_different_timesteps(self):
        """Test that different timesteps produce different embeddings."""
        embed = SinusoidalPositionEmbedding(dim=64)
        
        t1 = embed(torch.tensor([10]))
        t2 = embed(torch.tensor([20]))
        
        assert not torch.allclose(t1, t2)


class TestDenoisingNetwork:
    """Tests for denoising network architecture."""
    
    def test_network_forward(self):
        """Test forward pass of denoising network."""
        config = DiffusionConfig(latent_dim=64, hidden_dim=128, num_layers=2)
        network = DenoisingNetwork(config)
        
        x = torch.randn(4, 64)
        timesteps = torch.tensor([10, 20, 30, 40])
        
        output = network(x, timesteps)
        
        assert output.shape == (4, 64)
    
    def test_network_with_condition(self):
        """Test network with conditioning."""
        config = DiffusionConfig(
            latent_dim=64,
            hidden_dim=128,
            condition_dim=16,
        )
        network = DenoisingNetwork(config)
        
        x = torch.randn(4, 64)
        timesteps = torch.tensor([10, 20, 30, 40])
        condition = torch.randn(4, 16)
        
        output = network(x, timesteps, condition)
        
        assert output.shape == (4, 64)
    
    def test_network_self_conditioning(self):
        """Test network with self-conditioning."""
        config = DiffusionConfig(
            latent_dim=64,
            use_self_condition=True,
        )
        network = DenoisingNetwork(config)
        
        x = torch.randn(4, 64)
        timesteps = torch.tensor([10, 20, 30, 40])
        self_cond = torch.randn(4, 64)
        
        output = network(x, timesteps, self_condition=self_cond)
        
        assert output.shape == (4, 64)


class TestMolecularDiffusion:
    """Tests for complete MolecularDiffusion model."""
    
    @pytest.fixture
    def model(self):
        """Create test model."""
        config = DiffusionConfig(
            latent_dim=32,
            hidden_dim=64,
            num_layers=2,
            num_timesteps=100,
        )
        return MolecularDiffusion(config)
    
    def test_model_init(self, model):
        """Test model initialization."""
        assert model.config.latent_dim == 32
        assert model.config.num_timesteps == 100
        assert model.scheduler is not None
        assert model.denoiser is not None
    
    def test_compute_loss(self, model):
        """Test loss computation."""
        x = torch.randn(8, 32)
        
        loss_dict = model.compute_loss(x)
        
        assert "loss" in loss_dict
        assert loss_dict["loss"].ndim == 0  # Scalar
        assert loss_dict["loss"].item() > 0
    
    def test_compute_loss_with_condition(self):
        """Test loss with conditioning."""
        config = DiffusionConfig(
            latent_dim=32,
            condition_dim=8,
            num_timesteps=50,
        )
        model = MolecularDiffusion(config)
        
        x = torch.randn(4, 32)
        condition = torch.randn(4, 8)
        
        loss_dict = model.compute_loss(x, condition)
        
        assert "loss" in loss_dict
    
    def test_generate(self, model):
        """Test sample generation."""
        samples = model.generate(
            num_samples=4,
            num_inference_steps=10,
            sampler_type="ddpm",
        )
        
        assert samples.shape == (4, 32)
    
    def test_generate_with_condition(self):
        """Test conditional generation."""
        config = DiffusionConfig(
            latent_dim=32,
            condition_dim=8,
            num_timesteps=50,
        )
        model = MolecularDiffusion(config)
        
        condition = torch.randn(4, 8)
        
        samples = model.generate(
            num_samples=4,
            condition=condition,
            num_inference_steps=10,
        )
        
        assert samples.shape == (4, 32)
    
    def test_save_and_load(self, model, tmp_path):
        """Test model serialization."""
        save_path = str(tmp_path / "model.pt")
        
        # Save
        model.save(save_path)
        
        # Load
        loaded = MolecularDiffusion.load(save_path)
        
        assert loaded.config.latent_dim == model.config.latent_dim
        assert loaded.config.num_timesteps == model.config.num_timesteps
    
    def test_ema_setup(self, model):
        """Test EMA initialization."""
        model.setup_ema()
        
        assert model.ema is not None
        assert len(model.ema.shadow) > 0


class TestDDIMSampler:
    """Tests for DDIM sampler."""
    
    def test_ddim_fewer_steps(self):
        """Test DDIM with fewer inference steps."""
        from src.generation.diffusion.sampler import DDIMSampler
        
        config = DiffusionConfig(latent_dim=32, num_timesteps=100)
        model = MolecularDiffusion(config)
        scheduler = model.scheduler
        
        # DDIM should work with fewer steps
        sampler = DDIMSampler(
            scheduler=scheduler,
            model=model.denoiser,
            num_inference_steps=10,
            eta=0.0,  # Deterministic
        )
        
        assert len(sampler.timesteps) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
