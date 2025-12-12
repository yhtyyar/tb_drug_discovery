"""Unit tests for molecular generation modules.

Tests cover:
- SMILES tokenization
- VAE architecture
- Generation and evaluation
- Latent space optimization
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


@pytest.fixture
def sample_smiles():
    """Sample SMILES for testing."""
    return [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CC(C)O",  # Isopropanol
        "CCN",  # Ethylamine
        "CCCC",  # Butane
        "C1CCCCC1",  # Cyclohexane
        "CC(=O)Nc1ccc(O)cc1",  # Paracetamol
    ]


class TestSmilesTokenizer:
    """Tests for SmilesTokenizer."""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization."""
        from src.generation.tokenizer import SmilesTokenizer
        
        tokenizer = SmilesTokenizer(max_length=100)
        
        assert tokenizer.vocab_size >= 4  # Special tokens
        assert tokenizer.max_length == 100
    
    def test_tokenize_simple(self):
        """Test simple tokenization."""
        from src.generation.tokenizer import SmilesTokenizer
        
        tokenizer = SmilesTokenizer()
        tokens = tokenizer.tokenize("CCO")
        
        assert tokens == ["C", "C", "O"]
    
    def test_tokenize_complex(self):
        """Test complex SMILES tokenization."""
        from src.generation.tokenizer import SmilesTokenizer
        
        tokenizer = SmilesTokenizer()
        
        # Chlorine (multi-char)
        tokens = tokenizer.tokenize("CCCl")
        assert "Cl" in tokens
        
        # Bromine
        tokens = tokenizer.tokenize("CCBr")
        assert "Br" in tokens
        
        # Ring
        tokens = tokenizer.tokenize("c1ccccc1")
        assert "1" in tokens
    
    def test_fit_vocabulary(self, sample_smiles):
        """Test vocabulary building."""
        from src.generation.tokenizer import SmilesTokenizer
        
        tokenizer = SmilesTokenizer()
        tokenizer.fit(sample_smiles)
        
        # Should have special tokens + SMILES tokens
        assert tokenizer.vocab_size > 4
        assert "C" in tokenizer.vocab
        assert "O" in tokenizer.vocab
    
    def test_encode_decode(self, sample_smiles):
        """Test encode-decode roundtrip."""
        from src.generation.tokenizer import SmilesTokenizer
        
        tokenizer = SmilesTokenizer()
        tokenizer.fit(sample_smiles)
        
        smiles = "CCO"
        encoded = tokenizer.encode(smiles, add_special=True, pad=True)
        decoded = tokenizer.decode(encoded, remove_special=True)
        
        assert decoded == smiles
    
    def test_batch_encode(self, sample_smiles):
        """Test batch encoding."""
        from src.generation.tokenizer import SmilesTokenizer
        
        tokenizer = SmilesTokenizer(max_length=50)
        tokenizer.fit(sample_smiles)
        
        batch = tokenizer.batch_encode(sample_smiles[:3])
        
        assert batch.shape == (3, 50)
        assert batch.dtype == np.int64
    
    def test_save_load(self, sample_smiles):
        """Test tokenizer save and load."""
        from src.generation.tokenizer import SmilesTokenizer
        
        tokenizer = SmilesTokenizer()
        tokenizer.fit(sample_smiles)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        
        try:
            tokenizer.save(path)
            loaded = SmilesTokenizer.load(path)
            
            assert loaded.vocab_size == tokenizer.vocab_size
            assert loaded.vocab == tokenizer.vocab
        finally:
            os.unlink(path)


class TestSmilesVAE:
    """Tests for SmilesVAE."""
    
    def test_vae_initialization(self):
        """Test VAE initialization."""
        from src.generation.vae import SmilesVAE
        
        vae = SmilesVAE(vocab_size=100, latent_dim=128)
        
        assert vae.vocab_size == 100
        assert vae.latent_dim == 128
        assert vae.encoder is not None
        assert vae.decoder is not None
    
    def test_encoder_forward(self):
        """Test encoder forward pass."""
        from src.generation.vae import VAEEncoder
        
        encoder = VAEEncoder(vocab_size=100, latent_dim=64)
        
        x = torch.randint(0, 100, (4, 50))  # Batch of 4, length 50
        mu, logvar = encoder(x)
        
        assert mu.shape == (4, 64)
        assert logvar.shape == (4, 64)
    
    def test_decoder_forward(self):
        """Test decoder forward pass."""
        from src.generation.vae import VAEDecoder
        
        decoder = VAEDecoder(vocab_size=100, latent_dim=64, max_length=50)
        
        z = torch.randn(4, 64)
        target = torch.randint(0, 100, (4, 50))
        
        outputs = decoder(z, target)
        
        assert outputs.shape == (4, 50, 100)
    
    def test_vae_forward(self):
        """Test VAE forward pass."""
        from src.generation.vae import SmilesVAE
        
        vae = SmilesVAE(vocab_size=100, latent_dim=64, max_length=50)
        
        x = torch.randint(0, 100, (4, 50))
        outputs, mu, logvar = vae(x)
        
        assert outputs.shape == (4, 50, 100)
        assert mu.shape == (4, 64)
        assert logvar.shape == (4, 64)
    
    def test_reparameterize(self):
        """Test reparameterization trick."""
        from src.generation.vae import SmilesVAE
        
        vae = SmilesVAE(vocab_size=100, latent_dim=64)
        
        mu = torch.zeros(4, 64)
        logvar = torch.zeros(4, 64)
        
        # In eval mode, should return mu
        vae.eval()
        z = vae.reparameterize(mu, logvar)
        assert torch.allclose(z, mu)
        
        # In train mode, should be different (stochastic)
        vae.train()
        z = vae.reparameterize(mu, logvar)
        # Can't assert exact values, but shape should match
        assert z.shape == mu.shape
    
    def test_loss_function(self):
        """Test VAE loss computation."""
        from src.generation.vae import SmilesVAE
        
        vae = SmilesVAE(vocab_size=100, latent_dim=64, max_length=50)
        
        outputs = torch.randn(4, 50, 100)
        targets = torch.randint(0, 100, (4, 50))
        mu = torch.randn(4, 64)
        logvar = torch.randn(4, 64)
        
        losses = vae.loss_function(outputs, targets, mu, logvar)
        
        assert 'loss' in losses
        assert 'recon_loss' in losses
        assert 'kl_loss' in losses
        assert losses['loss'].requires_grad
    
    def test_generate(self):
        """Test molecule generation."""
        from src.generation.vae import SmilesVAE
        
        vae = SmilesVAE(vocab_size=100, latent_dim=64, max_length=50)
        vae.eval()
        
        generated = vae.generate(num_samples=5)
        
        assert generated.shape[0] == 5
        assert generated.shape[1] <= 50
    
    def test_interpolate(self):
        """Test latent space interpolation."""
        from src.generation.vae import SmilesVAE
        
        vae = SmilesVAE(vocab_size=100, latent_dim=64, max_length=50)
        vae.eval()
        
        x1 = torch.randint(0, 100, (1, 50))
        x2 = torch.randint(0, 100, (1, 50))
        
        interpolations = vae.interpolate(x1, x2, num_steps=5)
        
        assert len(interpolations) == 5
    
    def test_save_load(self):
        """Test VAE save and load."""
        from src.generation.vae import SmilesVAE
        
        vae = SmilesVAE(vocab_size=100, latent_dim=64)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        
        try:
            vae.save(path)
            loaded = SmilesVAE.load(path)
            
            assert loaded.vocab_size == vae.vocab_size
            assert loaded.latent_dim == vae.latent_dim
        finally:
            os.unlink(path)


class TestMoleculeGenerator:
    """Tests for MoleculeGenerator."""
    
    def test_generator_initialization(self, sample_smiles):
        """Test generator initialization."""
        from src.generation.tokenizer import SmilesTokenizer
        from src.generation.generator import MoleculeGenerator
        
        tokenizer = SmilesTokenizer()
        tokenizer.fit(sample_smiles)
        
        generator = MoleculeGenerator(tokenizer)
        
        assert generator.tokenizer is not None
        assert generator.vae is not None
    
    def test_generate_simple(self, sample_smiles):
        """Test simple generation."""
        from src.generation.tokenizer import SmilesTokenizer
        from src.generation.generator import MoleculeGenerator
        
        tokenizer = SmilesTokenizer(max_length=50)
        tokenizer.fit(sample_smiles)
        
        generator = MoleculeGenerator(tokenizer)
        
        # Generate without training (random)
        generated = generator.generate(num_samples=5, valid_only=False)
        
        assert len(generated) == 5


class TestValidation:
    """Tests for validation functions."""
    
    @pytest.mark.skipif(not HAS_RDKIT, reason="Requires RDKit")
    def test_validate_smiles_valid(self):
        """Test validation of valid SMILES."""
        from src.generation.generator import validate_smiles
        
        assert validate_smiles("CCO") == True
        assert validate_smiles("c1ccccc1") == True
        assert validate_smiles("CC(=O)O") == True
    
    @pytest.mark.skipif(not HAS_RDKIT, reason="Requires RDKit")
    def test_validate_smiles_invalid(self):
        """Test validation of invalid SMILES."""
        from src.generation.generator import validate_smiles
        
        assert validate_smiles("invalid") == False
        assert validate_smiles("") == False
        assert validate_smiles("C(C(C") == False
    
    @pytest.mark.skipif(not HAS_RDKIT, reason="Requires RDKit")
    def test_calculate_properties(self):
        """Test property calculation."""
        from src.generation.generator import calculate_properties
        
        props = calculate_properties("CCO")
        
        assert props['valid'] == True
        assert 'mw' in props
        assert 'logp' in props
        assert 'qed' in props
        assert props['mw'] > 0
    
    @pytest.mark.skipif(not HAS_RDKIT, reason="Requires RDKit")
    def test_calculate_properties_invalid(self):
        """Test properties for invalid SMILES."""
        from src.generation.generator import calculate_properties
        
        props = calculate_properties("invalid")
        
        assert props['valid'] == False


class TestLatentOptimizer:
    """Tests for LatentOptimizer."""
    
    def test_property_predictor(self):
        """Test property predictor."""
        from src.generation.optimizer import PropertyPredictor
        
        predictor = PropertyPredictor(latent_dim=64)
        
        z = torch.randn(10, 64)
        pred = predictor(z)
        
        assert pred.shape == (10, 1)
        assert (pred >= 0).all() and (pred <= 1).all()  # Sigmoid output
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        from src.generation.vae import SmilesVAE
        from src.generation.optimizer import LatentOptimizer
        
        vae = SmilesVAE(vocab_size=100, latent_dim=64)
        optimizer = LatentOptimizer(vae)
        
        assert optimizer.vae is not None


class TestGenerationMetrics:
    """Tests for generation metrics."""
    
    @pytest.mark.skipif(not HAS_RDKIT, reason="Requires RDKit")
    def test_validity_metric(self):
        """Test validity calculation."""
        from src.generation.generator import GenerationMetrics
        
        smiles = ["CCO", "invalid", "c1ccccc1", "bad"]
        validity = GenerationMetrics.validity(smiles)
        
        assert validity == 0.5  # 2 out of 4
    
    @pytest.mark.skipif(not HAS_RDKIT, reason="Requires RDKit")
    def test_uniqueness_metric(self):
        """Test uniqueness calculation."""
        from src.generation.generator import GenerationMetrics
        
        smiles = ["CCO", "CCO", "CCO", "c1ccccc1"]
        uniqueness = GenerationMetrics.uniqueness(smiles)
        
        assert uniqueness == 0.5  # 2 unique out of 4
    
    @pytest.mark.skipif(not HAS_RDKIT, reason="Requires RDKit")
    def test_novelty_metric(self):
        """Test novelty calculation."""
        from src.generation.generator import GenerationMetrics
        
        generated = ["CCO", "CCCO", "CCCCO"]
        reference = ["CCO", "CC"]
        
        novelty = GenerationMetrics.novelty(generated, reference)
        
        # 2 novel (CCCO, CCCCO) out of 3
        assert abs(novelty - 2/3) < 0.01


class TestSmilesStatistics:
    """Tests for SMILES statistics."""
    
    def test_get_statistics(self, sample_smiles):
        """Test statistics calculation."""
        from src.generation.tokenizer import get_smiles_statistics
        
        stats = get_smiles_statistics(sample_smiles)
        
        assert 'num_molecules' in stats
        assert 'avg_length' in stats
        assert 'max_length' in stats
        assert stats['num_molecules'] == len(sample_smiles)
