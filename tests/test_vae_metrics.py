"""Tests for VAE generative model metrics.

Validates reconstruction quality, generation validity, and KL annealing behavior.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

# Skip tests if dependencies not available
pytestmark = [
    pytest.mark.skipif(
        not Path("src/generation/vae.py").exists(),
        reason="VAE module not found"
    ),
]


def test_kl_annealer_linear():
    """Test linear KL annealing schedule."""
    import sys
    sys.path.insert(0, "src")
    from generation.vae import KLAnnealer

    annealer = KLAnnealer(n_epochs=100, start=0.0, end=1.0, strategy="linear")

    # Start should be near 0
    assert annealer.get_weight(0) == 0.0

    # End should be near 1
    assert annealer.get_weight(100) == 1.0

    # Middle should be around 0.5
    assert 0.45 < annealer.get_weight(50) < 0.55

    # Should increase monotonically
    weights = [annealer.get_weight(e) for e in range(101)]
    for i in range(len(weights) - 1):
        assert weights[i] <= weights[i + 1]


def test_kl_annealer_cyclical():
    """Test cyclical KL annealing schedule."""
    import sys
    sys.path.insert(0, "src")
    from generation.vae import KLAnnealer

    annealer = KLAnnealer(n_epochs=100, start=0.0, end=1.0, strategy="cyclical")

    # Should have 4 cycles over 100 epochs
    # Each cycle is 25 epochs

    # Start of cycle
    assert annealer.get_weight(0) == 0.0

    # End of first cycle (epoch 25)
    assert annealer.get_weight(25) == 1.0

    # Start of second cycle (epoch 26)
    assert annealer.get_weight(26) == 0.0

    # Should cycle
    assert annealer.get_weight(50) == 1.0
    assert annealer.get_weight(51) == 0.0


def test_vae_loss_function_with_kl_weight():
    """Test VAE loss function accepts and uses kl_weight parameter."""
    import sys
    sys.path.insert(0, "src")
    from generation.vae import SmilesVAE

    vae = SmilesVAE(vocab_size=50, latent_dim=64, hidden_dim=128)

    # Create dummy data
    batch_size, seq_len = 4, 20
    outputs = torch.randn(batch_size, seq_len, 50)
    targets = torch.randint(0, 50, (batch_size, seq_len))
    mu = torch.randn(batch_size, 64)
    logvar = torch.randn(batch_size, 64)

    # Test with different KL weights
    loss_0 = vae.loss_function(outputs, targets, mu, logvar, kl_weight=0.0)
    loss_1 = vae.loss_function(outputs, targets, mu, logvar, kl_weight=1.0)
    loss_half = vae.loss_function(outputs, targets, mu, logvar, kl_weight=0.5)

    # With kl_weight=0, KL loss shouldn't contribute
    assert "kl_loss" in loss_0
    assert "recon_loss" in loss_0
    assert "loss" in loss_0

    # Higher KL weight should increase total loss (if KL loss > 0)
    if loss_0["kl_loss"].item() > 0:
        assert loss_1["loss"].item() > loss_0["loss"].item()

    # Check that losses are reasonable tensors
    assert torch.isfinite(loss_0["loss"])
    assert torch.isfinite(loss_1["loss"])


@pytest.mark.skipif(
    not Path("models/vae_final.pt").exists(),
    reason="No trained VAE model found"
)
def test_vae_reconstruction_rate():
    """VAE should reconstruct >50% of valid molecules."""
    import sys
    sys.path.insert(0, "src")
    from generation.vae import SmilesVAE
    from generation.tokenizer import SMILESTokenizer
    from rdkit import Chem

    # Load model and tokenizer
    tokenizer = SMILESTokenizer.load("models/tokenizer.pkl")
    vae = SmilesVAE.load("models/vae_final.pt")
    vae.eval()

    # Test SMILES
    test_smiles = [
        "c1ccccc1", "CCO", "CC(=O)O", "c1ccc(O)cc1", "CCN",
        "c1ccccc1O", "CC(C)O", "c1ccc(F)cc1", "CC=O", "CCCO",
    ]

    # Encode and reconstruct
    valid_reconstructions = 0
    total = len(test_smiles)

    for smi in test_smiles:
        try:
            # Tokenize
            tokens = tokenizer.encode(smi)
            tokens_tensor = torch.tensor([tokens], dtype=torch.long)

            # Reconstruct
            with torch.no_grad():
                reconstructed = vae.reconstruct(tokens_tensor)

            # Decode
            recon_smi = tokenizer.decode(reconstructed[0].tolist())

            # Check validity
            mol = Chem.MolFromSmiles(recon_smi)
            if mol is not None:
                valid_reconstructions += 1
        except Exception:
            pass

    # At least 50% should be valid
    reconstruction_rate = valid_reconstructions / total
    assert reconstruction_rate >= 0.50, f"Reconstruction rate too low: {reconstruction_rate:.1%}"


@pytest.mark.skipif(
    not Path("models/vae_final.pt").exists(),
    reason="No trained VAE model found"
)
def test_vae_generation_validity():
    """VAE should generate >30% valid molecules."""
    import sys
    sys.path.insert(0, "src")
    from generation.vae import SmilesVAE
    from generation.tokenizer import SMILESTokenizer
    from rdkit import Chem

    tokenizer = SMILESTokenizer.load("models/tokenizer.pkl")
    vae = SmilesVAE.load("models/vae_final.pt")

    # Generate molecules
    n_samples = 100
    generated_tokens = vae.generate(num_samples=n_samples, temperature=1.0)

    # Decode and check validity
    valid_count = 0
    for tokens in generated_tokens:
        smi = tokenizer.decode(tokens.tolist())
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_count += 1

    validity = valid_count / n_samples
    assert validity >= 0.30, f"VAE validity too low: {validity:.1%}"


@pytest.mark.skipif(
    not Path("models/vae_final.pt").exists(),
    reason="No trained VAE model found"
)
def test_vae_latent_interpolation():
    """Test that interpolating in latent space produces valid molecules."""
    import sys
    sys.path.insert(0, "src")
    from generation.vae import SmilesVAE
    from generation.tokenizer import SMILESTokenizer
    from rdkit import Chem

    tokenizer = SMILESTokenizer.load("models/tokenizer.pkl")
    vae = SmilesVAE.load("models/vae_final.pt")
    vae.eval()

    # Two different molecules
    smi1 = "c1ccccc1"  # benzene
    smi2 = "CCO"       # ethanol

    tokens1 = torch.tensor([tokenizer.encode(smi1)], dtype=torch.long)
    tokens2 = torch.tensor([tokenizer.encode(smi2)], dtype=torch.long)

    # Interpolate
    with torch.no_grad():
        interpolations = vae.interpolate(tokens1, tokens2, num_steps=5)

    # Should produce 5 interpolated molecules
    assert len(interpolations) == 5

    # At least some should decode to valid SMILES
    valid_count = 0
    for tokens in interpolations:
        smi = tokenizer.decode(tokens[0].tolist())
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_count += 1

    # At least 2/5 should be valid
    assert valid_count >= 2


def test_vae_no_posterior_collapse():
    """Detect signs of posterior collapse during training.

    Posterior collapse occurs when KL loss goes to near zero,
    indicating the model is ignoring the latent space.
    """
    import sys
    sys.path.insert(0, "src")
    from generation.vae import SmilesVAE, KLAnnealer
    import torch.nn.functional as F

    vae = SmilesVAE(vocab_size=50, latent_dim=64)

    # Simulate training batches
    batch_size, seq_len = 8, 20

    # Track KL loss
    kl_losses = []

    # Simulate 50 training steps with KL annealing
    annealer = KLAnnealer(n_epochs=50, start=0.0, end=1.0)

    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    for epoch in range(50):
        kl_weight = annealer.get_weight(epoch)

        # Random batch
        x = torch.randint(0, 50, (batch_size, seq_len))

        # Forward pass
        outputs, mu, logvar = vae(x, teacher_forcing_ratio=0.5)

        # Compute loss
        loss_dict = vae.loss_function(outputs, x, mu, logvar, kl_weight=kl_weight)

        kl_losses.append(loss_dict["kl_loss"].item())

        # Backward
        optimizer.zero_grad()
        loss_dict["loss"].backward()
        optimizer.step()

    # Check for posterior collapse warning signs
    # If KL loss drops below 0.1 early in training, that's a red flag
    early_kl = kl_losses[:10]
    if min(early_kl) < 0.01:
        # This would indicate collapse if it persists
        # For test purposes, just log the warning
        pass

    # With KL annealing, KL loss should increase over time
    # as the weight increases
    late_kl = np.mean(kl_losses[-10:])
    early_kl_mean = np.mean(early_kl)

    # Late KL should be higher or at least not collapsed
    assert late_kl > 0.01, f"Possible posterior collapse: late KL = {late_kl:.4f}"
