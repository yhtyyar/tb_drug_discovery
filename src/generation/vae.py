"""Variational Autoencoder for SMILES generation.

This module implements a VAE architecture for learning continuous
representations of molecules and generating novel SMILES strings.

Architecture:
- Encoder: GRU-based sequence encoder
- Latent space: Gaussian with reparameterization trick
- Decoder: GRU-based sequence decoder with teacher forcing

Example:
    >>> vae = SmilesVAE(vocab_size=100, latent_dim=256)
    >>> z, mu, logvar = vae.encode(x)
    >>> reconstructed = vae.decode(z)
    >>> loss = vae.loss_function(reconstructed, x, mu, logvar)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class VAEEncoder(nn.Module):
    """GRU-based encoder for SMILES VAE.
    
    Encodes SMILES sequences to latent space representations.
    
    Args:
        vocab_size: Size of vocabulary.
        embed_dim: Embedding dimension.
        hidden_dim: GRU hidden dimension.
        latent_dim: Latent space dimension.
        num_layers: Number of GRU layers.
        dropout: Dropout probability.
        bidirectional: Use bidirectional GRU.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # GRU encoder
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Latent space projections
        encoder_output_dim = hidden_dim * self.num_directions * num_layers
        self.fc_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode sequences to latent parameters.
        
        Args:
            x: Input tensor of shape (batch, seq_len).
            
        Returns:
            Tuple of (mu, logvar) each of shape (batch, latent_dim).
        """
        # Embed
        embedded = self.embedding(x)
        
        # Encode
        _, hidden = self.gru(embedded)
        
        # Reshape hidden state: (num_layers * num_directions, batch, hidden) -> (batch, -1)
        hidden = hidden.permute(1, 0, 2).contiguous()
        hidden = hidden.view(hidden.size(0), -1)
        
        # Project to latent space
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        return mu, logvar


class VAEDecoder(nn.Module):
    """GRU-based decoder for SMILES VAE.
    
    Decodes latent vectors to SMILES sequences.
    
    Args:
        vocab_size: Size of vocabulary.
        embed_dim: Embedding dimension.
        hidden_dim: GRU hidden dimension.
        latent_dim: Latent space dimension.
        num_layers: Number of GRU layers.
        dropout: Dropout probability.
        max_length: Maximum sequence length.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_length: int = 120,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_length = max_length
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Latent to hidden projection
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        
        # GRU decoder
        self.gru = nn.GRU(
            embed_dim + latent_dim,  # Concatenate embedding with latent
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
    
    def forward(
        self,
        z: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        """Decode latent vectors to sequences.
        
        Args:
            z: Latent vectors of shape (batch, latent_dim).
            target: Target sequences for teacher forcing (batch, seq_len).
            teacher_forcing_ratio: Probability of using teacher forcing.
            
        Returns:
            Output logits of shape (batch, seq_len, vocab_size).
        """
        batch_size = z.size(0)
        device = z.device
        
        # Initialize hidden state from latent
        hidden = self.fc_hidden(z)
        hidden = hidden.view(batch_size, self.num_layers, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous()
        
        # Determine sequence length
        if target is not None:
            seq_len = target.size(1)
        else:
            seq_len = self.max_length
        
        # Initialize output tensor
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)
        
        # Start token (index 1)
        input_token = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        for t in range(seq_len):
            # Embed input
            embedded = self.embedding(input_token)
            
            # Concatenate with latent vector
            z_expanded = z.unsqueeze(1)
            gru_input = torch.cat([embedded, z_expanded], dim=2)
            
            # GRU step
            output, hidden = self.gru(gru_input, hidden)
            
            # Project to vocabulary
            logits = self.fc_out(output.squeeze(1))
            outputs[:, t, :] = logits
            
            # Determine next input
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing
                input_token = target[:, t:t+1]
            else:
                # Use own prediction
                input_token = logits.argmax(dim=1, keepdim=True)
        
        return outputs
    
    def generate(
        self,
        z: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Generate sequences from latent vectors.
        
        Args:
            z: Latent vectors of shape (batch, latent_dim).
            max_length: Maximum sequence length.
            temperature: Sampling temperature.
            top_k: Top-k sampling (0 = greedy).
            
        Returns:
            Generated token indices of shape (batch, seq_len).
        """
        if max_length is None:
            max_length = self.max_length
        
        batch_size = z.size(0)
        device = z.device
        
        # Initialize hidden state
        hidden = self.fc_hidden(z)
        hidden = hidden.view(batch_size, self.num_layers, self.hidden_dim)
        hidden = hidden.permute(1, 0, 2).contiguous()
        
        # Start token
        input_token = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        # Generate tokens
        generated = [input_token]
        
        for _ in range(max_length - 1):
            embedded = self.embedding(input_token)
            z_expanded = z.unsqueeze(1)
            gru_input = torch.cat([embedded, z_expanded], dim=2)
            
            output, hidden = self.gru(gru_input, hidden)
            logits = self.fc_out(output.squeeze(1))
            
            # Apply temperature
            logits = logits / temperature
            
            # Sample next token
            if top_k > 0:
                # Top-k sampling
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                probs = F.softmax(top_k_logits, dim=-1)
                sampled_idx = torch.multinomial(probs, 1)
                input_token = top_k_indices.gather(-1, sampled_idx)
            else:
                # Greedy
                input_token = logits.argmax(dim=-1, keepdim=True)
            
            generated.append(input_token)
            
            # Stop if all sequences have END token (index 2)
            if (input_token == 2).all():
                break
        
        return torch.cat(generated, dim=1)


class SmilesVAE(nn.Module):
    """Variational Autoencoder for SMILES generation.
    
    Combines encoder and decoder with reparameterization trick
    for end-to-end training.
    
    Args:
        vocab_size: Size of vocabulary.
        embed_dim: Embedding dimension.
        hidden_dim: RNN hidden dimension.
        latent_dim: Latent space dimension.
        num_layers: Number of RNN layers.
        dropout: Dropout probability.
        max_length: Maximum sequence length.
        
    Example:
        >>> vae = SmilesVAE(vocab_size=100, latent_dim=256)
        >>> outputs, mu, logvar = vae(x)
        >>> loss = vae.loss_function(outputs, x, mu, logvar)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_length: int = 120,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.max_length = max_length
        
        # Encoder
        self.encoder = VAEEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # Decoder
        self.decoder = VAEDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            max_length=max_length,
        )
        
        logger.info(f"SmilesVAE initialized: latent_dim={latent_dim}, vocab_size={vocab_size}")
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling.
        
        Args:
            mu: Mean of latent distribution.
            logvar: Log variance of latent distribution.
            
        Returns:
            Sampled latent vector.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode sequences to latent space.
        
        Args:
            x: Input sequences of shape (batch, seq_len).
            
        Returns:
            Tuple of (z, mu, logvar).
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(
        self,
        z: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        """Decode latent vectors to sequences.
        
        Args:
            z: Latent vectors.
            target: Target sequences for teacher forcing.
            teacher_forcing_ratio: Teacher forcing probability.
            
        Returns:
            Output logits.
        """
        return self.decoder(z, target, teacher_forcing_ratio)
    
    def forward(
        self,
        x: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE.
        
        Args:
            x: Input sequences.
            teacher_forcing_ratio: Teacher forcing probability.
            
        Returns:
            Tuple of (output_logits, mu, logvar).
        """
        z, mu, logvar = self.encode(x)
        outputs = self.decode(z, x, teacher_forcing_ratio)
        return outputs, mu, logvar
    
    def loss_function(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE loss.
        
        Args:
            outputs: Predicted logits (batch, seq_len, vocab_size).
            targets: Target sequences (batch, seq_len).
            mu: Latent mean.
            logvar: Latent log variance.
            kl_weight: Weight for KL divergence (for annealing).
            
        Returns:
            Dictionary with 'loss', 'recon_loss', 'kl_loss'.
        """
        # Reconstruction loss (cross entropy)
        outputs_flat = outputs.view(-1, self.vocab_size)
        targets_flat = targets.view(-1)
        
        # Ignore padding (index 0)
        recon_loss = F.cross_entropy(
            outputs_flat, targets_flat, 
            ignore_index=0, reduction='mean'
        )
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        loss = recon_loss + kl_weight * kl_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
        }
    
    def generate(
        self,
        num_samples: int = 1,
        temperature: float = 1.0,
        top_k: int = 0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate novel SMILES from random latent vectors.
        
        Args:
            num_samples: Number of molecules to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling.
            device: Device for generation.
            
        Returns:
            Generated token indices.
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Sample from prior
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Generate
        self.eval()
        with torch.no_grad():
            generated = self.decoder.generate(z, temperature=temperature, top_k=top_k)
        
        return generated
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input sequences.
        
        Args:
            x: Input sequences.
            
        Returns:
            Reconstructed token indices.
        """
        self.eval()
        with torch.no_grad():
            z, _, _ = self.encode(x)
            reconstructed = self.decoder.generate(z)
        return reconstructed
    
    def interpolate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        num_steps: int = 10,
    ) -> List[torch.Tensor]:
        """Interpolate between two molecules in latent space.
        
        Args:
            x1: First sequence.
            x2: Second sequence.
            num_steps: Number of interpolation steps.
            
        Returns:
            List of generated sequences along interpolation path.
        """
        self.eval()
        with torch.no_grad():
            z1, _, _ = self.encode(x1)
            z2, _, _ = self.encode(x2)
            
            interpolations = []
            for alpha in np.linspace(0, 1, num_steps):
                z = (1 - alpha) * z1 + alpha * z2
                generated = self.decoder.generate(z)
                interpolations.append(generated)
        
        return interpolations
    
    def save(self, path: str) -> None:
        """Save model state."""
        torch.save({
            'state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'latent_dim': self.latent_dim,
            'max_length': self.max_length,
        }, path)
        logger.info(f"VAE saved: {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'SmilesVAE':
        """Load model from file."""
        if device is None:
            device = torch.device('cpu')
        
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            latent_dim=checkpoint['latent_dim'],
            max_length=checkpoint['max_length'],
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        
        logger.info(f"VAE loaded: {path}")
        return model
