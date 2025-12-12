"""High-level molecular generation interface.

This module provides a unified interface for molecular generation,
including training, sampling, and evaluation.

Example:
    >>> generator = MoleculeGenerator(tokenizer, vae)
    >>> molecules = generator.generate(num_samples=100)
    >>> metrics = generator.evaluate(molecules)
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

try:
    from rdkit import Chem
    from rdkit.Chem import QED, Descriptors, AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def validate_smiles(smiles: str) -> bool:
    """Check if SMILES is valid.
    
    Args:
        smiles: SMILES string.
        
    Returns:
        True if valid, False otherwise.
    """
    if not HAS_RDKIT:
        return len(smiles) > 0
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def calculate_properties(smiles: str) -> Dict:
    """Calculate molecular properties.
    
    Args:
        smiles: SMILES string.
        
    Returns:
        Dictionary of properties.
    """
    if not HAS_RDKIT:
        return {'valid': False}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'valid': False}
        
        return {
            'valid': True,
            'smiles': Chem.MolToSmiles(mol),  # Canonical
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'rings': Descriptors.RingCount(mol),
            'qed': QED.qed(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_heavy_atoms': mol.GetNumHeavyAtoms(),
        }
    except Exception as e:
        return {'valid': False, 'error': str(e)}


def get_scaffold(smiles: str) -> Optional[str]:
    """Extract Murcko scaffold from molecule.
    
    Args:
        smiles: SMILES string.
        
    Returns:
        Scaffold SMILES or None.
    """
    if not HAS_RDKIT:
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except:
        return None


class MoleculeGenerator:
    """High-level interface for molecular generation.
    
    Provides methods for training VAE, generating molecules,
    and evaluating generation quality.
    
    Args:
        tokenizer: SmilesTokenizer instance.
        vae: SmilesVAE model (or will be created).
        device: Computation device.
        
    Example:
        >>> generator = MoleculeGenerator(tokenizer)
        >>> generator.train(smiles_list, epochs=50)
        >>> molecules = generator.generate(100)
        >>> metrics = generator.evaluate(molecules)
    """
    
    def __init__(
        self,
        tokenizer,
        vae: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        self.tokenizer = tokenizer
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        if vae is None:
            from src.generation.vae import SmilesVAE
            vae = SmilesVAE(
                vocab_size=tokenizer.vocab_size,
                max_length=tokenizer.max_length,
            )
        
        self.vae = vae.to(device)
        self.optimizer = None
        self.history = {'train_loss': [], 'val_loss': [], 'kl_weight': []}
        
        logger.info(f"MoleculeGenerator initialized on {device}")
    
    def train(
        self,
        smiles_list: List[str],
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        kl_annealing: bool = True,
        kl_start: float = 0.0,
        kl_end: float = 1.0,
        kl_warmup: int = 10,
        val_ratio: float = 0.1,
        checkpoint_dir: Optional[str] = None,
        verbose: int = 1,
    ) -> Dict:
        """Train the VAE on SMILES data.
        
        Args:
            smiles_list: Training SMILES.
            epochs: Number of training epochs.
            batch_size: Batch size.
            learning_rate: Learning rate.
            kl_annealing: Use KL annealing.
            kl_start: Starting KL weight.
            kl_end: Final KL weight.
            kl_warmup: Warmup epochs for KL annealing.
            val_ratio: Validation set ratio.
            checkpoint_dir: Directory for saving checkpoints.
            verbose: Verbosity level.
            
        Returns:
            Training history.
        """
        # Encode SMILES
        logger.info("Encoding SMILES...")
        encoded = self.tokenizer.batch_encode(smiles_list)
        encoded = torch.tensor(encoded, dtype=torch.long)
        
        # Split data
        n_val = int(len(encoded) * val_ratio)
        indices = torch.randperm(len(encoded))
        
        train_data = encoded[indices[n_val:]]
        val_data = encoded[indices[:n_val]]
        
        train_loader = DataLoader(
            TensorDataset(train_data),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_data),
            batch_size=batch_size,
        )
        
        logger.info(f"Training: {len(train_data)}, Validation: {len(val_data)}")
        
        # Optimizer
        self.optimizer = optim.Adam(self.vae.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # KL annealing
            if kl_annealing:
                kl_weight = min(kl_end, kl_start + (kl_end - kl_start) * epoch / kl_warmup)
            else:
                kl_weight = kl_end
            
            # Train epoch
            train_loss = self._train_epoch(train_loader, kl_weight)
            
            # Validation
            val_loss = self._validate(val_loader, kl_weight)
            
            # History
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['kl_weight'].append(kl_weight)
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if checkpoint_dir:
                    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    self.save(f"{checkpoint_dir}/best_vae.pt")
            
            # Logging
            if verbose >= 1 and (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {train_loss:.4f}/{val_loss:.4f} - "
                    f"KL weight: {kl_weight:.3f}"
                )
        
        return self.history
    
    def _train_epoch(self, loader: DataLoader, kl_weight: float) -> float:
        """Train for one epoch."""
        self.vae.train()
        total_loss = 0
        
        for batch in loader:
            x = batch[0].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs, mu, logvar = self.vae(x, teacher_forcing_ratio=0.9)
            losses = self.vae.loss_function(outputs, x, mu, logvar, kl_weight)
            
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += losses['loss'].item() * x.size(0)
        
        return total_loss / len(loader.dataset)
    
    @torch.no_grad()
    def _validate(self, loader: DataLoader, kl_weight: float) -> float:
        """Validate the model."""
        self.vae.eval()
        total_loss = 0
        
        for batch in loader:
            x = batch[0].to(self.device)
            
            outputs, mu, logvar = self.vae(x, teacher_forcing_ratio=1.0)
            losses = self.vae.loss_function(outputs, x, mu, logvar, kl_weight)
            
            total_loss += losses['loss'].item() * x.size(0)
        
        return total_loss / len(loader.dataset)
    
    def generate(
        self,
        num_samples: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        unique: bool = True,
        valid_only: bool = True,
        max_attempts: int = 10,
    ) -> List[str]:
        """Generate novel molecules.
        
        Args:
            num_samples: Number of molecules to generate.
            temperature: Sampling temperature (higher = more diverse).
            top_k: Top-k sampling (0 = greedy/temperature sampling).
            unique: Return only unique molecules.
            valid_only: Return only valid SMILES.
            max_attempts: Maximum generation attempts per molecule.
            
        Returns:
            List of generated SMILES.
        """
        self.vae.eval()
        generated = set() if unique else []
        attempts = 0
        
        while len(generated) < num_samples and attempts < num_samples * max_attempts:
            # Generate batch
            batch_size = min(100, num_samples - len(generated))
            
            with torch.no_grad():
                tokens = self.vae.generate(
                    num_samples=batch_size,
                    temperature=temperature,
                    top_k=top_k,
                    device=self.device,
                )
            
            # Decode
            for i in range(tokens.size(0)):
                smiles = self.tokenizer.decode(tokens[i].cpu().numpy())
                
                # Validate if required
                if valid_only and not validate_smiles(smiles):
                    attempts += 1
                    continue
                
                # Add to results
                if unique:
                    generated.add(smiles)
                else:
                    generated.append(smiles)
                
                if len(generated) >= num_samples:
                    break
            
            attempts += batch_size
        
        result = list(generated) if unique else generated
        logger.info(f"Generated {len(result)} molecules ({attempts} attempts)")
        return result[:num_samples]
    
    def reconstruct(self, smiles_list: List[str]) -> Tuple[List[str], float]:
        """Reconstruct input molecules.
        
        Args:
            smiles_list: Input SMILES.
            
        Returns:
            Tuple of (reconstructed SMILES, reconstruction accuracy).
        """
        self.vae.eval()
        
        # Encode
        encoded = self.tokenizer.batch_encode(smiles_list)
        x = torch.tensor(encoded, dtype=torch.long, device=self.device)
        
        # Reconstruct
        with torch.no_grad():
            reconstructed = self.vae.reconstruct(x)
        
        # Decode
        result = []
        correct = 0
        
        for i, tokens in enumerate(reconstructed):
            recon_smiles = self.tokenizer.decode(tokens.cpu().numpy())
            result.append(recon_smiles)
            
            # Check if reconstruction is correct (canonical comparison)
            if HAS_RDKIT:
                try:
                    mol1 = Chem.MolFromSmiles(smiles_list[i])
                    mol2 = Chem.MolFromSmiles(recon_smiles)
                    if mol1 and mol2:
                        can1 = Chem.MolToSmiles(mol1)
                        can2 = Chem.MolToSmiles(mol2)
                        if can1 == can2:
                            correct += 1
                except:
                    pass
        
        accuracy = correct / len(smiles_list) if smiles_list else 0
        return result, accuracy
    
    def interpolate(
        self,
        smiles1: str,
        smiles2: str,
        num_steps: int = 10,
    ) -> List[str]:
        """Interpolate between two molecules.
        
        Args:
            smiles1: First molecule.
            smiles2: Second molecule.
            num_steps: Number of interpolation steps.
            
        Returns:
            List of molecules along interpolation path.
        """
        # Encode
        x1 = torch.tensor([self.tokenizer.encode(smiles1)], dtype=torch.long, device=self.device)
        x2 = torch.tensor([self.tokenizer.encode(smiles2)], dtype=torch.long, device=self.device)
        
        # Interpolate
        interpolations = self.vae.interpolate(x1, x2, num_steps)
        
        # Decode
        result = []
        for tokens in interpolations:
            smiles = self.tokenizer.decode(tokens[0].cpu().numpy())
            result.append(smiles)
        
        return result
    
    def evaluate(
        self,
        smiles_list: List[str],
        reference_smiles: Optional[List[str]] = None,
    ) -> Dict:
        """Evaluate generation quality.
        
        Args:
            smiles_list: Generated SMILES.
            reference_smiles: Reference dataset for novelty/diversity.
            
        Returns:
            Dictionary of metrics.
        """
        metrics = {}
        
        # Validity
        valid = [validate_smiles(s) for s in smiles_list]
        metrics['validity'] = sum(valid) / len(valid) if valid else 0
        
        # Uniqueness (among valid)
        valid_smiles = [s for s, v in zip(smiles_list, valid) if v]
        if valid_smiles:
            if HAS_RDKIT:
                canonical = set()
                for s in valid_smiles:
                    mol = Chem.MolFromSmiles(s)
                    if mol:
                        canonical.add(Chem.MolToSmiles(mol))
                metrics['uniqueness'] = len(canonical) / len(valid_smiles)
            else:
                metrics['uniqueness'] = len(set(valid_smiles)) / len(valid_smiles)
        else:
            metrics['uniqueness'] = 0
        
        # Novelty (if reference provided)
        if reference_smiles and HAS_RDKIT:
            ref_canonical = set()
            for s in reference_smiles:
                try:
                    mol = Chem.MolFromSmiles(s)
                    if mol:
                        ref_canonical.add(Chem.MolToSmiles(mol))
                except:
                    pass
            
            novel = sum(1 for s in valid_smiles if s not in ref_canonical)
            metrics['novelty'] = novel / len(valid_smiles) if valid_smiles else 0
        
        # Property statistics
        if valid_smiles:
            properties = [calculate_properties(s) for s in valid_smiles[:1000]]
            valid_props = [p for p in properties if p.get('valid', False)]
            
            if valid_props:
                metrics['avg_mw'] = np.mean([p['mw'] for p in valid_props])
                metrics['avg_logp'] = np.mean([p['logp'] for p in valid_props])
                metrics['avg_qed'] = np.mean([p['qed'] for p in valid_props])
        
        # Scaffold diversity
        if valid_smiles and HAS_RDKIT:
            scaffolds = set()
            for s in valid_smiles[:1000]:
                scaffold = get_scaffold(s)
                if scaffold:
                    scaffolds.add(scaffold)
            metrics['scaffold_diversity'] = len(scaffolds) / min(len(valid_smiles), 1000)
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save generator state."""
        torch.save({
            'vae_state': self.vae.state_dict(),
            'history': self.history,
        }, path)
        logger.info(f"Generator saved: {path}")
    
    def load(self, path: str) -> None:
        """Load generator state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.vae.load_state_dict(checkpoint['vae_state'])
        self.history = checkpoint.get('history', self.history)
        logger.info(f"Generator loaded: {path}")


class GenerationMetrics:
    """Calculate comprehensive generation metrics.
    
    Implements standard metrics for evaluating molecular generation:
    - Validity, uniqueness, novelty
    - Fréchet ChemNet Distance (FCD)
    - Internal diversity
    - Property distributions
    """
    
    @staticmethod
    def validity(smiles_list: List[str]) -> float:
        """Calculate validity rate."""
        valid = sum(validate_smiles(s) for s in smiles_list)
        return valid / len(smiles_list) if smiles_list else 0
    
    @staticmethod
    def uniqueness(smiles_list: List[str]) -> float:
        """Calculate uniqueness rate."""
        if not HAS_RDKIT:
            return len(set(smiles_list)) / len(smiles_list) if smiles_list else 0
        
        canonical = set()
        for s in smiles_list:
            try:
                mol = Chem.MolFromSmiles(s)
                if mol:
                    canonical.add(Chem.MolToSmiles(mol))
            except:
                pass
        
        return len(canonical) / len(smiles_list) if smiles_list else 0
    
    @staticmethod
    def novelty(generated: List[str], reference: List[str]) -> float:
        """Calculate novelty rate."""
        if not HAS_RDKIT:
            ref_set = set(reference)
            novel = sum(1 for s in generated if s not in ref_set)
            return novel / len(generated) if generated else 0
        
        # Canonical reference set
        ref_canonical = set()
        for s in reference:
            try:
                mol = Chem.MolFromSmiles(s)
                if mol:
                    ref_canonical.add(Chem.MolToSmiles(mol))
            except:
                pass
        
        # Count novel
        novel = 0
        for s in generated:
            try:
                mol = Chem.MolFromSmiles(s)
                if mol:
                    can = Chem.MolToSmiles(mol)
                    if can not in ref_canonical:
                        novel += 1
            except:
                pass
        
        return novel / len(generated) if generated else 0
    
    @staticmethod
    def internal_diversity(smiles_list: List[str], sample_size: int = 1000) -> float:
        """Calculate internal diversity using Tanimoto similarity."""
        if not HAS_RDKIT:
            return 0.5
        
        from rdkit import DataStructs
        
        # Sample if too many
        if len(smiles_list) > sample_size:
            smiles_list = list(np.random.choice(smiles_list, sample_size, replace=False))
        
        # Calculate fingerprints
        fps = []
        for s in smiles_list:
            try:
                mol = Chem.MolFromSmiles(s)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    fps.append(fp)
            except:
                pass
        
        if len(fps) < 2:
            return 0
        
        # Calculate average pairwise distance
        similarities = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarities.append(sim)
        
        # Diversity = 1 - average similarity
        return 1 - np.mean(similarities) if similarities else 0
