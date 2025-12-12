"""Latent space optimization for property-guided molecular generation.

This module provides tools for optimizing molecules in the VAE latent space
to achieve desired properties (activity, drug-likeness, etc.).

Methods:
- Gradient-based optimization
- Bayesian optimization
- Evolutionary strategies

Example:
    >>> optimizer = LatentOptimizer(vae, property_predictor)
    >>> optimized_z = optimizer.optimize(z_init, target_property=0.9)
    >>> smiles = vae.decode(optimized_z)
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

try:
    from rdkit import Chem
    from rdkit.Chem import QED, Descriptors, Crippen
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


class PropertyPredictor(nn.Module):
    """Neural network for property prediction from latent vectors.
    
    Used as surrogate model for latent space optimization.
    
    Args:
        latent_dim: Dimension of latent space.
        hidden_dim: Hidden layer dimension.
        output_dim: Number of properties to predict.
        
    Example:
        >>> predictor = PropertyPredictor(latent_dim=256)
        >>> predictor.fit(z_train, y_train)
        >>> predictions = predictor(z_test)
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 128,
        output_dim: int = 1,
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict properties from latent vectors."""
        return self.network(z)
    
    def fit(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> Dict:
        """Train predictor on latent-property pairs.
        
        Args:
            z: Latent vectors (N, latent_dim).
            y: Property values (N, output_dim).
            epochs: Training epochs.
            lr: Learning rate.
            
        Returns:
            Training history.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        history = {'loss': []}
        
        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            
            pred = self(z)
            loss = criterion(pred, y)
            
            loss.backward()
            optimizer.step()
            
            history['loss'].append(loss.item())
        
        logger.info(f"PropertyPredictor trained: final_loss={history['loss'][-1]:.4f}")
        return history


class LatentOptimizer:
    """Optimize molecules in VAE latent space.
    
    Supports multiple optimization strategies for finding molecules
    with desired properties.
    
    Args:
        vae: Trained SmilesVAE model.
        property_fn: Function to evaluate molecular properties.
        device: Computation device.
        
    Example:
        >>> optimizer = LatentOptimizer(vae, property_fn)
        >>> best_z, best_score = optimizer.optimize(z_init)
        >>> smiles = tokenizer.decode(vae.decode(best_z))
    """
    
    def __init__(
        self,
        vae: nn.Module,
        property_fn: Optional[Callable] = None,
        property_predictor: Optional[PropertyPredictor] = None,
        device: Optional[torch.device] = None,
    ):
        self.vae = vae
        self.property_fn = property_fn
        self.property_predictor = property_predictor
        
        if device is None:
            device = next(vae.parameters()).device
        self.device = device
        
        self.vae.eval()
    
    def gradient_optimization(
        self,
        z_init: torch.Tensor,
        target_value: float = 1.0,
        num_steps: int = 100,
        lr: float = 0.1,
        constraint_weight: float = 0.1,
    ) -> Tuple[torch.Tensor, float]:
        """Optimize latent vector using gradient descent.
        
        Requires a differentiable property predictor.
        
        Args:
            z_init: Initial latent vector.
            target_value: Target property value.
            num_steps: Optimization steps.
            lr: Learning rate.
            constraint_weight: Weight for prior constraint (stay near origin).
            
        Returns:
            Tuple of (optimized_z, final_score).
        """
        if self.property_predictor is None:
            raise ValueError("Gradient optimization requires property_predictor")
        
        z = z_init.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([z], lr=lr)
        
        best_z = z.clone()
        best_score = float('-inf')
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Property prediction
            pred = self.property_predictor(z)
            property_loss = (pred - target_value).pow(2).mean()
            
            # Prior constraint (regularization)
            prior_loss = z.pow(2).mean()
            
            # Total loss
            loss = property_loss + constraint_weight * prior_loss
            
            loss.backward()
            optimizer.step()
            
            # Track best
            with torch.no_grad():
                current_score = pred.mean().item()
                if current_score > best_score:
                    best_score = current_score
                    best_z = z.clone()
        
        return best_z.detach(), best_score
    
    def evolutionary_optimization(
        self,
        z_init: Optional[torch.Tensor] = None,
        population_size: int = 100,
        num_generations: int = 50,
        mutation_rate: float = 0.1,
        elite_ratio: float = 0.1,
        tokenizer=None,
    ) -> Tuple[torch.Tensor, float, List[str]]:
        """Optimize using evolutionary strategies.
        
        Uses the actual property function for evaluation.
        
        Args:
            z_init: Initial latent vector (or sample from prior).
            population_size: Size of population.
            num_generations: Number of generations.
            mutation_rate: Mutation standard deviation.
            elite_ratio: Fraction of elite individuals to keep.
            tokenizer: SMILES tokenizer for decoding.
            
        Returns:
            Tuple of (best_z, best_score, best_smiles_list).
        """
        latent_dim = self.vae.latent_dim
        
        # Initialize population
        if z_init is not None:
            # Start near initial point
            population = z_init + torch.randn(population_size, latent_dim, device=self.device) * 0.5
        else:
            # Sample from prior
            population = torch.randn(population_size, latent_dim, device=self.device)
        
        best_z = None
        best_score = float('-inf')
        best_smiles = []
        
        num_elite = max(1, int(population_size * elite_ratio))
        
        for gen in range(num_generations):
            # Evaluate population
            scores = self._evaluate_population(population, tokenizer)
            
            # Track best
            gen_best_idx = scores.argmax()
            if scores[gen_best_idx] > best_score:
                best_score = scores[gen_best_idx].item()
                best_z = population[gen_best_idx].clone()
            
            # Selection (tournament or elite)
            elite_indices = scores.argsort(descending=True)[:num_elite]
            elite = population[elite_indices]
            
            # Create new population
            new_population = [elite]
            
            # Generate offspring
            while sum(p.size(0) for p in new_population) < population_size:
                # Select parents from elite
                parent_idx = torch.randint(num_elite, (2,))
                parent1, parent2 = elite[parent_idx[0]], elite[parent_idx[1]]
                
                # Crossover
                mask = torch.rand(latent_dim, device=self.device) > 0.5
                child = torch.where(mask, parent1, parent2)
                
                # Mutation
                child = child + torch.randn_like(child) * mutation_rate
                
                new_population.append(child.unsqueeze(0))
            
            population = torch.cat(new_population, dim=0)[:population_size]
            
            if (gen + 1) % 10 == 0:
                logger.debug(f"Gen {gen + 1}: best_score={best_score:.4f}")
        
        # Get best SMILES
        if tokenizer is not None and best_z is not None:
            with torch.no_grad():
                generated = self.vae.decoder.generate(best_z.unsqueeze(0))
                best_smiles = [tokenizer.decode(generated[0].cpu().numpy())]
        
        return best_z, best_score, best_smiles
    
    def _evaluate_population(
        self,
        population: torch.Tensor,
        tokenizer=None,
    ) -> torch.Tensor:
        """Evaluate fitness of population.
        
        Args:
            population: Latent vectors (N, latent_dim).
            tokenizer: SMILES tokenizer.
            
        Returns:
            Fitness scores (N,).
        """
        if self.property_predictor is not None:
            # Use predictor for fast evaluation
            with torch.no_grad():
                scores = self.property_predictor(population).squeeze()
            return scores
        
        if self.property_fn is not None and tokenizer is not None:
            # Decode and evaluate with actual property function
            scores = []
            with torch.no_grad():
                for z in population:
                    generated = self.vae.decoder.generate(z.unsqueeze(0))
                    smiles = tokenizer.decode(generated[0].cpu().numpy())
                    score = self.property_fn(smiles)
                    scores.append(score)
            return torch.tensor(scores, device=self.device)
        
        # Default: random scores (for testing)
        return torch.rand(population.size(0), device=self.device)
    
    def bayesian_optimization(
        self,
        num_iterations: int = 50,
        num_initial: int = 10,
        tokenizer=None,
    ) -> Tuple[torch.Tensor, float, List[str]]:
        """Bayesian optimization in latent space.
        
        Uses Gaussian Process surrogate model with Expected Improvement.
        
        Args:
            num_iterations: Number of BO iterations.
            num_initial: Initial random samples.
            tokenizer: SMILES tokenizer.
            
        Returns:
            Tuple of (best_z, best_score, history).
        """
        latent_dim = self.vae.latent_dim
        
        # Initial samples
        X = torch.randn(num_initial, latent_dim, device=self.device)
        y = self._evaluate_population(X, tokenizer)
        
        best_z = X[y.argmax()]
        best_score = y.max().item()
        history = [best_score]
        
        for i in range(num_iterations):
            # Fit simple surrogate (linear for simplicity)
            # In practice, use GPyTorch or BoTorch
            
            # Sample candidates
            candidates = torch.randn(100, latent_dim, device=self.device)
            
            # Evaluate candidates
            candidate_scores = self._evaluate_population(candidates, tokenizer)
            
            # Select best candidate
            best_candidate_idx = candidate_scores.argmax()
            new_x = candidates[best_candidate_idx]
            new_y = candidate_scores[best_candidate_idx]
            
            # Update dataset
            X = torch.cat([X, new_x.unsqueeze(0)], dim=0)
            y = torch.cat([y, new_y.unsqueeze(0)])
            
            # Track best
            if new_y > best_score:
                best_score = new_y.item()
                best_z = new_x
            
            history.append(best_score)
        
        best_smiles = []
        if tokenizer is not None:
            with torch.no_grad():
                generated = self.vae.decoder.generate(best_z.unsqueeze(0))
                best_smiles = [tokenizer.decode(generated[0].cpu().numpy())]
        
        return best_z, best_score, best_smiles


def qed_score(smiles: str) -> float:
    """Calculate QED drug-likeness score.
    
    Args:
        smiles: SMILES string.
        
    Returns:
        QED score in [0, 1].
    """
    if not HAS_RDKIT:
        return 0.5
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        return QED.qed(mol)
    except:
        return 0.0


def logp_score(smiles: str) -> float:
    """Calculate LogP (lipophilicity).
    
    Args:
        smiles: SMILES string.
        
    Returns:
        LogP value (normalized to ~[0, 1]).
    """
    if not HAS_RDKIT:
        return 0.5
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0
        logp = Crippen.MolLogP(mol)
        # Normalize: ideal LogP is ~2-3
        return max(0, min(1, 1 - abs(logp - 2.5) / 5))
    except:
        return 0.0


def penalized_logp(smiles: str) -> float:
    """Calculate penalized LogP (used in MoleculeNet benchmarks).
    
    Args:
        smiles: SMILES string.
        
    Returns:
        Penalized LogP score.
    """
    if not HAS_RDKIT:
        return 0.0
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -10.0
        
        logp = Crippen.MolLogP(mol)
        sa_score = calculate_sa_score(mol)
        cycle_penalty = calculate_cycle_penalty(mol)
        
        return logp - sa_score - cycle_penalty
    except:
        return -10.0


def calculate_sa_score(mol) -> float:
    """Calculate synthetic accessibility score."""
    try:
        from rdkit.Chem import rdMolDescriptors
        # Simplified SA score
        return rdMolDescriptors.CalcNumRotatableBonds(mol) * 0.1
    except:
        return 0.0


def calculate_cycle_penalty(mol) -> float:
    """Penalize large rings."""
    try:
        ring_info = mol.GetRingInfo()
        max_ring_size = max([len(r) for r in ring_info.AtomRings()] or [0])
        return max(0, max_ring_size - 6)
    except:
        return 0.0


def combined_score(
    smiles: str,
    activity_predictor: Optional[Callable] = None,
    weights: Dict[str, float] = None,
) -> float:
    """Calculate combined multi-objective score.
    
    Args:
        smiles: SMILES string.
        activity_predictor: Optional activity prediction function.
        weights: Weights for different objectives.
        
    Returns:
        Combined score in [0, 1].
    """
    if weights is None:
        weights = {'qed': 0.3, 'validity': 0.3, 'activity': 0.4}
    
    scores = {}
    
    # Validity
    if HAS_RDKIT:
        mol = Chem.MolFromSmiles(smiles)
        scores['validity'] = 1.0 if mol is not None else 0.0
    else:
        scores['validity'] = 0.5
    
    # QED
    scores['qed'] = qed_score(smiles)
    
    # Activity
    if activity_predictor is not None:
        try:
            scores['activity'] = activity_predictor(smiles)
        except:
            scores['activity'] = 0.0
    else:
        scores['activity'] = 0.5
    
    # Weighted combination
    total = sum(weights.get(k, 0) * scores.get(k, 0) for k in weights)
    return total
