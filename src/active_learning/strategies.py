"""Active learning acquisition strategies.

This module implements various strategies for selecting
the most informative samples in active learning.

Strategies:
- Uncertainty Sampling: Select most uncertain predictions
- Query-by-Committee: Use ensemble disagreement
- Expected Improvement: Bayesian optimization inspired
- Diversity Sampling: Maximize structural diversity
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from loguru import logger


class ActiveLearningStrategy(ABC):
    """Base class for active learning strategies."""
    
    @abstractmethod
    def select(
        self,
        X_pool: np.ndarray,
        n_select: int,
        **kwargs,
    ) -> np.ndarray:
        """Select samples from pool.
        
        Args:
            X_pool: Pool of unlabeled samples.
            n_select: Number of samples to select.
            
        Returns:
            Indices of selected samples.
        """
        pass
    
    @abstractmethod
    def score(self, X_pool: np.ndarray) -> np.ndarray:
        """Score all samples in pool.
        
        Returns:
            Acquisition scores (higher = more informative).
        """
        pass


class UncertaintySampling(ActiveLearningStrategy):
    """Uncertainty-based active learning.
    
    Selects samples where the model is most uncertain.
    
    Args:
        model: Trained model with predict method.
        uncertainty_method: 'entropy', 'margin', or 'least_confident'.
        
    Example:
        >>> strategy = UncertaintySampling(model, method='entropy')
        >>> indices = strategy.select(X_pool, n_select=10)
    """
    
    def __init__(
        self,
        model: Any,
        uncertainty_method: str = "entropy",
    ):
        self.model = model
        self.uncertainty_method = uncertainty_method
    
    def score(self, X_pool: np.ndarray) -> np.ndarray:
        """Compute uncertainty scores."""
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X_pool)
            
            if self.uncertainty_method == "entropy":
                # Shannon entropy
                scores = -np.sum(proba * np.log(proba + 1e-10), axis=1)
            elif self.uncertainty_method == "margin":
                # Difference between top two probabilities
                sorted_proba = np.sort(proba, axis=1)
                scores = 1 - (sorted_proba[:, -1] - sorted_proba[:, -2])
            elif self.uncertainty_method == "least_confident":
                # 1 - max probability
                scores = 1 - np.max(proba, axis=1)
            else:
                raise ValueError(f"Unknown method: {self.uncertainty_method}")
        else:
            # For regression: use prediction variance if available
            if hasattr(self.model, 'predict_with_uncertainty'):
                _, scores = self.model.predict_with_uncertainty(X_pool)
            else:
                # Random fallback
                scores = np.random.rand(len(X_pool))
        
        return scores
    
    def select(
        self,
        X_pool: np.ndarray,
        n_select: int,
        **kwargs,
    ) -> np.ndarray:
        """Select most uncertain samples."""
        scores = self.score(X_pool)
        indices = np.argsort(scores)[-n_select:][::-1]
        return indices


class QueryByCommittee(ActiveLearningStrategy):
    """Query-by-Committee active learning.
    
    Uses disagreement between committee members to identify
    informative samples.
    
    Args:
        models: List of trained models (committee).
        disagreement_method: 'vote_entropy' or 'kl_divergence'.
    """
    
    def __init__(
        self,
        models: List[Any],
        disagreement_method: str = "vote_entropy",
    ):
        self.models = models
        self.disagreement_method = disagreement_method
    
    def score(self, X_pool: np.ndarray) -> np.ndarray:
        """Compute committee disagreement scores."""
        n_samples = len(X_pool)
        n_models = len(self.models)
        
        if self.disagreement_method == "vote_entropy":
            # Collect votes
            predictions = np.array([m.predict(X_pool) for m in self.models])
            
            # Compute vote entropy
            scores = np.zeros(n_samples)
            for i in range(n_samples):
                votes = predictions[:, i]
                unique, counts = np.unique(votes, return_counts=True)
                probs = counts / n_models
                scores[i] = -np.sum(probs * np.log(probs + 1e-10))
        
        elif self.disagreement_method == "kl_divergence":
            # Average KL divergence from consensus
            probas = []
            for model in self.models:
                if hasattr(model, 'predict_proba'):
                    probas.append(model.predict_proba(X_pool))
            
            if probas:
                probas = np.array(probas)
                consensus = probas.mean(axis=0)
                
                kl_divs = []
                for p in probas:
                    kl = np.sum(p * np.log(p / (consensus + 1e-10) + 1e-10), axis=1)
                    kl_divs.append(kl)
                
                scores = np.mean(kl_divs, axis=0)
            else:
                scores = np.random.rand(n_samples)
        else:
            raise ValueError(f"Unknown method: {self.disagreement_method}")
        
        return scores
    
    def select(
        self,
        X_pool: np.ndarray,
        n_select: int,
        **kwargs,
    ) -> np.ndarray:
        """Select samples with highest disagreement."""
        scores = self.score(X_pool)
        indices = np.argsort(scores)[-n_select:][::-1]
        return indices


class ExpectedImprovement(ActiveLearningStrategy):
    """Expected Improvement acquisition function.
    
    Balances exploration and exploitation by considering
    both predicted value and uncertainty.
    
    Args:
        model: Model with predict and uncertainty estimation.
        best_value: Current best observed value.
        maximize: Whether to maximize the objective.
    """
    
    def __init__(
        self,
        model: Any,
        best_value: float = 0.0,
        maximize: bool = True,
    ):
        self.model = model
        self.best_value = best_value
        self.maximize = maximize
    
    def score(self, X_pool: np.ndarray) -> np.ndarray:
        """Compute Expected Improvement scores."""
        from scipy.stats import norm
        
        # Get predictions and uncertainty
        if hasattr(self.model, 'predict_with_uncertainty'):
            mu, sigma = self.model.predict_with_uncertainty(X_pool)
        else:
            mu = self.model.predict(X_pool)
            sigma = np.ones_like(mu) * 0.1
        
        sigma = np.maximum(sigma, 1e-6)
        
        if self.maximize:
            improvement = mu - self.best_value
        else:
            improvement = self.best_value - mu
        
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        return ei
    
    def select(
        self,
        X_pool: np.ndarray,
        n_select: int,
        **kwargs,
    ) -> np.ndarray:
        """Select samples with highest expected improvement."""
        scores = self.score(X_pool)
        indices = np.argsort(scores)[-n_select:][::-1]
        return indices
    
    def update_best(self, new_value: float) -> None:
        """Update best observed value."""
        if self.maximize:
            self.best_value = max(self.best_value, new_value)
        else:
            self.best_value = min(self.best_value, new_value)


class DiversitySampling(ActiveLearningStrategy):
    """Diversity-based sampling for structural coverage.
    
    Selects diverse samples to maximize coverage of chemical space.
    
    Args:
        distance_metric: 'euclidean' or 'tanimoto'.
        cluster_method: 'kmeans' or 'maxmin'.
    """
    
    def __init__(
        self,
        distance_metric: str = "euclidean",
        cluster_method: str = "maxmin",
    ):
        self.distance_metric = distance_metric
        self.cluster_method = cluster_method
    
    def score(self, X_pool: np.ndarray) -> np.ndarray:
        """Score by distance to nearest selected point."""
        # This is computed during selection
        return np.zeros(len(X_pool))
    
    def select(
        self,
        X_pool: np.ndarray,
        n_select: int,
        X_selected: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Select diverse samples using MaxMin algorithm."""
        n_pool = len(X_pool)
        
        if n_select >= n_pool:
            return np.arange(n_pool)
        
        if self.cluster_method == "maxmin":
            return self._maxmin_select(X_pool, n_select, X_selected)
        elif self.cluster_method == "kmeans":
            return self._kmeans_select(X_pool, n_select)
        else:
            raise ValueError(f"Unknown method: {self.cluster_method}")
    
    def _maxmin_select(
        self,
        X_pool: np.ndarray,
        n_select: int,
        X_selected: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """MaxMin diversity selection."""
        from scipy.spatial.distance import cdist
        
        n_pool = len(X_pool)
        selected_indices = []
        
        # Initialize with point furthest from selected or random
        if X_selected is not None and len(X_selected) > 0:
            distances = cdist(X_pool, X_selected, metric=self.distance_metric)
            min_distances = distances.min(axis=1)
            first_idx = np.argmax(min_distances)
        else:
            first_idx = np.random.randint(n_pool)
        
        selected_indices.append(first_idx)
        
        # Greedy MaxMin selection
        for _ in range(n_select - 1):
            selected_points = X_pool[selected_indices]
            distances = cdist(X_pool, selected_points, metric=self.distance_metric)
            min_distances = distances.min(axis=1)
            
            # Exclude already selected
            min_distances[selected_indices] = -np.inf
            
            next_idx = np.argmax(min_distances)
            selected_indices.append(next_idx)
        
        return np.array(selected_indices)
    
    def _kmeans_select(self, X_pool: np.ndarray, n_select: int) -> np.ndarray:
        """K-means based diversity selection."""
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_select, random_state=42, n_init=10)
        kmeans.fit(X_pool)
        
        # Select point closest to each cluster center
        selected_indices = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(X_pool - center, axis=1)
            closest = np.argmin(distances)
            
            while closest in selected_indices:
                distances[closest] = np.inf
                closest = np.argmin(distances)
            
            selected_indices.append(closest)
        
        return np.array(selected_indices)


class CombinedStrategy(ActiveLearningStrategy):
    """Combine multiple strategies with weighting.
    
    Args:
        strategies: List of (strategy, weight) tuples.
    """
    
    def __init__(self, strategies: List[Tuple[ActiveLearningStrategy, float]]):
        self.strategies = strategies
        total_weight = sum(w for _, w in strategies)
        self.weights = [w / total_weight for _, w in strategies]
    
    def score(self, X_pool: np.ndarray) -> np.ndarray:
        """Combine scores from all strategies."""
        combined_scores = np.zeros(len(X_pool))
        
        for (strategy, _), weight in zip(self.strategies, self.weights):
            scores = strategy.score(X_pool)
            # Normalize to [0, 1]
            if scores.max() > scores.min():
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            combined_scores += weight * scores
        
        return combined_scores
    
    def select(
        self,
        X_pool: np.ndarray,
        n_select: int,
        **kwargs,
    ) -> np.ndarray:
        """Select based on combined scores."""
        scores = self.score(X_pool)
        indices = np.argsort(scores)[-n_select:][::-1]
        return indices


class BatchModeAL(ActiveLearningStrategy):
    """Batch mode active learning with diversity.
    
    Combines informativeness with batch diversity to avoid
    selecting redundant samples.
    
    Args:
        base_strategy: Base acquisition strategy.
        diversity_weight: Weight for diversity term (0-1).
    """
    
    def __init__(
        self,
        base_strategy: ActiveLearningStrategy,
        diversity_weight: float = 0.5,
    ):
        self.base_strategy = base_strategy
        self.diversity_weight = diversity_weight
    
    def score(self, X_pool: np.ndarray) -> np.ndarray:
        """Score using base strategy."""
        return self.base_strategy.score(X_pool)
    
    def select(
        self,
        X_pool: np.ndarray,
        n_select: int,
        **kwargs,
    ) -> np.ndarray:
        """Select batch with informativeness + diversity."""
        from scipy.spatial.distance import cdist
        
        # Get informativeness scores
        info_scores = self.score(X_pool)
        
        # Normalize
        if info_scores.max() > info_scores.min():
            info_scores = (info_scores - info_scores.min()) / (info_scores.max() - info_scores.min())
        
        # Greedy selection with diversity
        selected_indices = []
        remaining = set(range(len(X_pool)))
        
        for _ in range(n_select):
            if not remaining:
                break
            
            remaining_list = list(remaining)
            
            if not selected_indices:
                # First: pure informativeness
                scores = info_scores[remaining_list]
            else:
                # Combine informativeness with diversity
                selected_points = X_pool[selected_indices]
                pool_points = X_pool[remaining_list]
                
                distances = cdist(pool_points, selected_points).min(axis=1)
                
                # Normalize diversity
                if distances.max() > distances.min():
                    div_scores = (distances - distances.min()) / (distances.max() - distances.min())
                else:
                    div_scores = np.ones(len(distances))
                
                # Combine
                scores = (1 - self.diversity_weight) * info_scores[remaining_list] + \
                         self.diversity_weight * div_scores
            
            best_idx = remaining_list[np.argmax(scores)]
            selected_indices.append(best_idx)
            remaining.remove(best_idx)
        
        return np.array(selected_indices)
