"""Active Learning loop for drug discovery.

This module provides a complete active learning workflow
for iterative compound selection and model improvement.

Example:
    >>> learner = ActiveLearner(model, strategy, oracle)
    >>> learner.run(X_pool, n_iterations=10, batch_size=20)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .strategies import ActiveLearningStrategy, UncertaintySampling


class ActiveLearner:
    """Complete active learning loop for drug discovery.
    
    Manages the iterative process of:
    1. Selecting informative compounds
    2. Querying oracle (experiments/simulations)
    3. Updating the model
    4. Evaluating progress
    
    Args:
        model: ML model with fit/predict methods.
        strategy: Active learning acquisition strategy.
        oracle: Function to get labels (can be experimental).
        
    Example:
        >>> def oracle(X):
        ...     # Simulate experiments or call API
        ...     return experimental_results
        >>> 
        >>> learner = ActiveLearner(model, strategy, oracle)
        >>> history = learner.run(X_pool, n_iterations=10)
    """
    
    def __init__(
        self,
        model: Any,
        strategy: Optional[ActiveLearningStrategy] = None,
        oracle: Optional[Callable] = None,
    ):
        self.model = model
        self.strategy = strategy or UncertaintySampling(model)
        self.oracle = oracle
        
        # Training data
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        
        # History
        self.history: Dict[str, List] = {
            "n_samples": [],
            "selected_indices": [],
            "metrics": [],
        }
    
    def initialize(
        self,
        X_initial: np.ndarray,
        y_initial: np.ndarray,
    ) -> None:
        """Initialize with labeled seed data.
        
        Args:
            X_initial: Initial labeled features.
            y_initial: Initial labels.
        """
        self.X_train = X_initial.copy()
        self.y_train = y_initial.copy()
        
        # Fit initial model
        self.model.fit(self.X_train, self.y_train)
        
        logger.info(f"Initialized with {len(X_initial)} labeled samples")
    
    def run(
        self,
        X_pool: np.ndarray,
        n_iterations: int = 10,
        batch_size: int = 10,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        callback: Optional[Callable] = None,
    ) -> Dict[str, List]:
        """Run active learning loop.
        
        Args:
            X_pool: Pool of unlabeled samples.
            n_iterations: Number of AL iterations.
            batch_size: Samples to select per iteration.
            X_test: Optional test set for evaluation.
            y_test: Optional test labels.
            callback: Optional callback(iteration, metrics).
            
        Returns:
            Training history dictionary.
        """
        if self.X_train is None:
            raise ValueError("Must call initialize() first")
        
        # Track remaining pool
        pool_mask = np.ones(len(X_pool), dtype=bool)
        
        for iteration in range(n_iterations):
            logger.info(f"AL Iteration {iteration + 1}/{n_iterations}")
            
            # Get remaining pool
            remaining_indices = np.where(pool_mask)[0]
            X_remaining = X_pool[remaining_indices]
            
            if len(X_remaining) < batch_size:
                logger.warning("Pool exhausted")
                break
            
            # Select samples
            selected_local = self.strategy.select(
                X_remaining, 
                n_select=batch_size,
            )
            selected_global = remaining_indices[selected_local]
            
            # Query oracle
            X_new = X_pool[selected_global]
            if self.oracle is not None:
                y_new = self.oracle(X_new)
            else:
                logger.warning("No oracle provided, using dummy labels")
                y_new = np.zeros(len(X_new))
            
            # Update training data
            self.X_train = np.vstack([self.X_train, X_new])
            self.y_train = np.concatenate([self.y_train, y_new])
            
            # Update pool mask
            pool_mask[selected_global] = False
            
            # Retrain model
            self.model.fit(self.X_train, self.y_train)
            
            # Evaluate
            metrics = {}
            if X_test is not None and y_test is not None:
                metrics = self._evaluate(X_test, y_test)
            
            # Update history
            self.history["n_samples"].append(len(self.X_train))
            self.history["selected_indices"].append(selected_global.tolist())
            self.history["metrics"].append(metrics)
            
            # Callback
            if callback is not None:
                callback(iteration, metrics)
            
            logger.info(f"  Samples: {len(self.X_train)}, Metrics: {metrics}")
        
        return self.history
    
    def _evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test set."""
        from sklearn.metrics import roc_auc_score, r2_score, accuracy_score
        
        y_pred = self.model.predict(X_test)
        
        metrics = {}
        
        # Classification metrics
        if hasattr(self.model, 'predict_proba'):
            try:
                y_proba = self.model.predict_proba(X_test)
                if y_proba.ndim == 2:
                    y_proba = y_proba[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
            except Exception:
                pass
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
        else:
            # Regression metrics
            metrics["r2"] = r2_score(y_test, y_pred)
        
        return metrics
    
    def select_next_batch(
        self,
        X_pool: np.ndarray,
        batch_size: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select next batch without running full loop.
        
        Args:
            X_pool: Pool of candidates.
            batch_size: Number to select.
            
        Returns:
            Tuple of (selected_indices, selected_samples).
        """
        indices = self.strategy.select(X_pool, n_select=batch_size)
        return indices, X_pool[indices]
    
    def add_labeled_samples(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        retrain: bool = True,
    ) -> None:
        """Add new labeled samples to training set.
        
        Args:
            X_new: New features.
            y_new: New labels.
            retrain: Whether to retrain model.
        """
        if self.X_train is None:
            self.X_train = X_new
            self.y_train = y_new
        else:
            self.X_train = np.vstack([self.X_train, X_new])
            self.y_train = np.concatenate([self.y_train, y_new])
        
        if retrain:
            self.model.fit(self.X_train, self.y_train)
        
        logger.info(f"Added {len(X_new)} samples. Total: {len(self.X_train)}")
    
    def get_learning_curve(self) -> Tuple[List[int], List[float]]:
        """Get learning curve from history.
        
        Returns:
            Tuple of (n_samples_list, metric_values).
        """
        n_samples = self.history["n_samples"]
        
        # Get primary metric
        metrics_list = self.history["metrics"]
        if not metrics_list or not metrics_list[0]:
            return n_samples, []
        
        metric_key = list(metrics_list[0].keys())[0]
        values = [m.get(metric_key, 0) for m in metrics_list]
        
        return n_samples, values
    
    def save_state(self, path: str) -> None:
        """Save learner state."""
        import pickle
        
        state = {
            "X_train": self.X_train,
            "y_train": self.y_train,
            "history": self.history,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"State saved to {path}")
    
    def load_state(self, path: str) -> None:
        """Load learner state."""
        import pickle
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.X_train = state["X_train"]
        self.y_train = state["y_train"]
        self.history = state["history"]
        
        # Retrain model
        if self.X_train is not None:
            self.model.fit(self.X_train, self.y_train)
        
        logger.info(f"State loaded from {path}")


class SimulatedOracle:
    """Simulated oracle for testing active learning.
    
    Uses a pre-trained model or known function to simulate
    experimental results.
    
    Args:
        true_function: Function that returns true labels.
        noise_std: Standard deviation of noise to add.
        delay: Simulated delay per query (seconds).
    """
    
    def __init__(
        self,
        true_function: Callable,
        noise_std: float = 0.0,
        delay: float = 0.0,
    ):
        self.true_function = true_function
        self.noise_std = noise_std
        self.delay = delay
        self.query_count = 0
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Query oracle for labels."""
        import time
        
        if self.delay > 0:
            time.sleep(self.delay * len(X))
        
        y = self.true_function(X)
        
        if self.noise_std > 0:
            y = y + np.random.normal(0, self.noise_std, size=y.shape)
        
        self.query_count += len(X)
        
        return y


def run_active_learning_experiment(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_fn: Callable,
    strategy_name: str = "uncertainty",
    n_initial: int = 20,
    n_iterations: int = 10,
    batch_size: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run complete active learning experiment.
    
    Args:
        X_pool: Pool features.
        y_pool: Pool labels (for simulated oracle).
        X_test: Test features.
        y_test: Test labels.
        model_fn: Function to create model.
        strategy_name: 'uncertainty', 'qbc', 'diversity', or 'random'.
        n_initial: Initial labeled samples.
        n_iterations: AL iterations.
        batch_size: Samples per iteration.
        seed: Random seed.
        
    Returns:
        Experiment results dictionary.
    """
    np.random.seed(seed)
    
    # Create model
    model = model_fn()
    
    # Create strategy
    if strategy_name == "uncertainty":
        strategy = UncertaintySampling(model)
    elif strategy_name == "random":
        strategy = None  # Will use random
    else:
        strategy = UncertaintySampling(model)
    
    # Create oracle
    oracle = SimulatedOracle(lambda X: y_pool[np.argmin(
        np.linalg.norm(X_pool[:, None] - X, axis=2), axis=0
    )])
    
    # Initialize with random samples
    initial_indices = np.random.choice(len(X_pool), n_initial, replace=False)
    X_initial = X_pool[initial_indices]
    y_initial = y_pool[initial_indices]
    
    # Remove initial from pool
    pool_mask = np.ones(len(X_pool), dtype=bool)
    pool_mask[initial_indices] = False
    X_pool_remaining = X_pool[pool_mask]
    
    # Update oracle for remaining pool
    y_pool_remaining = y_pool[pool_mask]
    oracle = SimulatedOracle(lambda X: y_pool_remaining[np.argmin(
        np.linalg.norm(X_pool_remaining[:, None] - X, axis=2), axis=0
    )])
    
    # Create learner
    learner = ActiveLearner(model, strategy, oracle)
    learner.initialize(X_initial, y_initial)
    
    # Run
    history = learner.run(
        X_pool_remaining,
        n_iterations=n_iterations,
        batch_size=batch_size,
        X_test=X_test,
        y_test=y_test,
    )
    
    return {
        "strategy": strategy_name,
        "history": history,
        "final_n_samples": len(learner.X_train),
        "final_metrics": history["metrics"][-1] if history["metrics"] else {},
    }
