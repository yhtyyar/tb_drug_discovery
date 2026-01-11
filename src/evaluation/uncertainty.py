"""Uncertainty Quantification for drug discovery predictions.

This module provides methods for estimating prediction uncertainty,
which is critical for prioritizing compounds for synthesis and testing.

Methods:
- MC Dropout: Monte Carlo Dropout for neural networks
- Deep Ensembles: Ensemble of independently trained models
- Conformal Prediction: Distribution-free uncertainty bounds
- Evidential Deep Learning: Direct uncertainty prediction

Example:
    >>> from src.evaluation.uncertainty import MCDropout, ConformalPredictor
    >>> mc = MCDropout(model, n_samples=100)
    >>> mean, std = mc.predict_with_uncertainty(X)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class UncertaintyEstimator(ABC):
    """Base class for uncertainty estimation methods."""
    
    @abstractmethod
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimates.
        
        Returns:
            Tuple of (predictions, uncertainties).
        """
        pass
    
    @abstractmethod
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Calibrate uncertainty estimates on calibration set."""
        pass


class MCDropout(UncertaintyEstimator):
    """Monte Carlo Dropout for uncertainty estimation.
    
    Uses dropout at inference time to generate multiple predictions,
    treating dropout as approximate Bayesian inference.
    
    Args:
        model: Neural network with dropout layers.
        n_samples: Number of MC samples.
        
    Example:
        >>> mc = MCDropout(model, n_samples=50)
        >>> mean, std = mc.predict_with_uncertainty(X)
    """
    
    def __init__(self, model: nn.Module, n_samples: int = 100):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for MC Dropout")
        
        self.model = model
        self.n_samples = n_samples
        self.device = next(model.parameters()).device
    
    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def predict_with_uncertainty(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions with uncertainty using MC Dropout.
        
        Args:
            X: Input features.
            
        Returns:
            Tuple of (mean predictions, standard deviations).
        """
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        self._enable_dropout()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(X)
                if isinstance(pred, torch.Tensor):
                    predictions.append(pred.cpu().numpy())
                else:
                    predictions.append(pred)
        
        predictions = np.array(predictions)
        
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        
        return mean, std
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Calibration not needed for MC Dropout."""
        pass
    
    def get_epistemic_uncertainty(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """Get epistemic (model) uncertainty only."""
        _, std = self.predict_with_uncertainty(X)
        return std


class DeepEnsemble(UncertaintyEstimator):
    """Deep Ensemble for uncertainty estimation.
    
    Trains multiple models independently and uses prediction
    disagreement as uncertainty measure.
    
    Args:
        model_fn: Function that creates a new model instance.
        n_models: Number of ensemble members.
        
    Example:
        >>> ensemble = DeepEnsemble(lambda: create_model(), n_models=5)
        >>> ensemble.fit(X_train, y_train)
        >>> mean, std = ensemble.predict_with_uncertainty(X_test)
    """
    
    def __init__(
        self,
        model_fn: callable,
        n_models: int = 5,
    ):
        self.model_fn = model_fn
        self.n_models = n_models
        self.models: List[Any] = []
        self.is_fitted = False
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **fit_kwargs,
    ) -> "DeepEnsemble":
        """Train ensemble of models.
        
        Args:
            X: Training features.
            y: Training targets.
            **fit_kwargs: Arguments passed to model.fit().
        """
        self.models = []
        
        for i in range(self.n_models):
            logger.debug(f"Training ensemble member {i + 1}/{self.n_models}")
            
            model = self.model_fn()
            
            # Bootstrap sampling
            n_samples = len(X)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            model.fit(X_boot, y_boot, **fit_kwargs)
            self.models.append(model)
        
        self.is_fitted = True
        logger.info(f"Deep Ensemble trained with {self.n_models} models")
        return self
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with ensemble uncertainty.
        
        Args:
            X: Input features.
            
        Returns:
            Tuple of (mean predictions, standard deviations).
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        
        return mean, std
    
    def predict_proba_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict probabilities with uncertainty (classification).
        
        Returns:
            Tuple of (mean probabilities, entropy).
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        probas = []
        
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                probas.append(proba)
            else:
                raise ValueError("Model does not support predict_proba")
        
        probas = np.array(probas)
        mean_proba = probas.mean(axis=0)
        
        # Entropy as uncertainty
        entropy = -np.sum(mean_proba * np.log(mean_proba + 1e-10), axis=1)
        
        return mean_proba, entropy
    
    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Calibration not needed for Deep Ensemble."""
        pass


class ConformalPredictor(UncertaintyEstimator):
    """Conformal Prediction for distribution-free uncertainty.
    
    Provides prediction intervals with guaranteed coverage
    without distributional assumptions.
    
    Args:
        model: Base predictor model.
        alpha: Significance level (1 - coverage probability).
        
    Example:
        >>> cp = ConformalPredictor(model, alpha=0.1)
        >>> cp.calibrate(X_cal, y_cal)
        >>> predictions, intervals = cp.predict_with_interval(X_test)
    """
    
    def __init__(
        self,
        model: Any,
        alpha: float = 0.1,
    ):
        self.model = model
        self.alpha = alpha
        self.calibration_scores: Optional[np.ndarray] = None
        self.quantile: Optional[float] = None
    
    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> None:
        """Calibrate conformal predictor on holdout set.
        
        Args:
            X_cal: Calibration features.
            y_cal: Calibration targets.
        """
        predictions = self.model.predict(X_cal)
        
        # Nonconformity scores (absolute residuals)
        self.calibration_scores = np.abs(y_cal - predictions)
        
        # Compute quantile
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = np.quantile(self.calibration_scores, min(q_level, 1.0))
        
        logger.info(f"Conformal predictor calibrated: quantile={self.quantile:.4f}")
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with conformal uncertainty.
        
        Returns:
            Tuple of (predictions, interval widths).
        """
        if self.quantile is None:
            raise ValueError("Predictor not calibrated")
        
        predictions = self.model.predict(X)
        uncertainties = np.full(len(predictions), self.quantile)
        
        return predictions, uncertainties
    
    def predict_with_interval(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with confidence intervals.
        
        Returns:
            Tuple of (predictions, lower bounds, upper bounds).
        """
        if self.quantile is None:
            raise ValueError("Predictor not calibrated")
        
        predictions = self.model.predict(X)
        lower = predictions - self.quantile
        upper = predictions + self.quantile
        
        return predictions, lower, upper
    
    def coverage(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate empirical coverage on test set."""
        _, lower, upper = self.predict_with_interval(X)
        covered = (y >= lower) & (y <= upper)
        return covered.mean()


class ClassificationConformalPredictor:
    """Conformal Prediction for classification.
    
    Provides prediction sets with guaranteed coverage.
    
    Args:
        model: Classifier with predict_proba method.
        alpha: Significance level.
    """
    
    def __init__(self, model: Any, alpha: float = 0.1):
        self.model = model
        self.alpha = alpha
        self.calibration_scores: Optional[np.ndarray] = None
        self.quantile: Optional[float] = None
    
    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> None:
        """Calibrate on holdout set."""
        probas = self.model.predict_proba(X_cal)
        
        # Nonconformity: 1 - probability of true class
        true_class_proba = probas[np.arange(len(y_cal)), y_cal.astype(int)]
        self.calibration_scores = 1 - true_class_proba
        
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = np.quantile(self.calibration_scores, min(q_level, 1.0))
        
        logger.info(f"Classification CP calibrated: quantile={self.quantile:.4f}")
    
    def predict_set(self, X: np.ndarray) -> List[List[int]]:
        """Predict set of possible classes.
        
        Returns:
            List of prediction sets for each sample.
        """
        if self.quantile is None:
            raise ValueError("Predictor not calibrated")
        
        probas = self.model.predict_proba(X)
        threshold = 1 - self.quantile
        
        prediction_sets = []
        for proba in probas:
            pred_set = np.where(proba >= threshold)[0].tolist()
            if not pred_set:  # Include at least the most likely class
                pred_set = [proba.argmax()]
            prediction_sets.append(pred_set)
        
        return prediction_sets
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with set size as uncertainty.
        
        Returns:
            Tuple of (predictions, set sizes).
        """
        prediction_sets = self.predict_set(X)
        predictions = np.array([ps[0] for ps in prediction_sets])
        uncertainties = np.array([len(ps) for ps in prediction_sets])
        
        return predictions, uncertainties


class TemperatureScaling:
    """Temperature Scaling for probability calibration.
    
    Learns a temperature parameter to calibrate predicted probabilities.
    
    Args:
        model: Classifier with predict_proba method.
    """
    
    def __init__(self, model: Any):
        self.model = model
        self.temperature: float = 1.0
    
    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> None:
        """Learn optimal temperature.
        
        Args:
            X_cal: Calibration features.
            y_cal: Calibration targets.
        """
        from scipy.optimize import minimize_scalar
        
        logits = self._get_logits(X_cal)
        
        def nll_loss(T):
            scaled = logits / T
            probs = self._softmax(scaled)
            log_probs = np.log(probs + 1e-10)
            nll = -log_probs[np.arange(len(y_cal)), y_cal.astype(int)].mean()
            return nll
        
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        
        logger.info(f"Temperature calibrated: T={self.temperature:.4f}")
    
    def _get_logits(self, X: np.ndarray) -> np.ndarray:
        """Get logits (pre-softmax) from model."""
        probas = self.model.predict_proba(X)
        # Inverse softmax approximation
        return np.log(probas + 1e-10)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_x = np.exp(x - x.max(axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated probabilities."""
        logits = self._get_logits(X)
        scaled = logits / self.temperature
        return self._softmax(scaled)


def create_uncertainty_estimator(
    method: str,
    model: Any,
    **kwargs,
) -> UncertaintyEstimator:
    """Factory function for uncertainty estimators.
    
    Args:
        method: 'mc_dropout', 'deep_ensemble', 'conformal'.
        model: Base model or model factory function.
        **kwargs: Method-specific arguments.
        
    Returns:
        UncertaintyEstimator instance.
    """
    if method == "mc_dropout":
        return MCDropout(model, **kwargs)
    elif method == "deep_ensemble":
        return DeepEnsemble(model, **kwargs)
    elif method == "conformal":
        return ConformalPredictor(model, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def evaluate_uncertainty_quality(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Evaluate quality of uncertainty estimates.
    
    Metrics:
    - Calibration error
    - Negative log-likelihood
    - Correlation between error and uncertainty
    
    Args:
        y_true: True values.
        y_pred: Predicted values.
        uncertainty: Uncertainty estimates.
        n_bins: Number of bins for calibration.
        
    Returns:
        Dictionary of uncertainty quality metrics.
    """
    errors = np.abs(y_true - y_pred)
    
    # Correlation between uncertainty and error
    correlation = np.corrcoef(uncertainty, errors)[0, 1]
    
    # Spearman correlation
    from scipy.stats import spearmanr
    spearman_corr, _ = spearmanr(uncertainty, errors)
    
    # ENCE (Expected Normalized Calibration Error)
    bins = np.percentile(uncertainty, np.linspace(0, 100, n_bins + 1))
    ence = 0.0
    
    for i in range(n_bins):
        mask = (uncertainty >= bins[i]) & (uncertainty < bins[i + 1])
        if mask.sum() > 0:
            bin_rmse = np.sqrt(np.mean(errors[mask] ** 2))
            bin_uncertainty = uncertainty[mask].mean()
            ence += np.abs(bin_rmse - bin_uncertainty) * mask.sum() / len(errors)
    
    return {
        "error_uncertainty_correlation": correlation,
        "spearman_correlation": spearman_corr,
        "ence": ence,
        "mean_uncertainty": uncertainty.mean(),
        "uncertainty_std": uncertainty.std(),
    }
