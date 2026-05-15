"""Conformal Prediction for uncertainty quantification in QSAR.

Provides distribution-free prediction intervals with guaranteed coverage.
Based on the split conformal prediction method.

References:
    - Vovk et al. (2005) "Algorithmic Learning in a Random World"
    - Shafer & Vovk (2008) "A Tutorial on Conformal Prediction"
    - Barber et al. (2021) "Predictive Inference with the Jackknife+"
"""

import numpy as np
from typing import Tuple, Optional


class ConformalPredictor:
    """Split conformal prediction for QSAR regression.

    Provides (1-alpha) confidence intervals without parametric assumptions
    about error distribution. Guaranteed coverage for exchangeable data.

    Args:
        alpha: Significance level (0.1 = 90% coverage).

    Example:
        >>> cp = ConformalPredictor(alpha=0.1)  # 90% intervals
        >>> cp.calibrate(y_cal, y_pred_cal)  # on held-out calibration set
        >>> lower, upper = cp.predict_interval(y_pred_test)
        >>> coverage = cp.coverage_empirical(y_test, y_pred_test)
    """

    def __init__(self, alpha: float = 0.1):
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.q_: Optional[float] = None
        self.n_cal_: int = 0

    def calibrate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> "ConformalPredictor":
        """Calibrate on held-out calibration set.

        Args:
            y_true: True values from calibration set.
            y_pred: Predicted values on calibration set.

        Returns:
            Self for method chaining.
        """
        # Nonconformity scores = absolute residuals
        scores = np.abs(y_true - y_pred)
        self.n_cal_ = len(scores)

        # Quantile level for (1-alpha) coverage
        # Using (n+1) * (1-alpha) / n for finite-sample correction
        q_level = np.ceil((1 - self.alpha) * (self.n_cal_ + 1)) / self.n_cal_
        q_level = min(q_level, 1.0)

        self.q_ = np.quantile(scores, q_level)
        return self

    def predict_interval(
        self,
        y_pred: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate prediction intervals for new predictions.

        Args:
            y_pred: Point predictions from the model.

        Returns:
            Tuple of (lower_bound, upper_bound) arrays.
        """
        if self.q_ is None:
            raise ValueError("Predictor not calibrated. Call calibrate() first.")

        lower = y_pred - self.q_
        upper = y_pred + self.q_
        return lower, upper

    def predict_with_uncertainty(
        self,
        y_pred: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get predictions with interval and interval width.

        Args:
            y_pred: Point predictions.

        Returns:
            Tuple of (prediction, lower, upper, interval_width).
        """
        lower, upper = self.predict_interval(y_pred)
        width = upper - lower
        return lower, upper, width

    def coverage_empirical(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Check empirical coverage on test set.

        Should be >= (1-alpha) for correctly calibrated predictor.

        Args:
            y_true: True values.
            y_pred: Predicted values.

        Returns:
            Empirical coverage (fraction of y_true in intervals).
        """
        lower, upper = self.predict_interval(y_pred)
        return float(((y_true >= lower) & (y_true <= upper)).mean())

    def interval_width(self) -> float:
        """Get average interval width (2 * q)."""
        if self.q_ is None:
            raise ValueError("Predictor not calibrated.")
        return 2 * self.q_


class ClassificationConformalPredictor:
    """Conformal prediction for classification (Venn-ABERS predictor).

    Provides prediction sets with guaranteed coverage.

    Args:
        alpha: Significance level (0.1 = 90% coverage).
        n_classes: Number of classes.

    Example:
        >>> cp = ClassificationConformalPredictor(alpha=0.1, n_classes=2)
        >>> cp.calibrate(y_cal, y_proba_cal)
        >>> prediction_sets = cp.predict_sets(y_proba_test)
    """

    def __init__(self, alpha: float = 0.1, n_classes: int = 2):
        self.alpha = alpha
        self.n_classes = n_classes
        self.scores_: Optional[np.ndarray] = None
        self.q_: Optional[float] = None

    def calibrate(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> "ClassificationConformalPredictor":
        """Calibrate on held-out set.

        Args:
            y_true: True labels (integer indices).
            y_proba: Predicted class probabilities.
        """
        # Nonconformity score: 1 - p(y_true)
        true_class_proba = y_proba[np.arange(len(y_true)), y_true]
        self.scores_ = 1 - true_class_proba

        n = len(self.scores_)
        q_level = np.ceil((1 - self.alpha) * (n + 1)) / n
        q_level = min(q_level, 1.0)
        self.q_ = np.quantile(self.scores_, q_level)
        return self

    def predict_sets(
        self,
        y_proba: np.ndarray,
    ) -> np.ndarray:
        """Generate prediction sets.

        Args:
            y_proba: Predicted class probabilities.

        Returns:
            Boolean array of shape (n_samples, n_classes).
            True = class in prediction set.
        """
        if self.q_ is None:
            raise ValueError("Predictor not calibrated.")

        # Include all classes with score <= q
        # score(c) = 1 - p(c)
        scores = 1 - y_proba
        return scores <= self.q_

    def coverage_empirical(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> float:
        """Check empirical coverage."""
        prediction_sets = self.predict_sets(y_proba)
        covered = prediction_sets[np.arange(len(y_true)), y_true]
        return float(covered.mean())

    def average_set_size(self, y_proba: np.ndarray) -> float:
        """Average size of prediction sets (lower = more informative)."""
        prediction_sets = self.predict_sets(y_proba)
        return float(prediction_sets.sum(axis=1).mean())
