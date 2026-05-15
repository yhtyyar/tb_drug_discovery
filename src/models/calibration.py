"""Probability calibration for QSAR classifiers.

Random Forest predict_proba() outputs are often poorly calibrated:
they cluster near 0.5 for balanced datasets and near the class frequency
for imbalanced ones. Calibration maps raw scores to true probabilities,
which matters for:
- Conformal prediction (requires well-calibrated base model)
- Decision thresholds (prioritizing compounds for synthesis)
- Combining predictions from multiple models

Supported methods:
- Platt scaling (sigmoid): fast, works well for RF
- Isotonic regression: more flexible, needs ≥ 1000 calibration samples
- Temperature scaling: single parameter, standard in deep learning

References:
    Platt (1999) — Probabilistic Outputs for SVM
    Niculescu-Mizil & Caruana (2005) — Predicting Good Probabilities
    Guo et al. (2017) — On Calibration of Modern Neural Networks
"""

from typing import Literal, Optional

import numpy as np
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class QSARCalibrator:
    """Wraps a fitted QSARModel and calibrates its probability outputs.

    Usage:
        >>> model = QSARModel(task='classification')
        >>> model.fit(X_train, y_train)
        >>> cal = QSARCalibrator(model, method='isotonic')
        >>> cal.fit(X_cal, y_cal)          # calibration set (held-out)
        >>> proba = cal.predict_proba(X_test)[:, 1]
        >>> # proba is now much better calibrated

    Note: ``X_cal`` / ``y_cal`` must be a *held-out* calibration set,
    not the training set — using training data leads to over-confident
    calibration.
    """

    def __init__(
        self,
        model,
        method: Literal["sigmoid", "isotonic", "temperature"] = "sigmoid",
    ) -> None:
        """
        Args:
            model: Fitted QSARModel with task='classification'.
            method: Calibration method.
                - 'sigmoid': Platt scaling (logistic regression on scores).
                - 'isotonic': Isotonic regression (monotone, non-parametric).
                - 'temperature': Single temperature parameter T, p → sigmoid(logit/T).
        """
        if model.task != "classification":
            raise ValueError("Calibration only applies to classification models")
        if not model.is_fitted:
            raise ValueError("Model must be fitted before calibration")

        self.model = model
        self.method = method
        self._calibrator = None
        self._temperature: float = 1.0
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QSARCalibrator":
        """Fit calibration on a held-out calibration set.

        Args:
            X: Feature matrix for calibration.
            y: True binary labels.

        Returns:
            Self for chaining.
        """
        raw_proba = self.model.predict_proba(X)[:, 1]

        if self.method == "sigmoid":
            self._calibrator = LogisticRegression(C=1e10)
            self._calibrator.fit(raw_proba.reshape(-1, 1), y)

        elif self.method == "isotonic":
            if len(y) < 1000:
                logger.warning(
                    f"Isotonic calibration with only {len(y)} samples may overfit. "
                    "Use 'sigmoid' for small calibration sets."
                )
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(raw_proba, y)

        elif self.method == "temperature":
            # Optimize temperature T to minimize NLL on calibration set
            from scipy.optimize import minimize_scalar

            def nll(T: float) -> float:
                T = max(T, 1e-6)
                logits = np.log(raw_proba + 1e-9) - np.log(1 - raw_proba + 1e-9)
                p = 1 / (1 + np.exp(-logits / T))
                p = np.clip(p, 1e-9, 1 - 1e-9)
                return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

            result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
            self._temperature = result.x
            logger.info(f"Optimal temperature: {self._temperature:.4f}")

        self.is_fitted = True
        ece = self._expected_calibration_error(X, y)
        logger.info(f"Calibration ({self.method}) fitted. ECE = {ece:.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Array of shape (n_samples, 2) with [P(inactive), P(active)].
        """
        if not self.is_fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")

        raw_proba = self.model.predict_proba(X)[:, 1]

        if self.method == "sigmoid":
            cal_proba = self._calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]

        elif self.method == "isotonic":
            cal_proba = self._calibrator.predict(raw_proba)

        elif self.method == "temperature":
            logits = np.log(raw_proba + 1e-9) - np.log(1 - raw_proba + 1e-9)
            cal_proba = 1 / (1 + np.exp(-logits / self._temperature))

        cal_proba = np.clip(cal_proba, 0.0, 1.0)
        return np.column_stack([1 - cal_proba, cal_proba])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels using calibrated probabilities."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def _expected_calibration_error(
        self, X: np.ndarray, y: np.ndarray, n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error (ECE) — lower is better."""
        cal_proba = self.predict_proba(X)[:, 1]
        fraction_of_positives, mean_predicted = calibration_curve(
            y, cal_proba, n_bins=n_bins, strategy="uniform"
        )
        bin_sizes = np.histogram(cal_proba, bins=n_bins, range=(0, 1))[0]
        # Weighted absolute difference
        ece = float(np.sum(np.abs(fraction_of_positives - mean_predicted) * bin_sizes)
                    / len(y))
        return ece

    def calibration_report(
        self, X: np.ndarray, y: np.ndarray, n_bins: int = 10
    ) -> dict:
        """Return calibration diagnostics.

        Returns:
            Dict with ECE, reliability diagram data, Brier score.
        """
        from sklearn.metrics import brier_score_loss

        cal_proba = self.predict_proba(X)[:, 1]
        raw_proba = self.model.predict_proba(X)[:, 1]

        frac_pos, mean_pred = calibration_curve(y, cal_proba, n_bins=n_bins)

        return {
            "ece": self._expected_calibration_error(X, y, n_bins),
            "brier_score": float(brier_score_loss(y, cal_proba)),
            "brier_score_uncalibrated": float(brier_score_loss(y, raw_proba)),
            "reliability_fraction_positives": frac_pos.tolist(),
            "reliability_mean_predicted": mean_pred.tolist(),
            "method": self.method,
            "n_calibration_samples": len(y),
        }
