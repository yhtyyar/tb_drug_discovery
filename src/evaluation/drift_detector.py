"""Data drift detection for molecular descriptors.

Monitors changes in data distribution over time to detect when
model retraining might be needed.

Uses Kolmogorov-Smirnov test for univariate drift detection.
"""

from scipy.stats import ks_2samp
import numpy as np
from typing import Dict, List, Optional
from loguru import logger


class DescriptorDriftDetector:
    """KS-test based drift detection for molecular descriptors.

    Detects when the distribution of descriptors changes significantly,
    which may indicate model degradation.

    Args:
        alpha: Significance level for KS test (default 0.05).
        feature_names: Optional list of feature names for reporting.

    Example:
        >>> detector = DescriptorDriftDetector(alpha=0.05, feature_names=features)
        >>> detector.fit(X_reference)  # training data distribution
        >>> result = detector.detect(X_new)  # check new data
        >>> if result["drift_detected"]:
        ...     print(f"Drift in {result['n_features_drifted']} features")
    """

    def __init__(
        self,
        alpha: float = 0.05,
        feature_names: Optional[List[str]] = None,
    ):
        self.alpha = alpha
        self.feature_names = feature_names
        self.reference_X_: Optional[np.ndarray] = None

    def fit(self, X_reference: np.ndarray) -> "DescriptorDriftDetector":
        """Fit on reference data (e.g., training set).

        Args:
            X_reference: Reference feature matrix.

        Returns:
            Self for method chaining.
        """
        self.reference_X_ = X_reference
        logger.info(f"Drift detector fitted on {len(X_reference)} reference samples")
        return self

    def detect(self, X_new: np.ndarray) -> Dict[str, object]:
        """Detect drift in new data.

        Args:
            X_new: New feature matrix to check.

        Returns:
            Dictionary with drift detection results:
            - drift_detected: bool
            - n_features_drifted: int
            - fraction_drifted: float
            - drifted_features: list of (name, stat, p_value) tuples
            - p_values: dict of all p-values by feature name
        """
        if self.reference_X_ is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        n_features = X_new.shape[1]
        drifted = []
        p_values = {}

        for i in range(n_features):
            # KS test: reference vs new data
            stat, p = ks_2samp(self.reference_X_[:, i], X_new[:, i])

            # Get feature name
            if self.feature_names and i < len(self.feature_names):
                name = self.feature_names[i]
            else:
                name = f"feat_{i}"

            p_values[name] = float(p)

            # Significant difference detected
            if p < self.alpha:
                drifted.append((name, float(stat), float(p)))

        # Sort by p-value (most significant first)
        drifted.sort(key=lambda x: x[2])

        result = {
            "n_features_drifted": len(drifted),
            "fraction_drifted": len(drifted) / n_features if n_features > 0 else 0.0,
            "drifted_features": drifted,
            "p_values": p_values,
            "drift_detected": len(drifted) > 0,
            "alpha": self.alpha,
        }

        if result["drift_detected"]:
            logger.warning(
                f"Data drift detected in {len(drifted)}/{n_features} features: "
                f"{[d[0] for d in drifted[:5]]}{'...' if len(drifted) > 5 else ''}"
            )
        else:
            logger.info("No data drift detected")

        return result

    def get_drifted_features(self, X_new: np.ndarray) -> List[str]:
        """Get list of feature names with detected drift.

        Args:
            X_new: New feature matrix.

        Returns:
            List of feature names showing drift.
        """
        result = self.detect(X_new)
        return [name for name, _, _ in result["drifted_features"]]


class PopulationDriftDetector:
    """Multivariate drift detection using mean/covariance shift.

    Detects overall distribution shift using Mahalanobis distance.
    More sensitive to coordinated changes across features.
    """

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.mean_: Optional[np.ndarray] = None
        self.cov_inv_: Optional[np.ndarray] = None

    def fit(self, X_reference: np.ndarray) -> "PopulationDriftDetector":
        """Fit on reference data."""
        self.mean_ = X_reference.mean(axis=0)
        cov = np.cov(X_reference, rowvar=False)
        # Add regularization for stability
        cov += np.eye(cov.shape[0]) * 1e-6
        self.cov_inv_ = np.linalg.inv(cov)
        return self

    def mahalanobis_distance(self, X: np.ndarray) -> np.ndarray:
        """Calculate Mahalanobis distance for each sample.

        Args:
            X: Feature matrix.

        Returns:
            Distance array (higher = more different from reference).
        """
        if self.mean_ is None or self.cov_inv_ is None:
            raise ValueError("Detector not fitted. Call fit() first.")

        diff = X - self.mean_
        distances = np.sqrt(np.sum(diff @ self.cov_inv_ * diff, axis=1))
        return distances

    def detect(self, X_new: np.ndarray) -> Dict[str, object]:
        """Detect drift using mean Mahalanobis distance."""
        distances = self.mahalanobis_distance(X_new)
        mean_distance = float(distances.mean())
        max_distance = float(distances.max())

        # Drift if mean distance exceeds threshold
        drifted = mean_distance > self.threshold

        return {
            "drift_detected": drifted,
            "mean_distance": mean_distance,
            "max_distance": max_distance,
            "threshold": self.threshold,
            "n_outliers": int((distances > self.threshold).sum()),
        }
