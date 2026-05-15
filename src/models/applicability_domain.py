"""Applicability Domain (AD) estimation for QSAR models.

This module provides methods to determine if a prediction is reliable
based on how similar the query molecule is to the training set.
Predictions outside the AD should be flagged as unreliable.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional


class BoundingBoxAD:
    """Simple AD based on descriptor value ranges from training data.

    A molecule is inside the AD if all its descriptors fall within
    [mean - k*std, mean + k*std] of the training set.

    Args:
        k: Multiplier for standard deviation (default 3.0 = ~99.7% coverage).

    Example:
        >>> ad = BoundingBoxAD(k=3.0)
        >>> ad.fit(X_train)
        >>> inside_ad = ad.predict(X_test)  # boolean array
        >>> coverage = ad.coverage(X_test)  # fraction inside AD
    """

    def __init__(self, k: float = 3.0):
        self.k = k
        self.lower_: Optional[np.ndarray] = None
        self.upper_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray) -> "BoundingBoxAD":
        """Fit AD on training data.

        Args:
            X_train: Training feature matrix.

        Returns:
            Self for method chaining.
        """
        self.mean_ = X_train.mean(axis=0)
        self.std_ = X_train.std(axis=0)
        # Avoid division by zero for constant features
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        self.lower_ = self.mean_ - self.k * self.std_
        self.upper_ = self.mean_ + self.k * self.std_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Check if samples are inside the applicability domain.

        Args:
            X: Feature matrix to check.

        Returns:
            Boolean array: True = inside AD, False = outside AD.
        """
        if self.lower_ is None or self.upper_ is None:
            raise ValueError("AD not fitted. Call fit() first.")
        inside = np.all((X >= self.lower_) & (X <= self.upper_), axis=1)
        return inside

    def coverage(self, X: np.ndarray) -> float:
        """Calculate fraction of samples inside AD.

        Args:
            X: Feature matrix.

        Returns:
            Fraction of samples inside AD (0.0 to 1.0).
        """
        return float(self.predict(X).mean())

    def distance_to_ad(self, X: np.ndarray) -> np.ndarray:
        """Calculate normalized distance to AD boundary.

        Negative values = inside AD, positive = outside.

        Args:
            X: Feature matrix.

        Returns:
            Distance metric per sample (lower = closer to AD).
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("AD not fitted. Call fit() first.")
        # Normalize to z-scores
        z_scores = np.abs((X - self.mean_) / self.std_)
        # Max z-score per sample
        return z_scores.max(axis=1) - self.k


class LeverageAD:
    """Williams plot: AD based on leverage (hat matrix diagonal).

    Standard QSAR method. A molecule is outside AD if h > h* = 3*(k+1)/n,
    where k = number of features, n = number of training samples.

    Reference: OECD QSAR validation principles.

    Example:
        >>> ad = LeverageAD()
        >>> ad.fit(X_train)
        >>> inside_ad = ad.predict(X_test)
        >>> leverage_values = ad.leverage(X_test)
    """

    def __init__(self, h_threshold_multiplier: float = 3.0):
        self.h_threshold_multiplier = h_threshold_multiplier
        self.X_train_: Optional[np.ndarray] = None
        self.XtX_inv_: Optional[np.ndarray] = None
        self.h_star_: float = 0.0
        self.n_: int = 0
        self.k_: int = 0

    def fit(self, X_train: np.ndarray) -> "LeverageAD":
        """Fit AD on training data.

        Args:
            X_train: Training feature matrix.

        Returns:
            Self for method chaining.
        """
        self.X_train_ = X_train
        self.n_, self.k_ = X_train.shape
        # Critical leverage: h* = 3*(k+1)/n
        self.h_star_ = self.h_threshold_multiplier * (self.k_ + 1) / self.n_

        # Precompute (X^T X)^-1 for leverage calculation
        XtX = X_train.T @ X_train
        # Add small regularization for numerical stability
        XtX += np.eye(XtX.shape[0]) * 1e-8
        self.XtX_inv_ = np.linalg.inv(XtX)

        return self

    def leverage(self, X: np.ndarray) -> np.ndarray:
        """Compute leverage values h_i for test molecules.

        Leverage measures how "influential" a sample would be
        if it were in the training set. High leverage = far from
        training set centroid.

        Args:
            X: Feature matrix.

        Returns:
            Leverage values for each sample.
        """
        if self.XtX_inv_ is None:
            raise ValueError("AD not fitted. Call fit() first.")
        # h_i = x_i^T (X^T X)^-1 x_i
        h = np.array([x @ self.XtX_inv_ @ x for x in X])
        return h

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Check if samples are inside the applicability domain.

        Args:
            X: Feature matrix.

        Returns:
            Boolean array: True = inside AD (h <= h*), False = outside.
        """
        return self.leverage(X) <= self.h_star_

    def coverage(self, X: np.ndarray) -> float:
        """Calculate fraction of samples inside AD.

        Args:
            X: Feature matrix.

        Returns:
            Fraction inside AD.
        """
        return float(self.predict(X).mean())


class EnsembleAD:
    """Combine multiple AD methods for robust domain estimation.

    A molecule is inside AD only if all methods agree.

    Args:
        methods: List of AD method instances.
        mode: 'strict' (all must agree inside) or 'lenient' (any says inside).

    Example:
        >>> ad = EnsembleAD([
        ...     BoundingBoxAD(k=3.0),
        ...     LeverageAD(),
        ... ])
        >>> ad.fit(X_train)
        >>> inside_ad = ad.predict(X_test)
    """

    def __init__(self, methods: list, mode: str = "strict"):
        self.methods = methods
        self.mode = mode
        self.fitted_ = False

    def fit(self, X_train: np.ndarray) -> "EnsembleAD":
        """Fit all AD methods on training data."""
        for method in self.methods:
            method.fit(X_train)
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Check if samples are inside AD based on all methods."""
        if not self.fitted_:
            raise ValueError("AD not fitted. Call fit() first.")

        predictions = np.stack([m.predict(X) for m in self.methods], axis=1)

        if self.mode == "strict":
            # All methods must agree the sample is inside
            return predictions.all(axis=1)
        else:  # lenient
            # At least one method says inside
            return predictions.any(axis=1)

    def coverage(self, X: np.ndarray) -> float:
        """Calculate fraction of samples inside AD."""
        return float(self.predict(X).mean())

    def method_agreement(self, X: np.ndarray) -> np.ndarray:
        """Get agreement fraction across all AD methods.

        Returns:
            Array of values 0-1 showing how many methods
            consider each sample inside AD.
        """
        predictions = np.stack([m.predict(X) for m in self.methods], axis=1)
        return predictions.mean(axis=1)
