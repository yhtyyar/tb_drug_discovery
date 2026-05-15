"""Data preprocessing utilities for ML pipeline.

Key design decisions:
- RobustScaler instead of StandardScaler: median/IQR-based scaling is resistant
  to the outliers that frequently appear in computed molecular descriptors
  (e.g., BalabanJ diverges for certain graph topologies).
- joblib instead of pickle: safer serialization, sklearn version checking,
  optional compression.
- Explicit fit/transform separation with _is_fitted guard to prevent leakage.
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn import __version__ as sklearn_version
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


class DataPreprocessor:
    """Preprocess features for ML training.

    Pipeline (in order):
    1. Optional imputation of NaN values (mean / median / zero)
    2. RobustScaler — fit ONLY on training data, transform val/test

    Attributes:
        scaler: RobustScaler fitted on training data.
        imputer: SimpleImputer (optional).
        random_seed: For reproducible splits.

    Example:
        >>> prep = DataPreprocessor(random_seed=42)
        >>> X_train, X_val, X_test, y_train, y_val, y_test = prep.split_data(X, y)
        >>> X_train_sc = prep.fit_transform(X_train)
        >>> X_val_sc   = prep.transform(X_val)   # uses same scaler!
        >>> X_test_sc  = prep.transform(X_test)
    """

    def __init__(
        self,
        random_seed: int = 42,
        impute_strategy: Optional[Literal["mean", "median", "zero"]] = "median",
    ) -> None:
        """
        Args:
            random_seed: For train/test splitting.
            impute_strategy: How to fill NaN values before scaling.
                None = skip imputation (caller must guarantee no NaNs).
        """
        self.random_seed = random_seed
        self.impute_strategy = impute_strategy
        self.scaler = RobustScaler()
        self._imputer: Optional[SimpleImputer] = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        test_size: float = 0.15,
        val_size: float = 0.15,
        stratify: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train / validation / test sets.

        Splitting happens BEFORE any scaling so the scaler can never see
        validation or test statistics.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target values.
            test_size: Fraction for test set.
            val_size: Fraction for validation set (of the full dataset).
            stratify: Stratify splits by class label (classification).

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        X, y = self._to_numpy(X, y)
        self._validate_sizes(test_size, val_size)

        strat = y if stratify else None

        # First split off test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed, stratify=strat
        )

        # Then split off val from remaining
        val_relative = val_size / (1.0 - test_size)
        strat_inner = y_temp if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_relative,
            random_state=self.random_seed,
            stratify=strat_inner,
        )

        logger.info(
            f"Split → train={len(X_train)}, val={len(X_val)}, test={len(X_test)} "
            f"(stratified={stratify})"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def split_data_simple(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        test_size: float = 0.2,
        stratify: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simple train/test split (no validation set)."""
        X, y = self._to_numpy(X, y)
        strat = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed, stratify=strat
        )
        logger.info(f"Split → train={len(X_train)}, test={len(X_test)}")
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    # Scaling (fit only on train!)
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "DataPreprocessor":
        """Fit imputer + scaler on TRAINING data only.

        Never call this with validation or test data.
        """
        if self.impute_strategy is not None:
            strategy = "constant" if self.impute_strategy == "zero" else self.impute_strategy
            fill_value = 0.0 if self.impute_strategy == "zero" else None
            self._imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
            X = self._imputer.fit_transform(X)
        else:
            self._check_no_nan(X, "fit")

        self.scaler.fit(X)
        self._is_fitted = True

        n_nan = np.isnan(X).sum()
        logger.info(
            f"Preprocessor fitted: {X.shape[0]} samples, {X.shape[1]} features, "
            f"{n_nan} NaNs remaining"
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted imputer + scaler to new data."""
        if not self._is_fitted:
            raise ValueError("Not fitted — call fit() on training data first.")

        if self._imputer is not None:
            X = self._imputer.transform(X)
        else:
            self._check_no_nan(X, "transform")

        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X and transform X (for training set only)."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse scaling (scaler only — imputation is not reversible)."""
        if not self._is_fitted:
            raise ValueError("Not fitted.")
        return self.scaler.inverse_transform(X)

    # ------------------------------------------------------------------
    # Persistence — joblib, not pickle
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save preprocessor to joblib file.

        Safer than pickle: handles numpy arrays, records sklearn version.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "scaler": self.scaler,
            "imputer": self._imputer,
            "impute_strategy": self.impute_strategy,
            "random_seed": self.random_seed,
            "is_fitted": self._is_fitted,
            "sklearn_version": sklearn_version,
        }
        joblib.dump(state, path, compress=3)
        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: str) -> "DataPreprocessor":
        """Load preprocessor from joblib file."""
        if not Path(path).exists():
            raise FileNotFoundError(f"Preprocessor file not found: {path}")

        state = joblib.load(path)

        saved_ver = state.get("sklearn_version")
        if saved_ver and saved_ver != sklearn_version:
            logger.warning(
                f"sklearn version mismatch: saved={saved_ver}, current={sklearn_version}"
            )

        instance = cls(
            random_seed=state["random_seed"],
            impute_strategy=state.get("impute_strategy", "median"),
        )
        instance.scaler = state["scaler"]
        instance._imputer = state.get("imputer")
        instance._is_fitted = state["is_fitted"]

        logger.info(f"Preprocessor loaded from {path}")
        return instance

    # ------------------------------------------------------------------
    # High-level helper
    # ------------------------------------------------------------------

    def prepare_for_training(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "pIC50",
        test_size: float = 0.15,
        val_size: float = 0.15,
        stratify: bool = False,
        scale: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Full preparation: split → impute → scale, all in one call.

        Guarantees correct order: scaler is fit ONLY on X_train.
        """
        missing = [c for c in feature_cols + [target_col] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df_clean = df.dropna(subset=[target_col]).copy()
        n_dropped = len(df) - len(df_clean)
        if n_dropped:
            logger.warning(f"Dropped {n_dropped} rows with missing target '{target_col}'")

        X = df_clean[feature_cols].values.astype(float)
        y = df_clean[target_col].values

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            X, y, test_size=test_size, val_size=val_size, stratify=stratify
        )

        if scale:
            X_train = self.fit_transform(X_train)
            X_val = self.transform(X_val)
            X_test = self.transform(X_test)

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "feature_names": feature_cols,
            "n_features": len(feature_cols),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        return X.astype(float), y

    @staticmethod
    def _validate_sizes(test_size: float, val_size: float) -> None:
        if not (0 < test_size < 1) or not (0 < val_size < 1):
            raise ValueError("test_size and val_size must be in (0, 1)")
        if test_size + val_size >= 1.0:
            raise ValueError(
                f"test_size ({test_size}) + val_size ({val_size}) must be < 1.0"
            )

    @staticmethod
    def _check_no_nan(X: np.ndarray, context: str) -> None:
        n_nan = np.isnan(X).sum()
        if n_nan:
            raise ValueError(
                f"{n_nan} NaN values found during {context}. "
                "Pass impute_strategy='median' to DataPreprocessor or handle NaNs manually."
            )
