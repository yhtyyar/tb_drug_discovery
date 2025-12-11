"""Data preprocessing utilities for ML pipeline.

This module provides data splitting, scaling, and preparation
utilities for training ML models.
"""

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Preprocess data for ML training.
    
    Handles:
    - Train/validation/test splitting
    - Feature scaling
    - Missing value handling
    - Data export for training
    
    Attributes:
        scaler: StandardScaler for feature normalization.
        random_seed: Random seed for reproducibility.
        
    Example:
        >>> prep = DataPreprocessor(random_seed=42)
        >>> X_train, X_test, y_train, y_test = prep.split_data(X, y)
        >>> X_train_scaled = prep.fit_transform(X_train)
        >>> X_test_scaled = prep.transform(X_test)
    """
    
    def __init__(self, random_seed: int = 42) -> None:
        """Initialize preprocessor.
        
        Args:
            random_seed: Random seed for reproducibility.
        """
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def split_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        test_size: float = 0.15,
        val_size: float = 0.15,
        stratify: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/validation/test sets.
        
        Args:
            X: Feature matrix.
            y: Target values.
            test_size: Proportion for test set.
            val_size: Proportion for validation set.
            stratify: Whether to stratify by target (for classification).
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Calculate actual validation size from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        
        stratify_target = y if stratify else None
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=stratify_target,
        )
        
        # Second split: train vs val
        stratify_target = y_temp if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_seed,
            stratify=stratify_target,
        )
        
        logger.info(
            f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def split_data_simple(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        test_size: float = 0.2,
        stratify: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simple train/test split.
        
        Args:
            X: Feature matrix.
            y: Target values.
            test_size: Proportion for test set.
            stratify: Whether to stratify by target.
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        stratify_target = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=stratify_target,
        )
        
        logger.info(f"Data split: train={len(X_train)}, test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def fit(self, X: np.ndarray) -> "DataPreprocessor":
        """Fit scaler on training data.
        
        Args:
            X: Training feature matrix.
            
        Returns:
            Self for method chaining.
        """
        self.scaler.fit(X)
        self._is_fitted = True
        logger.info(f"Scaler fitted on {X.shape[0]} samples, {X.shape[1]} features")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler.
        
        Args:
            X: Feature matrix to transform.
            
        Returns:
            Scaled feature matrix.
        """
        if not self._is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler and transform in one step.
        
        Args:
            X: Training feature matrix.
            
        Returns:
            Scaled feature matrix.
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform scaled features.
        
        Args:
            X: Scaled feature matrix.
            
        Returns:
            Original scale feature matrix.
        """
        if not self._is_fitted:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        return self.scaler.inverse_transform(X)
    
    def handle_missing_values(
        self,
        X: np.ndarray,
        strategy: str = "mean",
    ) -> np.ndarray:
        """Handle missing values in feature matrix.
        
        Args:
            X: Feature matrix with potential NaN values.
            strategy: Imputation strategy ('mean', 'median', 'zero').
            
        Returns:
            Feature matrix with imputed values.
        """
        X = X.copy()
        
        for col in range(X.shape[1]):
            col_data = X[:, col]
            mask = np.isnan(col_data)
            
            if not mask.any():
                continue
            
            if strategy == "mean":
                fill_value = np.nanmean(col_data)
            elif strategy == "median":
                fill_value = np.nanmedian(col_data)
            elif strategy == "zero":
                fill_value = 0.0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            X[mask, col] = fill_value
        
        n_imputed = np.isnan(X).sum()
        if n_imputed > 0:
            logger.warning(f"Still {n_imputed} NaN values after imputation")
        
        return X
    
    def save(self, path: str) -> None:
        """Save preprocessor state.
        
        Args:
            path: Path to save pickle file.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "scaler": self.scaler,
            "random_seed": self.random_seed,
            "is_fitted": self._is_fitted,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "DataPreprocessor":
        """Load preprocessor from file.
        
        Args:
            path: Path to pickle file.
            
        Returns:
            Loaded DataPreprocessor instance.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        instance = cls(random_seed=state["random_seed"])
        instance.scaler = state["scaler"]
        instance._is_fitted = state["is_fitted"]
        
        logger.info(f"Preprocessor loaded from {path}")
        return instance
    
    def prepare_for_training(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str = "pIC50",
        test_size: float = 0.15,
        val_size: float = 0.15,
        scale: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Prepare data for ML training in one step.
        
        Args:
            df: DataFrame with features and target.
            feature_cols: List of feature column names.
            target_col: Name of target column.
            test_size: Test set proportion.
            val_size: Validation set proportion.
            scale: Whether to scale features.
            
        Returns:
            Dictionary with train/val/test splits.
        """
        # Drop rows with missing values
        df_clean = df.dropna(subset=feature_cols + [target_col])
        
        X = df_clean[feature_cols].values
        y = df_clean[target_col].values
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
            X, y, test_size=test_size, val_size=val_size
        )
        
        # Scale features
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
        }
