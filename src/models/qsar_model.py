"""QSAR Model implementation using Random Forest.

This module provides a scikit-learn based QSAR model for predicting
molecular activity (pIC50) against TB targets.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict


class QSARModel:
    """QSAR model for TB drug activity prediction.
    
    Implements both regression (pIC50 prediction) and classification
    (active/inactive) using Random Forest algorithm.
    
    Attributes:
        model: Trained Random Forest model.
        task: Either 'regression' or 'classification'.
        is_fitted: Whether model has been trained.
        
    Example:
        >>> model = QSARModel(task='regression')
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> metrics = model.evaluate(X_test, y_test)
        >>> print(f"R² = {metrics['r2']:.3f}")
    """
    
    def __init__(
        self,
        task: str = "regression",
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_seed: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        """Initialize QSAR model.
        
        Args:
            task: 'regression' for pIC50 prediction, 'classification' for active/inactive.
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of trees (None = unlimited).
            min_samples_split: Minimum samples required to split node.
            min_samples_leaf: Minimum samples required in leaf node.
            random_seed: Random seed for reproducibility.
            n_jobs: Number of parallel jobs (-1 = all CPUs).
            **kwargs: Additional arguments passed to RandomForest.
        """
        self.task = task
        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.is_fitted = False
        
        # Model parameters
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_seed,
            "n_jobs": n_jobs,
            **kwargs,
        }
        
        # Initialize model
        if task == "regression":
            self.model = RandomForestRegressor(**self.params)
        elif task == "classification":
            self.model = RandomForestClassifier(**self.params)
        else:
            raise ValueError(f"Unknown task: {task}. Use 'regression' or 'classification'.")
        
        # Storage for training history
        self.feature_names: Optional[List[str]] = None
        self.training_metrics: Dict[str, float] = {}
        
        logger.info(f"Initialized {task} QSAR model with {n_estimators} trees")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "QSARModel":
        """Train the QSAR model.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).
            feature_names: Optional list of feature names for interpretation.
            
        Returns:
            Self for method chaining.
        """
        logger.info(f"Training {self.task} model on {X.shape[0]} samples...")
        
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_names = feature_names
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        
        if self.task == "regression":
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            self.training_metrics = {
                "r2_train": float(r2_score(y, y_pred)),
                "rmse_train": float(np.sqrt(mean_squared_error(y, y_pred))),
                "mae_train": float(mean_absolute_error(y, y_pred)),
            }
        else:
            from sklearn.metrics import accuracy_score, roc_auc_score
            self.training_metrics = {
                "accuracy_train": float(accuracy_score(y, y_pred)),
                "roc_auc_train": float(roc_auc_score(y, self.model.predict_proba(X)[:, 1])),
            }
        
        logger.info(f"Training complete. Metrics: {self.training_metrics}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features).
            
        Returns:
            Predictions of shape (n_samples,).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only).
        
        Args:
            X: Feature matrix.
            
        Returns:
            Probability matrix of shape (n_samples, n_classes).
        """
        if self.task != "classification":
            raise ValueError("predict_proba only available for classification.")
        
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate model on test data.
        
        Args:
            X: Feature matrix.
            y: True target values.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        y_pred = self.predict(X)
        
        if self.task == "regression":
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            
            metrics = {
                "r2": float(r2_score(y, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
                "mae": float(mean_absolute_error(y, y_pred)),
                "n_samples": len(y),
            }
            
        else:  # classification
            from sklearn.metrics import (
                accuracy_score, roc_auc_score, precision_score,
                recall_score, f1_score, confusion_matrix
            )
            
            y_proba = self.predict_proba(X)[:, 1]
            
            metrics = {
                "accuracy": float(accuracy_score(y, y_pred)),
                "roc_auc": float(roc_auc_score(y, y_proba)),
                "precision": float(precision_score(y, y_pred, zero_division=0)),
                "recall": float(recall_score(y, y_pred, zero_division=0)),
                "f1": float(f1_score(y, y_pred, zero_division=0)),
                "n_samples": len(y),
            }
            
            # Confusion matrix
            cm = confusion_matrix(y, y_pred)
            metrics["tn"], metrics["fp"] = int(cm[0, 0]), int(cm[0, 1])
            metrics["fn"], metrics["tp"] = int(cm[1, 0]), int(cm[1, 1])
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
    ) -> Dict[str, float]:
        """Perform k-fold cross-validation.
        
        Args:
            X: Feature matrix.
            y: Target values.
            n_folds: Number of folds.
            
        Returns:
            Dictionary with CV metrics (mean and std).
        """
        logger.info(f"Running {n_folds}-fold cross-validation...")
        
        if self.task == "regression":
            scoring = ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]
        else:
            scoring = ["accuracy", "roc_auc", "precision", "recall", "f1"]
        
        from sklearn.model_selection import cross_validate as sklearn_cv
        
        cv_results = sklearn_cv(
            self.model, X, y,
            cv=n_folds,
            scoring=scoring,
            return_train_score=False,
            n_jobs=self.n_jobs,
        )
        
        metrics = {}
        for score_name in scoring:
            key = f"test_{score_name}"
            if key in cv_results:
                values = cv_results[key]
                # Handle negative scores
                if score_name.startswith("neg_"):
                    values = -values
                    score_name = score_name[4:]  # Remove 'neg_'
                
                metrics[f"{score_name}_mean"] = float(np.mean(values))
                metrics[f"{score_name}_std"] = float(np.std(values))
        
        logger.info(f"Cross-validation complete: {metrics}")
        return metrics
    
    def get_feature_importance(
        self,
        top_n: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get feature importance from trained model.
        
        Args:
            top_n: Return only top N features (None = all).
            
        Returns:
            DataFrame with feature names and importance scores.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        importance = self.model.feature_importances_
        
        if self.feature_names is not None:
            names = self.feature_names
        else:
            names = [f"feature_{i}" for i in range(len(importance))]
        
        df = pd.DataFrame({
            "feature": names,
            "importance": importance,
        })
        
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        
        if top_n is not None:
            df = df.head(top_n)
        
        return df
    
    def save(self, path: str) -> None:
        """Save trained model to file.
        
        Args:
            path: Path to save model (pickle format).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Cannot save.")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "model": self.model,
            "task": self.task,
            "params": self.params,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "random_seed": self.random_seed,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "QSARModel":
        """Load trained model from file.
        
        Args:
            path: Path to saved model.
            
        Returns:
            Loaded QSARModel instance.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        instance = cls(
            task=state["task"],
            random_seed=state["random_seed"],
        )
        instance.model = state["model"]
        instance.params = state["params"]
        instance.feature_names = state["feature_names"]
        instance.training_metrics = state["training_metrics"]
        instance.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
        return instance
    
    def save_metrics(self, path: str, metrics: Dict[str, float]) -> None:
        """Save evaluation metrics to JSON file.
        
        Args:
            path: Path to save JSON file.
            metrics: Dictionary of metrics.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {path}")
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dictionary of model parameters.
        """
        return self.params.copy()
    
    def set_params(self, **params: Any) -> "QSARModel":
        """Set model parameters.
        
        Args:
            **params: Parameters to set.
            
        Returns:
            Self for method chaining.
        """
        self.params.update(params)
        self.model.set_params(**params)
        self.is_fitted = False  # Model needs retraining
        return self
    
    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"QSARModel(task='{self.task}', n_estimators={self.params['n_estimators']}, {status})"
