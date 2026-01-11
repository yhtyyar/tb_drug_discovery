"""Advanced Ensemble Models for QSAR prediction.

This module implements sophisticated ensemble methods combining
multiple ML algorithms for improved prediction accuracy and robustness.

Features:
- Voting ensemble (hard/soft)
- Stacking with meta-learner
- Blending ensemble
- Weighted averaging
- Automatic model selection

Example:
    >>> ensemble = StackingEnsemble(task='classification')
    >>> ensemble.fit(X_train, y_train)
    >>> predictions = ensemble.predict(X_test)
    >>> metrics = ensemble.evaluate(X_test, y_test)
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error
)


class BaseEnsemble(BaseEstimator):
    """Base class for ensemble models.
    
    Provides common functionality for all ensemble methods.
    
    Args:
        task: 'classification' or 'regression'.
        random_seed: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        task: str = "classification",
        random_seed: int = 42,
    ):
        self.task = task
        self.random_seed = random_seed
        self.is_fitted = False
        self.base_models = []
        self.model_names = []
    
    def _get_default_models(self) -> List[Tuple[str, BaseEstimator]]:
        """Get default base models for ensemble."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.svm import SVC, SVR
        
        if self.task == "classification":
            return [
                ("rf", RandomForestClassifier(n_estimators=100, random_state=self.random_seed, n_jobs=-1)),
                ("gb", GradientBoostingClassifier(n_estimators=100, random_state=self.random_seed)),
                ("lr", LogisticRegression(max_iter=1000, random_state=self.random_seed)),
                ("svm", SVC(probability=True, random_state=self.random_seed)),
            ]
        else:
            return [
                ("rf", RandomForestRegressor(n_estimators=100, random_state=self.random_seed, n_jobs=-1)),
                ("gb", GradientBoostingRegressor(n_estimators=100, random_state=self.random_seed)),
                ("ridge", Ridge(random_state=self.random_seed)),
                ("svr", SVR()),
            ]
    
    def _add_boosting_models(self, models: List[Tuple[str, BaseEstimator]]) -> List[Tuple[str, BaseEstimator]]:
        """Add XGBoost and LightGBM if available."""
        try:
            import xgboost as xgb
            if self.task == "classification":
                models.append(("xgb", xgb.XGBClassifier(
                    n_estimators=100, random_state=self.random_seed, n_jobs=-1,
                    use_label_encoder=False, eval_metric="logloss"
                )))
            else:
                models.append(("xgb", xgb.XGBRegressor(
                    n_estimators=100, random_state=self.random_seed, n_jobs=-1
                )))
        except ImportError:
            pass
        
        try:
            import lightgbm as lgb
            if self.task == "classification":
                models.append(("lgb", lgb.LGBMClassifier(
                    n_estimators=100, random_state=self.random_seed, n_jobs=-1, verbose=-1
                )))
            else:
                models.append(("lgb", lgb.LGBMRegressor(
                    n_estimators=100, random_state=self.random_seed, n_jobs=-1, verbose=-1
                )))
        except ImportError:
            pass
        
        return models
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble on test data."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        y_pred = self.predict(X)
        
        if self.task == "classification":
            y_proba = self.predict_proba(X)[:, 1] if hasattr(self, "predict_proba") else y_pred
            
            metrics = {
                "accuracy": accuracy_score(y, y_pred),
                "precision": precision_score(y, y_pred, zero_division=0),
                "recall": recall_score(y, y_pred, zero_division=0),
                "f1": f1_score(y, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y, y_proba),
            }
        else:
            metrics = {
                "r2": r2_score(y, y_pred),
                "rmse": np.sqrt(mean_squared_error(y, y_pred)),
                "mae": mean_absolute_error(y, y_pred),
            }
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save ensemble to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Ensemble saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "BaseEnsemble":
        """Load ensemble from file."""
        with open(path, "rb") as f:
            return pickle.load(f)


class VotingEnsemble(BaseEnsemble, ClassifierMixin, RegressorMixin):
    """Voting ensemble combining multiple models.
    
    For classification: soft voting (probability averaging) or hard voting.
    For regression: averaging predictions.
    
    Args:
        models: List of (name, estimator) tuples.
        voting: 'soft' or 'hard' for classification.
        weights: Optional weights for each model.
        task: 'classification' or 'regression'.
        
    Example:
        >>> ensemble = VotingEnsemble(task='classification')
        >>> ensemble.fit(X_train, y_train)
        >>> proba = ensemble.predict_proba(X_test)
    """
    
    def __init__(
        self,
        models: Optional[List[Tuple[str, BaseEstimator]]] = None,
        voting: str = "soft",
        weights: Optional[List[float]] = None,
        task: str = "classification",
        random_seed: int = 42,
    ):
        super().__init__(task, random_seed)
        
        self.voting = voting
        self.weights = weights
        
        if models is None:
            models = self._get_default_models()
            models = self._add_boosting_models(models)
        
        self.model_names = [name for name, _ in models]
        self.base_models = [clone(model) for _, model in models]
        
        logger.info(f"VotingEnsemble initialized with {len(self.base_models)} models: {self.model_names}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "VotingEnsemble":
        """Fit all base models."""
        logger.info(f"Training {len(self.base_models)} base models...")
        
        for name, model in zip(self.model_names, self.base_models):
            logger.debug(f"Training {name}...")
            model.fit(X, y)
        
        self.is_fitted = True
        logger.info("Voting ensemble training complete")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using voting."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        if self.task == "classification":
            if self.voting == "soft":
                proba = self.predict_proba(X)
                return (proba[:, 1] >= 0.5).astype(int)
            else:
                # Hard voting
                predictions = np.array([model.predict(X) for model in self.base_models])
                return np.apply_along_axis(
                    lambda x: np.bincount(x.astype(int)).argmax(),
                    axis=0,
                    arr=predictions
                )
        else:
            predictions = np.array([model.predict(X) for model in self.base_models])
            if self.weights is not None:
                return np.average(predictions, axis=0, weights=self.weights)
            return predictions.mean(axis=0)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only)."""
        if self.task != "classification":
            raise ValueError("predict_proba only for classification")
        
        probas = []
        for model in self.base_models:
            if hasattr(model, "predict_proba"):
                probas.append(model.predict_proba(X))
            else:
                # For models without predict_proba
                pred = model.predict(X)
                proba = np.zeros((len(pred), 2))
                proba[:, 1] = pred
                proba[:, 0] = 1 - pred
                probas.append(proba)
        
        probas = np.array(probas)
        
        if self.weights is not None:
            return np.average(probas, axis=0, weights=self.weights)
        return probas.mean(axis=0)
    
    def get_model_scores(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Get individual model scores."""
        scores = {}
        
        for name, model in zip(self.model_names, self.base_models):
            y_pred = model.predict(X)
            
            if self.task == "classification":
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X)[:, 1]
                else:
                    y_proba = y_pred
                scores[name] = roc_auc_score(y, y_proba)
            else:
                scores[name] = r2_score(y, y_pred)
        
        return scores


class StackingEnsemble(BaseEnsemble, ClassifierMixin, RegressorMixin):
    """Stacking ensemble with meta-learner.
    
    First level: Train base models and get out-of-fold predictions.
    Second level: Train meta-learner on base model predictions.
    
    Args:
        base_models: List of (name, estimator) tuples for first level.
        meta_model: Meta-learner for second level.
        cv_folds: Number of CV folds for out-of-fold predictions.
        use_features: Include original features in meta-learner input.
        task: 'classification' or 'regression'.
        
    Example:
        >>> ensemble = StackingEnsemble(task='classification')
        >>> ensemble.fit(X_train, y_train)
        >>> predictions = ensemble.predict(X_test)
    """
    
    def __init__(
        self,
        base_models: Optional[List[Tuple[str, BaseEstimator]]] = None,
        meta_model: Optional[BaseEstimator] = None,
        cv_folds: int = 5,
        use_features: bool = False,
        task: str = "classification",
        random_seed: int = 42,
    ):
        super().__init__(task, random_seed)
        
        self.cv_folds = cv_folds
        self.use_features = use_features
        
        # Base models
        if base_models is None:
            base_models = self._get_default_models()
            base_models = self._add_boosting_models(base_models)
        
        self.model_names = [name for name, _ in base_models]
        self.base_models = [clone(model) for _, model in base_models]
        
        # Meta-learner
        if meta_model is None:
            if task == "classification":
                from sklearn.linear_model import LogisticRegression
                self.meta_model = LogisticRegression(max_iter=1000, random_state=random_seed)
            else:
                from sklearn.linear_model import Ridge
                self.meta_model = Ridge(random_state=random_seed)
        else:
            self.meta_model = clone(meta_model)
        
        # Fitted models for final predictions
        self.fitted_base_models = []
        
        logger.info(f"StackingEnsemble initialized with {len(self.base_models)} base models")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsemble":
        """Fit stacking ensemble."""
        logger.info("Training stacking ensemble...")
        
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        # Generate out-of-fold predictions for meta-learner training
        if self.task == "classification":
            oof_predictions = np.zeros((n_samples, n_models))
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
        else:
            oof_predictions = np.zeros((n_samples, n_models))
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
        
        # Get OOF predictions from each base model
        for i, (name, model) in enumerate(zip(self.model_names, self.base_models)):
            logger.debug(f"Getting OOF predictions for {name}...")
            
            if self.task == "classification" and hasattr(model, "predict_proba"):
                oof_predictions[:, i] = cross_val_predict(
                    clone(model), X, y, cv=cv, method="predict_proba", n_jobs=-1
                )[:, 1]
            else:
                oof_predictions[:, i] = cross_val_predict(
                    clone(model), X, y, cv=cv, n_jobs=-1
                )
        
        # Prepare meta-learner input
        if self.use_features:
            meta_X = np.hstack([X, oof_predictions])
        else:
            meta_X = oof_predictions
        
        # Train meta-learner
        logger.debug("Training meta-learner...")
        self.meta_model.fit(meta_X, y)
        
        # Fit base models on full training data for inference
        logger.debug("Fitting base models on full data...")
        self.fitted_base_models = []
        for name, model in zip(self.model_names, self.base_models):
            fitted_model = clone(model)
            fitted_model.fit(X, y)
            self.fitted_base_models.append(fitted_model)
        
        self.is_fitted = True
        logger.info("Stacking ensemble training complete")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using stacked ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.task != "classification":
            raise ValueError("predict_proba only for classification")
        
        meta_features = self._get_meta_features(X)
        
        if hasattr(self.meta_model, "predict_proba"):
            return self.meta_model.predict_proba(meta_features)
        else:
            pred = self.meta_model.predict(meta_features)
            proba = np.zeros((len(pred), 2))
            proba[:, 1] = pred
            proba[:, 0] = 1 - pred
            return proba
    
    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Get base model predictions as meta-features."""
        n_samples = X.shape[0]
        n_models = len(self.fitted_base_models)
        
        base_predictions = np.zeros((n_samples, n_models))
        
        for i, model in enumerate(self.fitted_base_models):
            if self.task == "classification" and hasattr(model, "predict_proba"):
                base_predictions[:, i] = model.predict_proba(X)[:, 1]
            else:
                base_predictions[:, i] = model.predict(X)
        
        if self.use_features:
            return np.hstack([X, base_predictions])
        return base_predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get meta-learner feature importance."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        if hasattr(self.meta_model, "coef_"):
            importance = np.abs(self.meta_model.coef_).flatten()
        elif hasattr(self.meta_model, "feature_importances_"):
            importance = self.meta_model.feature_importances_
        else:
            return pd.DataFrame()
        
        feature_names = self.model_names.copy()
        if self.use_features:
            feature_names = [f"feat_{i}" for i in range(len(importance) - len(self.model_names))] + feature_names
        
        return pd.DataFrame({
            "feature": feature_names[:len(importance)],
            "importance": importance,
        }).sort_values("importance", ascending=False)


class BlendingEnsemble(BaseEnsemble, ClassifierMixin, RegressorMixin):
    """Blending ensemble using holdout set for meta-learner training.
    
    Simpler than stacking - uses a single holdout set instead of CV
    for generating meta-features.
    
    Args:
        base_models: List of (name, estimator) tuples.
        meta_model: Meta-learner model.
        holdout_ratio: Fraction of training data for holdout.
        task: 'classification' or 'regression'.
    """
    
    def __init__(
        self,
        base_models: Optional[List[Tuple[str, BaseEstimator]]] = None,
        meta_model: Optional[BaseEstimator] = None,
        holdout_ratio: float = 0.2,
        task: str = "classification",
        random_seed: int = 42,
    ):
        super().__init__(task, random_seed)
        
        self.holdout_ratio = holdout_ratio
        
        if base_models is None:
            base_models = self._get_default_models()
            base_models = self._add_boosting_models(base_models)
        
        self.model_names = [name for name, _ in base_models]
        self.base_models = [clone(model) for _, model in base_models]
        
        if meta_model is None:
            if task == "classification":
                from sklearn.linear_model import LogisticRegression
                self.meta_model = LogisticRegression(max_iter=1000, random_state=random_seed)
            else:
                from sklearn.linear_model import Ridge
                self.meta_model = Ridge()
        else:
            self.meta_model = clone(meta_model)
        
        self.fitted_base_models = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BlendingEnsemble":
        """Fit blending ensemble."""
        from sklearn.model_selection import train_test_split
        
        # Split into train and holdout
        if self.task == "classification":
            X_train, X_holdout, y_train, y_holdout = train_test_split(
                X, y, test_size=self.holdout_ratio, stratify=y, random_state=self.random_seed
            )
        else:
            X_train, X_holdout, y_train, y_holdout = train_test_split(
                X, y, test_size=self.holdout_ratio, random_state=self.random_seed
            )
        
        # Train base models on training set
        n_holdout = X_holdout.shape[0]
        n_models = len(self.base_models)
        holdout_predictions = np.zeros((n_holdout, n_models))
        
        self.fitted_base_models = []
        for i, (name, model) in enumerate(zip(self.model_names, self.base_models)):
            logger.debug(f"Training {name}...")
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            self.fitted_base_models.append(fitted)
            
            if self.task == "classification" and hasattr(fitted, "predict_proba"):
                holdout_predictions[:, i] = fitted.predict_proba(X_holdout)[:, 1]
            else:
                holdout_predictions[:, i] = fitted.predict(X_holdout)
        
        # Train meta-learner on holdout predictions
        self.meta_model.fit(holdout_predictions, y_holdout)
        
        # Retrain base models on full data for inference
        self.fitted_base_models = []
        for name, model in zip(self.model_names, self.base_models):
            fitted = clone(model)
            fitted.fit(X, y)
            self.fitted_base_models.append(fitted)
        
        self.is_fitted = True
        logger.info("Blending ensemble training complete")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted")
        
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if self.task != "classification":
            raise ValueError("Only for classification")
        
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict_proba(meta_features)
    
    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Get meta-features from base model predictions."""
        n_samples = X.shape[0]
        n_models = len(self.fitted_base_models)
        predictions = np.zeros((n_samples, n_models))
        
        for i, model in enumerate(self.fitted_base_models):
            if self.task == "classification" and hasattr(model, "predict_proba"):
                predictions[:, i] = model.predict_proba(X)[:, 1]
            else:
                predictions[:, i] = model.predict(X)
        
        return predictions


def create_ensemble(
    ensemble_type: str = "stacking",
    task: str = "classification",
    **kwargs,
) -> BaseEnsemble:
    """Factory function to create ensemble models.
    
    Args:
        ensemble_type: 'voting', 'stacking', or 'blending'.
        task: 'classification' or 'regression'.
        **kwargs: Additional arguments for the ensemble.
        
    Returns:
        Ensemble model instance.
    """
    ensembles = {
        "voting": VotingEnsemble,
        "stacking": StackingEnsemble,
        "blending": BlendingEnsemble,
    }
    
    if ensemble_type not in ensembles:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")
    
    return ensembles[ensemble_type](task=task, **kwargs)


def train_best_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task: str = "classification",
) -> Tuple[BaseEnsemble, Dict[str, float]]:
    """Train and select the best ensemble type.
    
    Args:
        X_train: Training features.
        y_train: Training targets.
        X_val: Validation features.
        y_val: Validation targets.
        task: 'classification' or 'regression'.
        
    Returns:
        Tuple of (best_ensemble, evaluation_metrics).
    """
    results = {}
    
    for ensemble_type in ["voting", "stacking", "blending"]:
        logger.info(f"Training {ensemble_type} ensemble...")
        
        ensemble = create_ensemble(ensemble_type, task)
        ensemble.fit(X_train, y_train)
        
        metrics = ensemble.evaluate(X_val, y_val)
        results[ensemble_type] = {
            "ensemble": ensemble,
            "metrics": metrics,
        }
        
        score_key = "roc_auc" if task == "classification" else "r2"
        logger.info(f"{ensemble_type}: {score_key}={metrics[score_key]:.4f}")
    
    # Select best
    score_key = "roc_auc" if task == "classification" else "r2"
    best_type = max(results, key=lambda x: results[x]["metrics"][score_key])
    
    logger.info(f"Best ensemble: {best_type}")
    return results[best_type]["ensemble"], results[best_type]["metrics"]
