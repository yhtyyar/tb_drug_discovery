"""Cross-validation utilities for QSAR models.

This module provides robust cross-validation with proper handling
of molecular data and comprehensive metric reporting.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import KFold, StratifiedKFold

from .metrics import calculate_classification_metrics, calculate_metrics


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    task: str = "regression",
    stratify: bool = True,
    random_seed: int = 42,
    return_predictions: bool = False,
) -> Dict[str, Any]:
    """Perform k-fold cross-validation with detailed metrics.
    
    Args:
        model: Model with fit() and predict() methods.
        X: Feature matrix.
        y: Target values.
        n_folds: Number of CV folds.
        task: 'regression' or 'classification'.
        stratify: Use stratified folds (classification only).
        random_seed: Random seed for reproducibility.
        return_predictions: Whether to return all predictions.
        
    Returns:
        Dictionary with:
        - Fold-wise metrics
        - Aggregated statistics (mean, std)
        - Optional predictions
        
    Example:
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> model = RandomForestRegressor(n_estimators=100)
        >>> results = cross_validate_model(model, X, y, n_folds=5)
        >>> print(f"R² = {results['r2_mean']:.3f} ± {results['r2_std']:.3f}")
    """
    logger.info(f"Running {n_folds}-fold cross-validation for {task}...")
    
    # Select appropriate splitter
    if task == "classification" and stratify:
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    else:
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    # Storage for results
    fold_metrics: List[Dict[str, float]] = []
    all_predictions = np.zeros(len(y))
    all_probabilities = np.zeros(len(y)) if task == "classification" else None
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Clone and train model
        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)
        
        # Predict
        y_pred = fold_model.predict(X_val)
        all_predictions[val_idx] = y_pred
        
        # Calculate metrics
        if task == "regression":
            metrics = calculate_metrics(y_val, y_pred)
        else:
            y_proba = fold_model.predict_proba(X_val)[:, 1] if hasattr(fold_model, "predict_proba") else None
            if y_proba is not None and all_probabilities is not None:
                all_probabilities[val_idx] = y_proba
            metrics = calculate_classification_metrics(y_val, y_pred, y_proba)
        
        metrics["fold"] = fold_idx + 1
        fold_metrics.append(metrics)
        
        logger.debug(f"Fold {fold_idx + 1}: {metrics}")
    
    # Aggregate results
    results: Dict[str, Any] = {}
    
    # Calculate mean and std for each metric
    metric_names = [k for k in fold_metrics[0].keys() if k != "fold" and not k.startswith("n_")]
    
    for metric in metric_names:
        values = [fm[metric] for fm in fold_metrics]
        results[f"{metric}_mean"] = float(np.mean(values))
        results[f"{metric}_std"] = float(np.std(values))
        results[f"{metric}_values"] = values
    
    # Overall metrics on all predictions
    if task == "regression":
        overall = calculate_metrics(y, all_predictions)
    else:
        overall = calculate_classification_metrics(y, all_predictions, all_probabilities)
    
    results["overall"] = overall
    results["fold_metrics"] = fold_metrics
    results["n_folds"] = n_folds
    results["n_samples"] = len(y)
    
    if return_predictions:
        results["predictions"] = all_predictions
        if all_probabilities is not None:
            results["probabilities"] = all_probabilities
    
    # Log summary
    if task == "regression":
        logger.info(
            f"CV Results: R² = {results['r2_mean']:.3f} ± {results['r2_std']:.3f}, "
            f"RMSE = {results['rmse_mean']:.3f} ± {results['rmse_std']:.3f}"
        )
    else:
        logger.info(
            f"CV Results: ROC-AUC = {results.get('roc_auc_mean', 0):.3f} ± {results.get('roc_auc_std', 0):.3f}, "
            f"Accuracy = {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}"
        )
    
    return results


def nested_cross_validation(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Dict[str, List[Any]],
    outer_folds: int = 5,
    inner_folds: int = 3,
    task: str = "regression",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Nested cross-validation for unbiased model evaluation.
    
    Outer loop: Estimates generalization performance
    Inner loop: Selects best hyperparameters
    
    Args:
        model_factory: Callable that returns a new model instance.
        X: Feature matrix.
        y: Target values.
        param_grid: Dictionary of hyperparameters to search.
        outer_folds: Number of outer CV folds.
        inner_folds: Number of inner CV folds.
        task: 'regression' or 'classification'.
        random_seed: Random seed.
        
    Returns:
        Dictionary with nested CV results and best parameters per fold.
    """
    from sklearn.model_selection import GridSearchCV
    
    logger.info(f"Running nested CV: {outer_folds} outer × {inner_folds} inner folds")
    
    # Outer CV
    if task == "classification":
        outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_seed)
    else:
        outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=random_seed)
    
    fold_results = []
    best_params_per_fold = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Inner CV with grid search
        model = model_factory()
        scoring = "r2" if task == "regression" else "roc_auc"
        
        inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_seed) \
                   if task == "classification" else \
                   KFold(n_splits=inner_folds, shuffle=True, random_state=random_seed)
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
        )
        
        grid_search.fit(X_train, y_train)
        
        # Evaluate on outer test fold
        y_pred = grid_search.predict(X_test)
        
        if task == "regression":
            metrics = calculate_metrics(y_test, y_pred)
        else:
            y_proba = grid_search.predict_proba(X_test)[:, 1] if hasattr(grid_search, "predict_proba") else None
            metrics = calculate_classification_metrics(y_test, y_pred, y_proba)
        
        fold_results.append(metrics)
        best_params_per_fold.append(grid_search.best_params_)
        
        logger.debug(f"Outer fold {fold_idx + 1}: Best params = {grid_search.best_params_}")
    
    # Aggregate results
    results = {
        "fold_results": fold_results,
        "best_params_per_fold": best_params_per_fold,
    }
    
    # Calculate mean/std
    for metric in fold_results[0].keys():
        if not metric.startswith("n_"):
            values = [fr[metric] for fr in fold_results]
            results[f"{metric}_mean"] = float(np.mean(values))
            results[f"{metric}_std"] = float(np.std(values))
    
    return results


def learning_curve_analysis(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: Optional[List[float]] = None,
    n_folds: int = 5,
    task: str = "regression",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Analyze learning curves to detect overfitting/underfitting.
    
    Args:
        model: Model to analyze.
        X: Feature matrix.
        y: Target values.
        train_sizes: Fractions of training data to use.
        n_folds: Number of CV folds.
        task: 'regression' or 'classification'.
        random_seed: Random seed.
        
    Returns:
        DataFrame with train/test scores at each training size.
    """
    from sklearn.model_selection import learning_curve
    
    if train_sizes is None:
        train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    scoring = "r2" if task == "regression" else "roc_auc"
    
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=n_folds,
        scoring=scoring,
        n_jobs=-1,
        random_state=random_seed,
    )
    
    results = pd.DataFrame({
        "train_size": train_sizes_abs,
        "train_score_mean": train_scores.mean(axis=1),
        "train_score_std": train_scores.std(axis=1),
        "test_score_mean": test_scores.mean(axis=1),
        "test_score_std": test_scores.std(axis=1),
    })
    
    return results
