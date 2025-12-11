"""Evaluation metrics for QSAR and ML models.

This module provides comprehensive metric calculation for both
regression and classification tasks in drug discovery.
"""

from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Calculate regression metrics.
    
    Args:
        y_true: True target values.
        y_pred: Predicted values.
        
    Returns:
        Dictionary with regression metrics.
        
    Example:
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([1.1, 2.1, 2.9])
        >>> metrics = calculate_metrics(y_true, y_pred)
        >>> print(f"R² = {metrics['r2']:.3f}")
    """
    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "n_samples": len(y_true),
    }
    
    # Pearson correlation
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    metrics["pearson_r"] = float(correlation) if not np.isnan(correlation) else 0.0
    
    # Spearman correlation
    from scipy.stats import spearmanr
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    metrics["spearman_r"] = float(spearman_r) if not np.isnan(spearman_r) else 0.0
    
    return metrics


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate classification metrics.
    
    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_proba: Predicted probabilities for positive class (optional).
        
    Returns:
        Dictionary with classification metrics.
        
    Example:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> metrics = calculate_classification_metrics(y_true, y_pred)
        >>> print(f"Accuracy = {metrics['accuracy']:.3f}")
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_samples": len(y_true),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        metrics["tn"] = int(cm[0, 0])
        metrics["fp"] = int(cm[0, 1])
        metrics["fn"] = int(cm[1, 0])
        metrics["tp"] = int(cm[1, 1])
        
        # Specificity
        metrics["specificity"] = float(cm[0, 0] / (cm[0, 0] + cm[0, 1])) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0
        
        # Balanced accuracy
        sensitivity = metrics["recall"]
        specificity = metrics["specificity"]
        metrics["balanced_accuracy"] = float((sensitivity + specificity) / 2)
    
    # ROC-AUC if probabilities provided
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = 0.5  # Default for edge cases
    
    return metrics


def get_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Calculate ROC curve data.
    
    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities.
        
    Returns:
        Tuple of (fpr, tpr, thresholds, auc_score).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, thresholds, float(roc_auc)


def get_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Calculate Precision-Recall curve data.
    
    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities.
        
    Returns:
        Tuple of (precision, recall, thresholds, auc_score).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    return precision, recall, thresholds, float(pr_auc)


def calculate_enrichment_factor(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    top_percent: float = 0.01,
) -> float:
    """Calculate enrichment factor for virtual screening.
    
    EF measures how many times better the model is at finding
    actives in the top X% compared to random selection.
    
    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities.
        top_percent: Top fraction to consider (default 1%).
        
    Returns:
        Enrichment factor.
    """
    n_total = len(y_true)
    n_actives_total = y_true.sum()
    
    if n_actives_total == 0:
        return 0.0
    
    # Sort by predicted probability (descending)
    sorted_indices = np.argsort(y_proba)[::-1]
    
    # Top N compounds
    n_top = max(1, int(n_total * top_percent))
    top_indices = sorted_indices[:n_top]
    
    # Count actives in top N
    n_actives_top = y_true[top_indices].sum()
    
    # Calculate enrichment factor
    # EF = (Actives in top X% / Total in top X%) / (Total actives / Total compounds)
    hit_rate_top = n_actives_top / n_top
    hit_rate_random = n_actives_total / n_total
    
    ef = hit_rate_top / hit_rate_random if hit_rate_random > 0 else 0.0
    
    return float(ef)


def calculate_bedroc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    alpha: float = 20.0,
) -> float:
    """Calculate BEDROC score for early recognition.
    
    BEDROC (Boltzmann-Enhanced Discrimination of ROC) emphasizes
    early recognition of actives, important for virtual screening.
    
    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities.
        alpha: Exponential decay parameter (higher = more emphasis on top).
        
    Returns:
        BEDROC score (0-1, higher is better).
    """
    n = len(y_true)
    n_actives = y_true.sum()
    
    if n_actives == 0 or n_actives == n:
        return 0.5  # Cannot calculate
    
    # Sort by score (descending) and get ranks of actives
    sorted_indices = np.argsort(y_proba)[::-1]
    ranks = np.where(y_true[sorted_indices] == 1)[0] + 1  # 1-indexed ranks
    
    # Calculate BEDROC
    s = np.sum(np.exp(-alpha * ranks / n))
    
    # Random and perfect scores
    ra = n_actives / n
    ri = (1 - np.exp(-alpha * ra)) / (np.exp(alpha / n) - 1)
    ro = (1 - np.exp(alpha * ra)) * n_actives / (n * (1 - np.exp(-alpha)))
    
    bedroc = (s - ri) / (ro - ri) if (ro - ri) != 0 else 0.5
    
    return float(np.clip(bedroc, 0, 1))
