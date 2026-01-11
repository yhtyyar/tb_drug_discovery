"""Evaluation metrics, cross-validation, and uncertainty quantification.

This package provides:
- Standard ML metrics (ROC-AUC, R², RMSE, etc.)
- Cross-validation utilities
- Uncertainty quantification methods
"""

from .metrics import calculate_metrics, calculate_classification_metrics
from .cross_validation import cross_validate_model
from .uncertainty import (
    MCDropout,
    DeepEnsemble,
    ConformalPredictor,
    ClassificationConformalPredictor,
    TemperatureScaling,
    create_uncertainty_estimator,
    evaluate_uncertainty_quality,
)

__all__ = [
    "calculate_metrics",
    "calculate_classification_metrics",
    "cross_validate_model",
    "MCDropout",
    "DeepEnsemble",
    "ConformalPredictor",
    "ClassificationConformalPredictor",
    "TemperatureScaling",
    "create_uncertainty_estimator",
    "evaluate_uncertainty_quality",
]
