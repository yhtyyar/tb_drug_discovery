"""Evaluation metrics, cross-validation, and uncertainty quantification.

This package provides:
- Standard ML metrics (ROC-AUC, PR-AUC, R², RMSE, BEDROC, EF, etc.)
- Cross-validation utilities with scaffold-aware splits
- Uncertainty quantification: conformal prediction, MC-Dropout, ensemble
- Generation quality metrics (validity, novelty, QED, SA, diversity)
- Data drift detection
"""

from .metrics import (
    calculate_metrics,
    calculate_classification_metrics,
    get_roc_curve,
    get_precision_recall_curve,
    calculate_enrichment_factor,
    calculate_bedroc,
)
from .cross_validation import cross_validate_model, nested_cross_validation
from .generation_metrics import evaluate_generation, compute_validity, compute_novelty
from .conformal_prediction import ConformalPredictor, ClassificationConformalPredictor
from .drift_detector import DescriptorDriftDetector

try:
    from .uncertainty import (
        MCDropout,
        DeepEnsemble,
        TemperatureScaling,
        create_uncertainty_estimator,
        evaluate_uncertainty_quality,
    )
    _HAS_UNCERTAINTY = True
except ImportError:
    _HAS_UNCERTAINTY = False

__all__ = [
    # Metrics
    "calculate_metrics",
    "calculate_classification_metrics",
    "get_roc_curve",
    "get_precision_recall_curve",
    "calculate_enrichment_factor",
    "calculate_bedroc",
    # Cross-validation
    "cross_validate_model",
    "nested_cross_validation",
    # Generation metrics
    "evaluate_generation",
    "compute_validity",
    "compute_novelty",
    # Uncertainty
    "ConformalPredictor",
    "ClassificationConformalPredictor",
    "DescriptorDriftDetector",
]
