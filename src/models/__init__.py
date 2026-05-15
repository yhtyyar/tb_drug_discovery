"""Machine learning models for drug discovery.

This package provides:
- QSAR models (Random Forest, XGBoost, LightGBM)
- Ensemble methods (Voting, Stacking, Blending)
- Hyperparameter optimization with Optuna

Classes:
    QSARModel: Random Forest based QSAR model
    VotingEnsemble: Voting ensemble of multiple models
    StackingEnsemble: Stacking ensemble with meta-learner
    BlendingEnsemble: Blending ensemble with holdout
    QSAROptimizer: Hyperparameter optimization for QSAR
"""

from .qsar_model import QSARModel
from .calibration import QSARCalibrator
from .applicability_domain import BoundingBoxAD, LeverageAD, EnsembleAD
from .active_learning import ActiveLearner, UCB, ExpectedImprovement, ThompsonSampling, MaxEntropy
from .multitask_qsar import (
    MultiTaskQSAR,
    MultiTaskQSARSklearn,
    MultiTaskConfig,
    create_multitask_model,
    evaluate_multitask,
    TB_TARGETS,
    DEFAULT_TARGETS,
)

try:
    from .chemberta import ChemBERTaFeaturizer, ChemBERTaQSAR, compare_representations
    _HAS_CHEMBERTA = True
except ImportError:
    _HAS_CHEMBERTA = False
from .ensemble import (
    VotingEnsemble,
    StackingEnsemble, 
    BlendingEnsemble,
    create_ensemble,
    train_best_ensemble,
)
from .hyperopt import (
    QSAROptimizer,
    GNNOptimizer,
    OptimizationConfig,
    run_full_optimization,
)

__all__ = [
    "QSARModel",
    "VotingEnsemble",
    "StackingEnsemble",
    "BlendingEnsemble",
    "create_ensemble",
    "train_best_ensemble",
    "QSAROptimizer",
    "GNNOptimizer",
    "OptimizationConfig",
    "run_full_optimization",
    "QSARCalibrator",
    "BoundingBoxAD",
    "LeverageAD",
    "EnsembleAD",
    "ActiveLearner",
    "UCB",
    "ExpectedImprovement",
    "ThompsonSampling",
    "MaxEntropy",
    "MultiTaskQSAR",
    "MultiTaskQSARSklearn",
    "MultiTaskConfig",
    "create_multitask_model",
    "evaluate_multitask",
    "TB_TARGETS",
    "DEFAULT_TARGETS",
]
