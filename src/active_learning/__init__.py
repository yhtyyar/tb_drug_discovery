"""Active Learning for efficient drug discovery.

This module provides active learning strategies for selecting
the most informative compounds for experimental testing.

Strategies:
- Uncertainty sampling
- Query-by-committee
- Expected improvement
- Batch mode active learning

Example:
    >>> from src.active_learning import ActiveLearner, UncertaintySampling
    >>> strategy = UncertaintySampling(model)
    >>> next_compounds = strategy.select(candidates, n_select=10)
"""

from .strategies import (
    ActiveLearningStrategy,
    UncertaintySampling,
    QueryByCommittee,
    ExpectedImprovement,
    DiversitySampling,
)
from .learner import ActiveLearner

__all__ = [
    "ActiveLearningStrategy",
    "UncertaintySampling",
    "QueryByCommittee",
    "ExpectedImprovement",
    "DiversitySampling",
    "ActiveLearner",
]
