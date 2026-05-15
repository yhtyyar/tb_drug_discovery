"""
Production-grade QSAR Pipeline using sklearn Pipeline API.

This module provides a sklearn-compatible pipeline that chains descriptor
calculation, scaling, and model training into a single object. Using a
Pipeline prevents data leakage — the scaler is always fit only on training
data and applied identically to test data.

Example usage:
    >>> from src.pipeline.qsar_pipeline import build_pipeline, save_pipeline, load_pipeline
    >>> pipeline = build_pipeline(task='classification', n_estimators=200)
    >>> pipeline.fit(train_smiles, y_train)
    >>> probs = pipeline.predict_proba(test_smiles)
    >>> save_pipeline(pipeline, 'models/qsar_pipeline.joblib')
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Descriptor Transformer
# ---------------------------------------------------------------------------

class DescriptorTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that converts SMILES to descriptor arrays.

    Wraps the project's DescriptorCalculator so it slots directly into a
    sklearn Pipeline. Because BaseEstimator is used, get_params() / set_params()
    are available for free, enabling GridSearchCV compatibility.

    Parameters
    ----------
    descriptor_types : list of str, optional
        Which descriptor families to compute.  Defaults to all available
        families in DescriptorCalculator.
    n_jobs : int
        Passed to DescriptorCalculator for parallel RDKit computation.
    invalid_fill : float
        Value used to replace NaN/Inf entries that arise from invalid SMILES
        or descriptor failures.  Default 0.0.

    Attributes
    ----------
    feature_names_ : list of str
        Set after fit(); the ordered list of descriptor names.
    n_features_out_ : int
        Number of output features.
    """

    def __init__(
        self,
        descriptor_types: Optional[List[str]] = None,
        n_jobs: int = 1,
        invalid_fill: float = 0.0,
    ) -> None:
        self.descriptor_types = descriptor_types
        self.n_jobs = n_jobs
        self.invalid_fill = invalid_fill

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_calculator(self):
        """Lazily import DescriptorCalculator to avoid hard dep at module level."""
        try:
            from src.features.descriptors import DescriptorCalculator
        except ImportError:
            # Fallback: use RDKit directly for basic Morgan fingerprints
            from src.features.descriptors import DescriptorCalculator  # type: ignore
        kwargs: Dict[str, Any] = {"n_jobs": self.n_jobs}
        if self.descriptor_types is not None:
            kwargs["descriptor_types"] = self.descriptor_types
        return DescriptorCalculator(**kwargs)

    def _compute(self, smiles_list: Sequence[str]) -> np.ndarray:
        """Compute descriptors, replacing invalid entries with invalid_fill."""
        calculator = self._get_calculator()
        try:
            matrix = calculator.calculate(list(smiles_list))
        except AttributeError:
            # Some implementations expose compute() instead of calculate()
            matrix = calculator.compute(list(smiles_list))

        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix, dtype=float)

        # Replace NaN / Inf
        if not np.isfinite(matrix).all():
            logger.warning(
                "Descriptor matrix contains %d non-finite values; replacing with %.1f",
                (~np.isfinite(matrix)).sum(),
                self.invalid_fill,
            )
            matrix = np.where(np.isfinite(matrix), matrix, self.invalid_fill)

        return matrix.astype(np.float32)

    # ------------------------------------------------------------------
    # Sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: Sequence[str], y=None) -> "DescriptorTransformer":
        """Fit the transformer (resolves feature names).

        Parameters
        ----------
        X : sequence of str
            SMILES strings.
        y : ignored

        Returns
        -------
        self
        """
        matrix = self._compute(X)
        self.n_features_out_: int = matrix.shape[1]

        calculator = self._get_calculator()
        try:
            self.feature_names_: List[str] = list(calculator.get_feature_names())
        except AttributeError:
            self.feature_names_ = [f"desc_{i}" for i in range(self.n_features_out_)]

        self._is_fitted = True
        return self

    def transform(self, X: Sequence[str], y=None) -> np.ndarray:
        """Transform SMILES strings to descriptor matrix.

        Parameters
        ----------
        X : sequence of str
            SMILES strings.

        Returns
        -------
        np.ndarray of shape (n_samples, n_descriptors)
        """
        check_is_fitted(self, "_is_fitted")
        matrix = self._compute(X)

        # Guard against shape mismatch
        if matrix.shape[1] != self.n_features_out_:
            raise ValueError(
                f"Expected {self.n_features_out_} descriptors but got "
                f"{matrix.shape[1]}.  Ensure the same descriptor_types are used "
                "for training and inference."
            )
        return matrix

    def fit_transform(self, X: Sequence[str], y=None, **fit_params) -> np.ndarray:
        """Fit and transform in one pass (more efficient than fit then transform)."""
        matrix = self._compute(X)
        self.n_features_out_ = matrix.shape[1]

        calculator = self._get_calculator()
        try:
            self.feature_names_ = list(calculator.get_feature_names())
        except AttributeError:
            self.feature_names_ = [f"desc_{i}" for i in range(self.n_features_out_)]

        self._is_fitted = True
        return matrix

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return feature names for output features.

        Compatible with sklearn's set_output API.

        Parameters
        ----------
        input_features : ignored (kept for API compatibility)

        Returns
        -------
        np.ndarray of str
        """
        check_is_fitted(self, "_is_fitted")
        return np.array(self.feature_names_, dtype=object)


# ---------------------------------------------------------------------------
# QSAR Pipeline wrapper
# ---------------------------------------------------------------------------

class QSARPipeline:
    """Production QSAR pipeline: descriptors → scaling → model.

    Wraps a sklearn Pipeline so the public API accepts raw SMILES strings
    rather than pre-computed descriptor arrays.  This single object can be
    serialised with joblib and deployed without any pre-processing step
    living outside the object.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The underlying pipeline (DescriptorTransformer | scaler | estimator).
    task : {'classification', 'regression'}
        Governs whether predict_proba is available.

    Examples
    --------
    >>> qsar = build_pipeline(task='classification', n_estimators=100)
    >>> qsar.fit(train_smiles, y_train)
    >>> y_pred = qsar.predict(test_smiles)
    >>> y_prob = qsar.predict_proba(test_smiles)[:, 1]
    """

    def __init__(
        self,
        pipeline: Pipeline,
        task: Literal["classification", "regression"] = "classification",
    ) -> None:
        self.pipeline = pipeline
        self.task = task

    # ------------------------------------------------------------------
    # Core API — accepts raw SMILES
    # ------------------------------------------------------------------

    def fit(self, smiles: Sequence[str], y: np.ndarray, **fit_params) -> "QSARPipeline":
        """Fit all pipeline steps on training SMILES.

        Parameters
        ----------
        smiles : sequence of str
        y : array-like of shape (n_samples,)
        fit_params : passed to sklearn Pipeline.fit()

        Returns
        -------
        self
        """
        self.pipeline.fit(smiles, y, **fit_params)
        self.classes_: Optional[np.ndarray] = getattr(
            self.pipeline.named_steps.get("model"), "classes_", None
        )
        return self

    def predict(self, smiles: Sequence[str]) -> np.ndarray:
        """Predict labels / values for SMILES strings.

        Parameters
        ----------
        smiles : sequence of str

        Returns
        -------
        np.ndarray
        """
        return self.pipeline.predict(smiles)

    def predict_proba(self, smiles: Sequence[str]) -> np.ndarray:
        """Predict class probabilities (classification only).

        Parameters
        ----------
        smiles : sequence of str

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)

        Raises
        ------
        AttributeError if task is 'regression'.
        """
        if self.task != "classification":
            raise AttributeError("predict_proba is only available for classification tasks.")
        return self.pipeline.predict_proba(smiles)

    def score(self, smiles: Sequence[str], y: np.ndarray) -> float:
        """Return the pipeline's score on the given data.

        For classifiers this is accuracy; for regressors it is R².

        Parameters
        ----------
        smiles : sequence of str
        y : array-like

        Returns
        -------
        float
        """
        return self.pipeline.score(smiles, y)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_feature_names(self) -> List[str]:
        """Return descriptor names from the DescriptorTransformer step."""
        transformer: DescriptorTransformer = self.pipeline.named_steps["descriptors"]
        return list(transformer.get_feature_names_out())

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """Return feature importances from the final estimator, if available."""
        model = self.pipeline.named_steps.get("model")
        return getattr(model, "feature_importances_", None)

    @property
    def named_steps(self):
        """Expose sklearn Pipeline named_steps for direct step access."""
        return self.pipeline.named_steps

    def __repr__(self) -> str:  # pragma: no cover
        return f"QSARPipeline(task={self.task!r}, pipeline={self.pipeline})"


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_pipeline(
    task: Literal["classification", "regression"] = "classification",
    n_estimators: int = 200,
    random_seed: int = 42,
    descriptor_kwargs: Optional[Dict[str, Any]] = None,
    max_depth: Optional[int] = None,
    min_samples_leaf: int = 1,
    class_weight: Optional[Union[str, dict]] = "balanced",
) -> QSARPipeline:
    """Factory that constructs a ready-to-use QSARPipeline.

    Parameters
    ----------
    task : {'classification', 'regression'}
        Determines which sklearn estimator is placed at the end of the pipeline.
    n_estimators : int
        Number of trees in the Random Forest.
    random_seed : int
        Random seed for reproducibility.
    descriptor_kwargs : dict, optional
        Keyword arguments forwarded to DescriptorTransformer.__init__().
        E.g. ``{'descriptor_types': ['morgan', 'rdkit'], 'n_jobs': 4}``.
    max_depth : int or None
        Maximum depth of each tree.  None = unlimited.
    min_samples_leaf : int
        Minimum number of samples required to be at a leaf node.
    class_weight : str or dict or None
        Only relevant for classification. 'balanced' adjusts weights inversely
        proportional to class frequencies (recommended for imbalanced TB data).

    Returns
    -------
    QSARPipeline

    Examples
    --------
    >>> pipeline = build_pipeline(task='classification', n_estimators=500,
    ...                           descriptor_kwargs={'n_jobs': 8})
    >>> pipeline.fit(train_smiles, y_train)
    """
    desc_kwargs = descriptor_kwargs or {}
    descriptor_step = DescriptorTransformer(**desc_kwargs)
    scaler_step = RobustScaler()

    if task == "classification":
        model_step: Union[RandomForestClassifier, RandomForestRegressor] = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_seed,
            n_jobs=-1,
        )
    elif task == "regression":
        model_step = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")

    sk_pipeline = Pipeline(
        steps=[
            ("descriptors", descriptor_step),
            ("scaler", scaler_step),
            ("model", model_step),
        ],
        memory=None,  # set to a directory path to cache fit transforms
    )

    return QSARPipeline(pipeline=sk_pipeline, task=task)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def save_pipeline(pipeline: QSARPipeline, path: Union[str, Path]) -> None:
    """Persist a fitted QSARPipeline to disk using joblib.

    Parameters
    ----------
    pipeline : QSARPipeline
        A fitted pipeline.
    path : str or Path
        Destination file path.  Convention: use '.joblib' extension.

    Raises
    ------
    TypeError if pipeline is not a QSARPipeline instance.
    """
    if not isinstance(pipeline, QSARPipeline):
        raise TypeError(f"Expected QSARPipeline, got {type(pipeline)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path, compress=3)
    logger.info("Pipeline saved to %s", path)


def load_pipeline(path: Union[str, Path]) -> QSARPipeline:
    """Load a QSARPipeline from disk.

    Parameters
    ----------
    path : str or Path
        File path written by save_pipeline().

    Returns
    -------
    QSARPipeline

    Raises
    ------
    FileNotFoundError if path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {path}")

    pipeline = joblib.load(path)
    if not isinstance(pipeline, QSARPipeline):
        raise TypeError(
            f"Loaded object is {type(pipeline)}, expected QSARPipeline. "
            "The file may have been saved with an incompatible version."
        )

    logger.info("Pipeline loaded from %s", path)
    return pipeline
