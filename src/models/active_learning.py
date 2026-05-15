"""
Active learning module for compound prioritisation in TB drug discovery.

Active learning selects the most informative molecules from an unlabelled pool
for experimental testing.  By choosing compounds where the model is uncertain
rather than sampling at random, we can achieve the same predictive accuracy
with far fewer wet-lab assays — critical when each MIC measurement costs time
and resources.

Typical workflow
----------------
1. Start with a small labelled set (e.g. 50 confirmed actives/inactives).
2. Build a QSAR model on the labelled set.
3. Use an acquisition function to score all unlabelled compounds.
4. Send the top-N compounds to the lab for testing.
5. Add the new labels to the training set and repeat.

Example
-------
>>> from src.models.active_learning import ActiveLearner, UCB
>>> from src.pipeline.qsar_pipeline import build_pipeline
>>>
>>> model = build_pipeline(task='classification', n_estimators=200)
>>> model.fit(X_train_smiles, y_train)
>>>
>>> learner = ActiveLearner(
...     model=model.pipeline,    # pass the inner sklearn Pipeline
...     acquisition_fn=UCB(kappa=2.0),
...     X_pool=pool_smiles,
... )
>>> query_indices = learner.query(n_instances=10)
>>> # ... send compounds[query_indices] to biology team ...
>>> learner.teach(query_indices, new_labels)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Acquisition function interface
# ---------------------------------------------------------------------------

class AcquisitionFunction(ABC):
    """Abstract base class for all acquisition functions.

    Subclasses must implement ``score(model, X_pool)`` which returns a 1-D
    array of acquisition values — **higher = more worth querying**.

    Parameters
    ----------
    (None at base level; subclasses add their own hyperparameters.)
    """

    @abstractmethod
    def score(self, model, X_pool: Union[np.ndarray, Sequence]) -> np.ndarray:
        """Compute acquisition scores for every compound in the pool.

        Parameters
        ----------
        model : fitted sklearn estimator or QSARPipeline
            Must expose predict_proba (classifiers) or predict (regressors).
        X_pool : array-like of shape (n_pool,) or (n_pool, n_features)
            Unlabelled molecules — either SMILES strings or descriptor arrays
            depending on what ``model`` expects.

        Returns
        -------
        np.ndarray of shape (n_pool,)
            Score for each molecule.  Higher scores indicate higher priority.
        """

    def __repr__(self) -> str:  # pragma: no cover
        params = ", ".join(
            f"{k}={v!r}"
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        )
        return f"{self.__class__.__name__}({params})"


# ---------------------------------------------------------------------------
# Concrete acquisition functions
# ---------------------------------------------------------------------------

class UCB(AcquisitionFunction):
    """Upper Confidence Bound acquisition function.

    Balances exploration (high uncertainty) and exploitation (high predicted
    value) via:

        UCB(x) = mu(x) + kappa * sigma(x)

    where mu and sigma are estimated from the ensemble of RF trees.

    Parameters
    ----------
    kappa : float
        Exploration-exploitation trade-off parameter.  kappa=0 is pure
        exploitation; large kappa (e.g. 5.0) is near-pure exploration.
        Default 2.0 is a common starting point.

    Notes
    -----
    For a RandomForestClassifier, mu is the mean predicted probability
    across trees and sigma is its standard deviation.  For a regressor,
    mu and sigma are the mean and std of per-tree predictions.

    Example
    -------
    >>> ucb = UCB(kappa=2.5)
    >>> scores = ucb.score(model, X_pool)
    >>> top10 = np.argsort(scores)[-10:]
    """

    def __init__(self, kappa: float = 2.0) -> None:
        if kappa < 0:
            raise ValueError(f"kappa must be >= 0, got {kappa}")
        self.kappa = kappa

    def score(self, model, X_pool: Union[np.ndarray, Sequence]) -> np.ndarray:
        mu, sigma = _ensemble_mean_std(model, X_pool)
        return mu + self.kappa * sigma


class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement over the best observed value.

    The standard Gaussian EI formula:

        EI(x) = (mu(x) - y_best - xi) * Phi(Z) + sigma(x) * phi(Z)
        Z = (mu(x) - y_best - xi) / sigma(x)

    where Phi and phi are the CDF and PDF of the standard normal.

    Parameters
    ----------
    y_best : float
        The best (highest) activity value seen so far in the labelled set.
        Must be set before calling score(), or passed at construction.
    xi : float
        Small jitter that encourages exploration over pure exploitation.
        Default 0.01.

    Example
    -------
    >>> ei = ExpectedImprovement(y_best=0.85, xi=0.01)
    >>> scores = ei.score(model, X_pool)
    """

    def __init__(self, y_best: float = 0.0, xi: float = 0.01) -> None:
        self.y_best = y_best
        self.xi = xi

    def score(self, model, X_pool: Union[np.ndarray, Sequence]) -> np.ndarray:
        mu, sigma = _ensemble_mean_std(model, X_pool)

        # Avoid division by zero for deterministic predictions
        sigma = np.clip(sigma, 1e-9, None)

        z = (mu - self.y_best - self.xi) / sigma
        ei = (mu - self.y_best - self.xi) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.clip(ei, 0.0, None)  # EI is non-negative by definition
        return ei


class ThompsonSampling(AcquisitionFunction):
    """Thompson Sampling acquisition function.

    Draws one prediction from each tree in the Random Forest and uses the
    sample mean as the acquisition score.  Because each tree is a random
    sample from the posterior, this naturally balances exploration and
    exploitation without a free hyperparameter.

    Example
    -------
    >>> ts = ThompsonSampling(random_seed=0)
    >>> scores = ts.score(model, X_pool)
    """

    def __init__(self, random_seed: Optional[int] = None) -> None:
        self.random_seed = random_seed

    def score(self, model, X_pool: Union[np.ndarray, Sequence]) -> np.ndarray:
        rng = np.random.default_rng(self.random_seed)
        estimator = _get_estimator(model)

        X_arr = _to_array(model, X_pool)

        if not hasattr(estimator, "estimators_"):
            # Fallback: treat as single model
            mu, sigma = _ensemble_mean_std(model, X_pool)
            return rng.normal(mu, sigma + 1e-9)

        trees = estimator.estimators_
        # Sample one tree uniformly at random for each compound
        sampled_tree_idx = rng.integers(0, len(trees), size=len(X_arr))

        scores = np.zeros(len(X_arr))
        for i, tree_idx in enumerate(sampled_tree_idx):
            tree = trees[tree_idx]
            x_i = X_arr[[i]]
            if hasattr(tree, "predict_proba"):
                scores[i] = tree.predict_proba(x_i)[0, 1]
            else:
                scores[i] = tree.predict(x_i)[0]

        return scores


class MaxEntropy(AcquisitionFunction):
    """Maximum (predictive) entropy acquisition function.

    For classifiers, maximises:

        H[y|x] = -sum_c p(y=c|x) * log(p(y=c|x) + eps)

    Selecting the compound where the model is most uncertain (closest to
    uniform class distribution).  This is equivalent to maximising the
    information gain about the model parameters.

    Parameters
    ----------
    eps : float
        Small value to avoid log(0).  Default 1e-10.

    Example
    -------
    >>> me = MaxEntropy()
    >>> scores = me.score(classifier, X_pool)
    """

    def __init__(self, eps: float = 1e-10) -> None:
        self.eps = eps

    def score(self, model, X_pool: Union[np.ndarray, Sequence]) -> np.ndarray:
        estimator = _get_estimator(model)
        X_arr = _to_array(model, X_pool)

        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X_arr)  # (n, n_classes)
        else:
            # Regression fallback: use normalised uncertainty
            mu, sigma = _ensemble_mean_std(model, X_pool)
            return sigma

        proba = np.clip(proba, self.eps, 1.0)
        entropy = -np.sum(proba * np.log(proba), axis=1)
        return entropy


# ---------------------------------------------------------------------------
# Active Learner
# ---------------------------------------------------------------------------

class ActiveLearner:
    """Orchestrates the active learning loop for compound prioritisation.

    Maintains a labelled training set and an unlabelled pool.  At each
    iteration, query() selects the most informative compounds and teach()
    incorporates their labels into the model.

    Parameters
    ----------
    model : sklearn estimator
        A fitted (or unfitted) sklearn-compatible model.  Can be a plain
        RandomForestClassifier, a sklearn Pipeline, or a QSARPipeline
        (pass ``qsar_pipeline.pipeline`` for the inner sklearn Pipeline).
    acquisition_fn : AcquisitionFunction
        Determines how pool compounds are scored.
    X_pool : array-like
        Unlabelled molecules (SMILES strings or descriptor arrays).
    y_pool : array-like or None
        Ground-truth labels for pool compounds.  If provided, query() can
        be used in simulation mode where labels are automatically revealed.
    X_train : array-like or None
        Initial labelled training data.
    y_train : array-like or None
        Initial training labels.

    Examples
    --------
    Simulation experiment (all labels known, mimic active learning loop):

    >>> learner = ActiveLearner(
    ...     model=RandomForestClassifier(n_estimators=100),
    ...     acquisition_fn=UCB(kappa=2.0),
    ...     X_pool=pool_descriptors,
    ...     y_pool=pool_labels,
    ...     X_train=initial_train_descriptors,
    ...     y_train=initial_train_labels,
    ... )
    >>> for cycle in range(10):
    ...     indices = learner.query(n_instances=10)
    ...     labels = pool_labels[indices]           # simulated lab result
    ...     learner.teach(indices, labels)
    ...     print(f"Cycle {cycle}: pool size = {len(learner.X_pool)}")

    Real deployment (labels arrive from the biology team):

    >>> indices = learner.query(n_instances=5)
    >>> smiles_to_test = [pool_smiles[i] for i in indices]
    >>> # ... wait for MIC measurements ...
    >>> learner.teach(indices, measured_labels)
    """

    def __init__(
        self,
        model,
        acquisition_fn: AcquisitionFunction,
        X_pool: Union[np.ndarray, List],
        y_pool: Optional[Union[np.ndarray, List]] = None,
        X_train: Optional[Union[np.ndarray, List]] = None,
        y_train: Optional[Union[np.ndarray, List]] = None,
    ) -> None:
        self.model = model
        self.acquisition_fn = acquisition_fn

        self.X_pool: List = list(X_pool)
        self.y_pool: Optional[np.ndarray] = (
            np.asarray(y_pool) if y_pool is not None else None
        )
        self.X_train: List = list(X_train) if X_train is not None else []
        self.y_train: List = list(y_train) if y_train is not None else []

        # Fit model if initial training data provided
        if self.X_train and self.y_train:
            logger.info(
                "Fitting initial model on %d labelled samples", len(self.X_train)
            )
            self._fit_model()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def query(self, n_instances: int = 10) -> np.ndarray:
        """Select the most informative compounds from the unlabelled pool.

        Parameters
        ----------
        n_instances : int
            Number of compounds to select.

        Returns
        -------
        np.ndarray of int
            Indices into the current X_pool.

        Raises
        ------
        RuntimeError if the model has not been fitted yet.
        ValueError if n_instances > len(X_pool).
        """
        if not self.X_pool:
            raise ValueError("Pool is empty — no compounds left to query.")
        if n_instances > len(self.X_pool):
            logger.warning(
                "n_instances=%d > pool size=%d; returning entire pool.",
                n_instances,
                len(self.X_pool),
            )
            n_instances = len(self.X_pool)

        scores = self.acquisition_fn.score(self.model, self.X_pool)
        top_indices = np.argsort(scores)[-n_instances:][::-1]
        logger.info(
            "Queried %d compounds from pool of %d (top score: %.4f)",
            n_instances,
            len(self.X_pool),
            scores[top_indices[0]],
        )
        return top_indices

    def teach(
        self,
        query_indices: Union[np.ndarray, List[int]],
        y_new: Union[np.ndarray, List],
    ) -> None:
        """Add newly labelled compounds to the training set and refit the model.

        Compounds at ``query_indices`` are moved from X_pool to X_train.

        Parameters
        ----------
        query_indices : array-like of int
            Indices returned by query().
        y_new : array-like
            Labels for the queried compounds, in the same order as
            query_indices.

        Side effects
        ------------
        - X_pool and y_pool are updated (queried items removed).
        - X_train and y_train are extended with the new data.
        - The internal model is re-fitted.
        """
        query_indices = np.asarray(query_indices)
        y_new = np.asarray(y_new)

        if len(query_indices) != len(y_new):
            raise ValueError(
                f"query_indices length ({len(query_indices)}) must match "
                f"y_new length ({len(y_new)})."
            )

        # Move compounds from pool to labelled set
        new_X = [self.X_pool[i] for i in query_indices]
        self.X_train.extend(new_X)
        self.y_train.extend(y_new.tolist())

        # Remove from pool (reverse order to preserve indices)
        for i in sorted(query_indices, reverse=True):
            del self.X_pool[i]
            if self.y_pool is not None:
                self.y_pool = np.delete(self.y_pool, i)

        logger.info(
            "teach(): added %d compounds → training set now %d, pool now %d",
            len(query_indices),
            len(self.X_train),
            len(self.X_pool),
        )
        self._fit_model()

    def get_uncertainty_scores(
        self, X_pool: Optional[Union[np.ndarray, List]] = None
    ) -> np.ndarray:
        """Return per-molecule uncertainty (std of ensemble predictions).

        Parameters
        ----------
        X_pool : array-like or None
            Molecules to score.  Uses ``self.X_pool`` if None.

        Returns
        -------
        np.ndarray of shape (n_pool,)
            Standard deviation of ensemble predictions.  Higher = more uncertain.
        """
        pool = X_pool if X_pool is not None else self.X_pool
        _, sigma = _ensemble_mean_std(self.model, pool)
        return sigma

    def simulate(
        self,
        n_cycles: int = 10,
        n_instances_per_cycle: int = 10,
    ) -> List[dict]:
        """Run a full active learning simulation (requires y_pool to be set).

        Parameters
        ----------
        n_cycles : int
            Number of query → teach iterations.
        n_instances_per_cycle : int
            Compounds queried per cycle.

        Returns
        -------
        list of dict
            One entry per cycle with keys: cycle, n_train, n_pool, queried_indices.
        """
        if self.y_pool is None:
            raise ValueError(
                "simulate() requires y_pool to be provided at construction — "
                "ground-truth labels are needed to automatically assign labels."
            )

        history = []
        for cycle in range(n_cycles):
            if not self.X_pool:
                logger.info("Pool exhausted at cycle %d.", cycle)
                break

            indices = self.query(n_instances=n_instances_per_cycle)
            labels = np.asarray([self.y_pool[i] for i in indices])
            self.teach(indices, labels)

            history.append(
                {
                    "cycle": cycle + 1,
                    "n_train": len(self.X_train),
                    "n_pool": len(self.X_pool),
                    "queried_indices": indices.tolist(),
                }
            )

        return history

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_model(self) -> None:
        """Fit (or re-fit) the internal model on all labelled data."""
        X = np.asarray(self.X_train) if not isinstance(self.X_train[0], str) else self.X_train
        y = np.asarray(self.y_train)
        self.model.fit(X, y)
        logger.debug("Model re-fitted on %d samples.", len(y))


# ---------------------------------------------------------------------------
# Internal utility functions
# ---------------------------------------------------------------------------

def _get_estimator(model):
    """Unwrap the final estimator from a sklearn Pipeline or return as-is."""
    if hasattr(model, "named_steps"):
        # sklearn Pipeline
        steps = list(model.named_steps.values())
        return steps[-1]
    if hasattr(model, "pipeline"):
        # QSARPipeline wrapper
        steps = list(model.pipeline.named_steps.values())
        return steps[-1]
    return model


def _to_array(model, X_pool):
    """Transform X_pool through all pipeline steps except the final estimator."""
    if hasattr(model, "named_steps"):
        # sklearn Pipeline: transform through all-but-last steps
        steps = list(model.named_steps.items())
        X = X_pool
        for name, step in steps[:-1]:
            X = step.transform(X)
        return np.asarray(X)
    if hasattr(model, "pipeline"):
        return _to_array(model.pipeline, X_pool)
    # Plain estimator: X is already in the right form
    return np.asarray(X_pool)


def _ensemble_mean_std(
    model, X_pool: Union[np.ndarray, Sequence]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-compound mean and std across a Random Forest's trees.

    For classifiers, returns (mean_proba_class1, std_proba_class1).
    For regressors, returns (mean_prediction, std_prediction).

    Parameters
    ----------
    model : fitted sklearn estimator / Pipeline
    X_pool : array-like
        Unlabelled molecules.

    Returns
    -------
    mu : np.ndarray of shape (n_pool,)
    sigma : np.ndarray of shape (n_pool,)
    """
    estimator = _get_estimator(model)
    X_arr = _to_array(model, X_pool)

    if hasattr(estimator, "estimators_"):
        trees = estimator.estimators_
        is_classifier = hasattr(estimator, "classes_")

        preds = []
        for tree in trees:
            if is_classifier and hasattr(tree, "predict_proba"):
                preds.append(tree.predict_proba(X_arr)[:, 1])
            else:
                preds.append(tree.predict(X_arr))

        preds_arr = np.array(preds)  # shape: (n_trees, n_pool)
        mu = preds_arr.mean(axis=0)
        sigma = preds_arr.std(axis=0)
        return mu, sigma

    # Fallback for non-ensemble models: use predict mean, sigma=0
    if hasattr(estimator, "predict_proba"):
        mu = estimator.predict_proba(X_arr)[:, 1]
    else:
        mu = estimator.predict(X_arr).astype(float)
    sigma = np.zeros_like(mu)
    logger.warning(
        "Model %s is not an ensemble; uncertainty estimates will be zero.",
        type(estimator).__name__,
    )
    return mu, sigma
