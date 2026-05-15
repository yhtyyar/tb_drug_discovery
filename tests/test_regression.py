"""Regression tests: guard against metric degradation during refactoring.

These tests train on a fixed synthetic dataset and assert that key
metrics stay above known-good baselines.  If a refactor breaks the
model (e.g., wrong feature scaling, broken CV), these tests catch it.

Update baselines: pytest tests/test_regression.py --update-baselines
Read baselines from: tests/baselines/regression_baselines.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

BASELINES_FILE = Path(__file__).parent / "baselines" / "regression_baselines.json"

# Hard-coded tolerances — allow 5% degradation before failing
TOLERANCE = 0.05


def load_baselines() -> dict:
    if BASELINES_FILE.exists():
        with open(BASELINES_FILE) as f:
            return json.load(f)
    return {}


def save_baselines(data: dict) -> None:
    BASELINES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINES_FILE, "w") as f:
        json.dump(data, f, indent=2)


def pytest_addoption(parser):
    """Add --update-baselines flag."""
    try:
        parser.addoption(
            "--update-baselines",
            action="store_true",
            default=False,
            help="Overwrite regression baselines with current results",
        )
    except ValueError:
        pass  # already added by another conftest


# ---------------------------------------------------------------------------
# Synthetic dataset (fixed seed → deterministic)
# ---------------------------------------------------------------------------

def make_regression_dataset(n=300, n_features=50, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = X[:, 0] * 2.5 + X[:, 1] * 1.2 - X[:, 2] * 0.8 + rng.randn(n) * 0.3
    return X, y


def make_classification_dataset(n=300, n_features=50, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Regression model baselines
# ---------------------------------------------------------------------------

class TestQSARRegressionBaseline:
    """QSAR regression must maintain R² and RMSE within tolerance."""

    @pytest.fixture(scope="class")
    def trained_model(self):
        from models.qsar_model import QSARModel
        X, y = make_regression_dataset()
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = QSARModel(task="regression", n_estimators=100, random_seed=42)
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        return metrics

    def test_r2_above_baseline(self, trained_model, request):
        baselines = load_baselines()
        key = "qsar_regression_r2"

        if request.config.getoption("--update-baselines", default=False):
            baselines[key] = trained_model["r2"]
            save_baselines(baselines)
            pytest.skip(f"Updated baseline: {key}={trained_model['r2']:.4f}")

        baseline = baselines.get(key, 0.85)
        assert trained_model["r2"] >= baseline - TOLERANCE, (
            f"R² regressed: {trained_model['r2']:.4f} < {baseline - TOLERANCE:.4f}"
        )

    def test_rmse_below_baseline(self, trained_model, request):
        baselines = load_baselines()
        key = "qsar_regression_rmse"

        if request.config.getoption("--update-baselines", default=False):
            baselines[key] = trained_model["rmse"]
            save_baselines(baselines)
            pytest.skip(f"Updated baseline: {key}={trained_model['rmse']:.4f}")

        baseline = baselines.get(key, 0.5)
        assert trained_model["rmse"] <= baseline * (1 + TOLERANCE), (
            f"RMSE regressed: {trained_model['rmse']:.4f} > {baseline * (1 + TOLERANCE):.4f}"
        )


class TestQSARClassificationBaseline:
    """QSAR classification must maintain ROC-AUC within tolerance."""

    @pytest.fixture(scope="class")
    def trained_model(self):
        from models.qsar_model import QSARModel
        X, y = make_classification_dataset()
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = QSARModel(task="classification", n_estimators=100, random_seed=42)
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        return metrics

    def test_roc_auc_above_baseline(self, trained_model, request):
        baselines = load_baselines()
        key = "qsar_classification_roc_auc"

        if request.config.getoption("--update-baselines", default=False):
            baselines[key] = trained_model["roc_auc"]
            save_baselines(baselines)
            pytest.skip(f"Updated baseline: {key}={trained_model['roc_auc']:.4f}")

        baseline = baselines.get(key, 0.90)
        assert trained_model["roc_auc"] >= baseline - TOLERANCE, (
            f"ROC-AUC regressed: {trained_model['roc_auc']:.4f} < {baseline - TOLERANCE:.4f}"
        )


# ---------------------------------------------------------------------------
# Cross-validation stability
# ---------------------------------------------------------------------------

class TestCVStability:
    """CV std must remain low — high variance signals overfitting or data issues."""

    def test_regression_cv_std_stable(self):
        from evaluation.cross_validation import cross_validate_model
        from sklearn.ensemble import RandomForestRegressor

        X, y = make_regression_dataset()
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        results = cross_validate_model(model, X, y, n_folds=5, task="regression")

        assert results["r2_std"] < 0.15, (
            f"CV R² std too high: {results['r2_std']:.3f} — possible data leakage or instability"
        )

    def test_classification_cv_std_stable(self):
        from evaluation.cross_validation import cross_validate_model
        from sklearn.ensemble import RandomForestClassifier

        X, y = make_classification_dataset()
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        results = cross_validate_model(model, X, y, n_folds=5, task="classification")

        assert results.get("roc_auc_std", 0) < 0.15, (
            f"CV ROC-AUC std too high: {results.get('roc_auc_std', 0):.3f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
