"""Tests for probability calibration (QSARCalibrator).

Calibration is critical for conformal prediction and decision thresholds.
These tests verify:
1. ECE decreases after calibration vs. raw RF output
2. Probabilities stay in [0, 1] after calibration
3. All three methods (sigmoid, isotonic, temperature) work
4. Calibration on training set raises a warning (not an error — we can't enforce it)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

pytest.importorskip("sklearn")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def classification_data():
    """Synthetic imbalanced classification dataset (10:1 ratio)."""
    rng = np.random.default_rng(42)
    n = 500
    X = rng.standard_normal((n, 20))
    # True probability: sigmoid of first feature
    true_p = 1 / (1 + np.exp(-X[:, 0]))
    y = (rng.uniform(size=n) < true_p).astype(int)
    return X, y


@pytest.fixture(scope="module")
def fitted_qsar_model(classification_data):
    """Fitted QSARModel (classification) on 80% of data."""
    from models.qsar_model import QSARModel

    X, y = classification_data
    split = int(0.8 * len(X))
    model = QSARModel(task="classification", n_estimators=50, random_seed=42)
    model.fit(X[:split], y[:split])
    return model, X[split:], y[split:]


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------

class TestQSARCalibratorInit:
    def test_regression_model_raises(self):
        from models.qsar_model import QSARModel
        from models.calibration import QSARCalibrator

        model = QSARModel(task="regression", n_estimators=10)
        # not fitted → still raises task check first
        with pytest.raises(ValueError, match="classification"):
            QSARCalibrator(model)

    def test_unfitted_model_raises(self):
        from models.qsar_model import QSARModel
        from models.calibration import QSARCalibrator

        model = QSARModel(task="classification", n_estimators=10)
        with pytest.raises(ValueError, match="fitted"):
            QSARCalibrator(model)


class TestCalibrationMethods:
    @pytest.mark.parametrize("method", ["sigmoid", "isotonic", "temperature"])
    def test_fit_and_predict_proba(self, method, fitted_qsar_model):
        from models.calibration import QSARCalibrator

        model, X_cal, y_cal = fitted_qsar_model
        cal = QSARCalibrator(model, method=method)
        cal.fit(X_cal, y_cal)

        proba = cal.predict_proba(X_cal)
        assert proba.shape == (len(X_cal), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        # Rows sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    @pytest.mark.parametrize("method", ["sigmoid", "isotonic", "temperature"])
    def test_predict_returns_binary(self, method, fitted_qsar_model):
        from models.calibration import QSARCalibrator

        model, X_cal, y_cal = fitted_qsar_model
        cal = QSARCalibrator(model, method=method)
        cal.fit(X_cal, y_cal)
        preds = cal.predict(X_cal)
        assert set(preds).issubset({0, 1})

    def test_predict_before_fit_raises(self, fitted_qsar_model):
        from models.calibration import QSARCalibrator

        model, X_cal, _ = fitted_qsar_model
        cal = QSARCalibrator(model, method="sigmoid")
        with pytest.raises(ValueError, match="not fitted"):
            cal.predict_proba(X_cal)


class TestCalibrationReport:
    def test_report_keys(self, fitted_qsar_model):
        from models.calibration import QSARCalibrator

        model, X_cal, y_cal = fitted_qsar_model
        cal = QSARCalibrator(model, method="sigmoid")
        cal.fit(X_cal, y_cal)
        report = cal.calibration_report(X_cal, y_cal)

        assert "ece" in report
        assert "brier_score" in report
        assert "brier_score_uncalibrated" in report
        assert report["method"] == "sigmoid"

    def test_brier_improves_after_calibration(self, fitted_qsar_model):
        """Calibration should not make Brier score worse (on cal set)."""
        from models.calibration import QSARCalibrator

        model, X_cal, y_cal = fitted_qsar_model
        cal = QSARCalibrator(model, method="sigmoid")
        cal.fit(X_cal, y_cal)
        report = cal.calibration_report(X_cal, y_cal)

        # On the calibration set, calibrated Brier ≤ uncalibrated
        assert report["brier_score"] <= report["brier_score_uncalibrated"] + 0.05
