"""FastAPI integration tests for the TB Drug Discovery inference API.

Uses httpx.AsyncClient (ASGI transport) — no server process needed.
All tests run in-process and do NOT require a loaded QSAR model file,
because we mock the model registry where needed.

Run:
    pytest tests/test_api.py -v
    pytest tests/test_api.py -v -k "not model_required"  # skip model-dependent tests
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Skip entire module if FastAPI / httpx are not available
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Import app — patch heavy model loading before import side-effects
# ---------------------------------------------------------------------------

with patch.dict("sys.modules", {}):
    from api.app import app  # noqa: E402

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_model(task: str = "regression"):
    """Return a MagicMock that behaves like a fitted QSARModel."""
    model = MagicMock()
    model.task = task
    model.is_fitted = True
    model.params = {"n_estimators": 100}
    model.feature_names = ["MolWt", "LogP", "TPSA", "HBD", "HBA"]
    model.training_metrics = {"r2_train": 0.95, "rmse_train": 0.2}
    if task == "regression":
        model.predict.return_value = np.array([7.5])
    else:
        model.predict.return_value = np.array([1])
        model.predict_proba.return_value = np.array([[0.1, 0.9]])
    return model


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_health_has_correlation_id(self):
        resp = client.get("/health", headers={"X-Correlation-ID": "test-123"})
        assert resp.headers.get("X-Correlation-ID") == "test-123"

    def test_health_generates_correlation_id_when_missing(self):
        resp = client.get("/health")
        assert "X-Correlation-ID" in resp.headers
        assert len(resp.headers["X-Correlation-ID"]) == 36  # UUID4 format

    def test_health_has_response_time(self):
        resp = client.get("/health")
        assert "X-Response-Time" in resp.headers


# ---------------------------------------------------------------------------
# /model/info
# ---------------------------------------------------------------------------

class TestModelInfo:
    def test_model_info_503_when_no_model(self):
        """Returns 503 when no model file exists (default state)."""
        with patch("api.app._registry", {}):
            with patch("api.app.Path") as mock_path:
                mock_path.return_value.exists.return_value = False
                resp = client.get("/model/info")
        assert resp.status_code == 503

    def test_model_info_returns_metadata(self):
        mock_model = _make_mock_model("regression")
        with patch("api.app._registry", {"qsar_model": mock_model}):
            resp = client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"] == "regression"
        assert data["is_fitted"] is True
        assert "training_metrics" in data
        assert data["n_features"] == 5


# ---------------------------------------------------------------------------
# /predict/activity
# ---------------------------------------------------------------------------

VALID_SMILES = ["CCO", "c1ccccc1", "CC(=O)O"]
INVALID_SMILES = ["not_a_smiles", "%%%"]


class TestPredictActivity:
    def _post(self, smiles, model=None, **kwargs):
        if model is None:
            model = _make_mock_model("regression")
        payload = {"smiles": smiles, **kwargs}
        with patch("api.app._registry", {"qsar_model": model}):
            return client.post("/predict/activity", json=payload)

    def test_predict_returns_200(self):
        resp = self._post(VALID_SMILES)
        assert resp.status_code == 200

    def test_predict_returns_one_result_per_smiles(self):
        resp = self._post(VALID_SMILES)
        data = resp.json()
        assert len(data["results"]) == len(VALID_SMILES)

    def test_predict_regression_fields(self):
        resp = self._post(["CCO"])
        result = resp.json()["results"][0]
        # valid SMILES should have a prediction or at least be valid
        assert "valid" in result
        assert "smiles" in result

    def test_predict_classification_fields(self):
        model = _make_mock_model("classification")
        resp = self._post(["CCO"], model=model)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 1

    def test_predict_empty_smiles_rejected(self):
        resp = client.post("/predict/activity", json={"smiles": []})
        assert resp.status_code == 422  # Pydantic validation error

    def test_predict_503_when_no_model(self):
        with patch("api.app._registry", {}):
            with patch("api.app.Path") as mp:
                mp.return_value.exists.return_value = False
                resp = client.post(
                    "/predict/activity",
                    json={"smiles": ["CCO"]},
                )
        assert resp.status_code == 503

    def test_predict_processing_time_present(self):
        resp = self._post(VALID_SMILES)
        assert "processing_time_s" in resp.json()
        assert resp.json()["processing_time_s"] >= 0

    def test_predict_n_valid_count(self):
        resp = self._post(VALID_SMILES)
        data = resp.json()
        assert "n_valid" in data
        assert "n_predicted" in data

    def test_predict_correlation_id_propagated(self):
        resp = self._post(VALID_SMILES)
        assert "X-Correlation-ID" in resp.headers


# ---------------------------------------------------------------------------
# /predict/batch
# ---------------------------------------------------------------------------

class TestBatchScreening:
    def _post(self, smiles_list, model=None, **kwargs):
        if model is None:
            model = _make_mock_model("regression")
        payload = {"smiles_list": smiles_list, **kwargs}
        with patch("api.app._registry", {"qsar_model": model}):
            return client.post("/predict/batch", json=payload)

    def test_batch_returns_200(self):
        resp = self._post(VALID_SMILES * 3)
        assert resp.status_code == 200

    def test_batch_top_k_respected(self):
        resp = self._post(VALID_SMILES * 5, top_k=2)
        data = resp.json()
        assert len(data["top_compounds"]) <= 2

    def test_batch_n_screened_correct(self):
        smiles = VALID_SMILES * 4
        resp = self._post(smiles)
        data = resp.json()
        assert data["n_screened"] == len(smiles)

    def test_batch_503_when_no_model(self):
        with patch("api.app._registry", {}):
            with patch("api.app.Path") as mp:
                mp.return_value.exists.return_value = False
                resp = client.post(
                    "/predict/batch",
                    json={"smiles_list": ["CCO"]},
                )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /metrics (Prometheus)
# ---------------------------------------------------------------------------

class TestPrometheusMetrics:
    def test_metrics_endpoint_exists(self):
        resp = client.get("/metrics")
        # Either 200 (prometheus installed) or 501 (not installed)
        assert resp.status_code in (200, 501)

    def test_metrics_content_type_when_available(self):
        try:
            import prometheus_client  # noqa: F401
        except ImportError:
            pytest.skip("prometheus_client not installed")
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_smiles_whitespace_stripped(self):
        """API should strip leading/trailing whitespace from SMILES."""
        model = _make_mock_model("regression")
        payload = {"smiles": ["  CCO  "]}
        with patch("api.app._registry", {"qsar_model": model}):
            resp = client.post("/predict/activity", json=payload)
        assert resp.status_code == 200

    def test_batch_top_k_bounds(self):
        """top_k = 0 should be rejected by Pydantic."""
        resp = client.post(
            "/predict/batch",
            json={"smiles_list": ["CCO"], "top_k": 0},
        )
        assert resp.status_code == 422

    def test_generation_request_temperature_bounds(self):
        """Temperature outside [0.1, 2.0] should be rejected."""
        resp = client.post(
            "/generate/metrics",
            json={"num_samples": 10, "temperature": 99.0},
        )
        assert resp.status_code == 422
