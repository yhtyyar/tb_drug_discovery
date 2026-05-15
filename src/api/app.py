"""FastAPI inference service for TB Drug Discovery pipeline.

Endpoints:
    POST /predict/activity     — QSAR activity prediction for 1-N molecules
    POST /predict/batch        — Batch SMILES file for virtual screening
    POST /generate/molecules   — Generate novel molecules with quality metrics
    GET  /health               — Liveness probe
    GET  /model/info           — Loaded model metadata

Usage:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

Clients (example with curl):
    curl -X POST http://localhost:8000/predict/activity \\
         -H "Content-Type: application/json" \\
         -d '{"smiles": ["CCO", "c1ccccc1"]}'
"""

import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, validator
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    raise ImportError("FastAPI required: pip install fastapi uvicorn")

import numpy as np
from loguru import logger

# Prometheus metrics (optional — graceful degradation if not installed)
try:
    from prometheus_client import (
        Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST,
    )
    _req_counter = Counter(
        "tb_api_requests_total",
        "Total API requests",
        ["method", "endpoint", "status"],
    )
    _req_latency = Histogram(
        "tb_api_request_latency_seconds",
        "Request latency in seconds",
        ["endpoint"],
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0),
    )
    _predictions_total = Counter("tb_api_predictions_total", "Total SMILES predictions made")
    _invalid_smiles = Counter("tb_api_invalid_smiles_total", "SMILES that failed validation")
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from data.chembl_loader import ChEMBLLoader
from data.descriptor_calculator import DescriptorCalculator
from models.qsar_model import QSARModel
from evaluation.generation_metrics import evaluate_generation
from evaluation.drift_detector import DescriptorDriftDetector

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TB Drug Discovery API",
    description="QSAR prediction and molecular generation for M. tuberculosis targets",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Attach X-Correlation-ID to every request/response for tracing."""
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    t0 = time.time()

    with logger.contextualize(correlation_id=correlation_id):
        response = await call_next(request)

    latency = time.time() - t0
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Response-Time"] = f"{latency:.4f}s"

    if PROMETHEUS_AVAILABLE:
        endpoint = request.url.path
        _req_counter.labels(
            method=request.method,
            endpoint=endpoint,
            status=str(response.status_code),
        ).inc()
        _req_latency.labels(endpoint=endpoint).observe(latency)

    return response


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Prometheus scrape endpoint — used by Grafana dashboards."""
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(status_code=501, detail="prometheus_client not installed")
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ---------------------------------------------------------------------------
# Global model registry (lazy loading)
# ---------------------------------------------------------------------------

_registry: Dict[str, object] = {}
_loader = ChEMBLLoader.__new__(ChEMBLLoader)
_loader.target_id = "CHEMBL1849"
_loader.min_compounds = 0
_loader.random_seed = 42


def get_descriptor_calculator() -> DescriptorCalculator:
    if "descriptor_calc" not in _registry:
        _registry["descriptor_calc"] = DescriptorCalculator(
            lipinski=True, topological=True, extended=True
        )
    return _registry["descriptor_calc"]  # type: ignore


def get_qsar_model(model_path: str = "models/qsar_model.pkl") -> Optional[QSARModel]:
    if "qsar_model" not in _registry:
        p = Path(model_path)
        if p.exists():
            _registry["qsar_model"] = QSARModel.load(str(p))
            logger.info(f"QSAR model loaded from {p}")
        else:
            logger.warning(f"No QSAR model at {p} — predictions will fail")
            return None
    return _registry["qsar_model"]  # type: ignore


def _background_drift_check(feature_matrix: np.ndarray) -> None:
    """Run KS-test drift detection in background after each prediction batch.

    Compares incoming descriptor distribution against a reference stored at
    models/drift_reference.npy. Logs a WARNING if drift is detected.
    Designed to be non-blocking — called via FastAPI BackgroundTasks.
    """
    ref_path = Path("models/drift_reference.npy")
    if not ref_path.exists():
        return

    try:
        ref_data = np.load(str(ref_path))
        detector = DescriptorDriftDetector(significance_level=0.01)
        detector.fit(ref_data)
        result = detector.detect(feature_matrix)

        if result.get("drift_detected"):
            n_drifted = result.get("n_drifted_features", 0)
            logger.warning(
                f"[DRIFT] {n_drifted} features drifted vs reference "
                f"(p<0.01, batch_size={len(feature_matrix)}). "
                "Model predictions may be unreliable — consider retraining."
            )
        else:
            logger.debug(f"[DRIFT] No drift detected (batch_size={len(feature_matrix)})")
    except Exception as e:
        logger.debug(f"[DRIFT] Background check failed: {e}")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

_MAX_SMILES_LEN = 500  # longest drug-like molecule rarely exceeds 200 chars


class ActivityRequest(BaseModel):
    smiles: List[str] = Field(..., min_items=1, max_items=10_000, description="List of SMILES")
    model_path: str = Field("models/qsar_model.pkl", description="Path to trained QSAR model")

    @validator("smiles", each_item=True)
    def validate_smiles_string(cls, v):
        if not v or not v.strip():
            raise ValueError("SMILES must not be empty")
        v = v.strip()
        if len(v) > _MAX_SMILES_LEN:
            raise ValueError(
                f"SMILES too long ({len(v)} chars > {_MAX_SMILES_LEN}). "
                "Likely invalid or an injection attempt."
            )
        return v


class MoleculeResult(BaseModel):
    smiles: str
    valid: bool
    canonical_smiles: Optional[str]
    predicted_pic50: Optional[float]
    predicted_active: Optional[bool]
    probability_active: Optional[float]
    error: Optional[str]


class ActivityResponse(BaseModel):
    results: List[MoleculeResult]
    n_valid: int
    n_predicted: int
    processing_time_s: float


class GenerationRequest(BaseModel):
    num_samples: int = Field(100, ge=1, le=5000, description="Number of molecules to generate")
    training_smiles: Optional[List[str]] = Field(None, description="Training SMILES for novelty calc")
    temperature: float = Field(1.0, ge=0.1, le=2.0)


class GenerationResponse(BaseModel):
    metrics: Dict[str, float]
    n_generated: int
    processing_time_s: float


class BatchScreeningRequest(BaseModel):
    smiles_list: List[str] = Field(..., min_items=1, max_items=100_000)
    activity_threshold: float = Field(7.0, description="pIC50 threshold for 'active' label")
    top_k: int = Field(100, ge=1, le=10_000, description="Return top-K predicted compounds")


class BatchScreeningResponse(BaseModel):
    top_compounds: List[MoleculeResult]
    n_screened: int
    n_valid: int
    n_predicted_active: int
    processing_time_s: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smiles_to_features(smiles: str) -> Optional[np.ndarray]:
    """Compute descriptor vector for one SMILES string."""
    calc = get_descriptor_calculator()
    result = calc.calculate(smiles)
    if result is None:
        return None
    # Align feature order with trained model
    model = get_qsar_model()
    if model is not None and model.feature_names:
        feats = [result.get(name, 0.0) for name in model.feature_names]
    else:
        feats = list(result.values())
    arr = np.array(feats, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr.reshape(1, -1)


def _predict_one(smiles: str, model: QSARModel) -> MoleculeResult:
    canonical = _loader.standardize_smiles(smiles)
    if canonical is None:
        if PROMETHEUS_AVAILABLE:
            _invalid_smiles.inc()
        return MoleculeResult(smiles=smiles, valid=False, canonical_smiles=None,
                               predicted_pic50=None, predicted_active=None,
                               probability_active=None, error="Invalid SMILES")

    features = _smiles_to_features(canonical)
    if features is None:
        return MoleculeResult(smiles=smiles, valid=True, canonical_smiles=canonical,
                               predicted_pic50=None, predicted_active=None,
                               probability_active=None, error="Descriptor calculation failed")

    try:
        if PROMETHEUS_AVAILABLE:
            _predictions_total.inc()
        if model.task == "regression":
            pic50 = float(model.predict(features)[0])
            active = pic50 >= 7.0
            return MoleculeResult(smiles=smiles, valid=True, canonical_smiles=canonical,
                                   predicted_pic50=round(pic50, 3), predicted_active=active,
                                   probability_active=None, error=None)
        else:
            pred = int(model.predict(features)[0])
            proba = float(model.predict_proba(features)[0, 1])
            return MoleculeResult(smiles=smiles, valid=True, canonical_smiles=canonical,
                                   predicted_pic50=None, predicted_active=bool(pred),
                                   probability_active=round(proba, 4), error=None)
    except Exception as e:
        return MoleculeResult(smiles=smiles, valid=True, canonical_smiles=canonical,
                               predicted_pic50=None, predicted_active=None,
                               probability_active=None, error=str(e))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/model/info")
async def model_info():
    model = get_qsar_model()
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    return {
        "task": model.task,
        "n_estimators": model.params.get("n_estimators"),
        "is_fitted": model.is_fitted,
        "n_features": len(model.feature_names or []),
        "training_metrics": model.training_metrics,
        # Reproducibility provenance
        "git_commit": getattr(model, "git_commit", "unknown"),
        "saved_at": getattr(model, "saved_at", "unknown"),
        "training_data_hash": getattr(model, "training_data_hash", None),
        "n_training_samples": getattr(model, "n_training_samples", None),
    }


@app.post("/predict/activity", response_model=ActivityResponse)
async def predict_activity(request: ActivityRequest, background_tasks: BackgroundTasks):
    t0 = time.time()
    model = get_qsar_model(request.model_path)
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not found: {request.model_path}")

    results = [_predict_one(smi, model) for smi in request.smiles]
    n_valid = sum(r.valid for r in results)
    n_predicted = sum(r.predicted_pic50 is not None or r.predicted_active is not None
                      for r in results)

    # Async drift detection — does not block the response
    valid_results = [r for r in results if r.valid and r.error is None]
    if valid_results and model.feature_names:
        feat_matrix = np.array([
            [_smiles_to_features(r.canonical_smiles or r.smiles)]
            for r in valid_results
            if _smiles_to_features(r.canonical_smiles or r.smiles) is not None
        ])
        if feat_matrix.ndim == 3:
            feat_matrix = feat_matrix.squeeze(1)
        if feat_matrix.size > 0:
            background_tasks.add_task(_background_drift_check, feat_matrix)

    return ActivityResponse(
        results=results,
        n_valid=n_valid,
        n_predicted=n_predicted,
        processing_time_s=round(time.time() - t0, 3),
    )


@app.post("/predict/batch", response_model=BatchScreeningResponse)
async def batch_screening(request: BatchScreeningRequest):
    """Virtual screening endpoint: returns top-K predicted actives."""
    t0 = time.time()
    model = get_qsar_model()
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    all_results = [_predict_one(smi, model) for smi in request.smiles_list]
    valid_results = [r for r in all_results if r.valid and r.error is None]

    # Sort by pIC50 (regression) or probability (classification)
    if model.task == "regression":
        scored = sorted(
            [r for r in valid_results if r.predicted_pic50 is not None],
            key=lambda r: r.predicted_pic50 or 0,
            reverse=True,
        )
    else:
        scored = sorted(
            [r for r in valid_results if r.probability_active is not None],
            key=lambda r: r.probability_active or 0,
            reverse=True,
        )

    top_k = scored[:request.top_k]
    n_predicted_active = sum(
        1 for r in valid_results
        if (r.predicted_pic50 or 0) >= request.activity_threshold
        or r.predicted_active is True
    )

    return BatchScreeningResponse(
        top_compounds=top_k,
        n_screened=len(request.smiles_list),
        n_valid=len(valid_results),
        n_predicted_active=n_predicted_active,
        processing_time_s=round(time.time() - t0, 3),
    )


@app.post("/generate/metrics", response_model=GenerationResponse)
async def generation_metrics_endpoint(request: GenerationRequest):
    """Evaluate quality metrics for a list of generated SMILES.

    Send your generated SMILES and get back validity, novelty, QED, etc.
    Does NOT run the generative model — you provide the output.
    """
    t0 = time.time()
    # Placeholder: in production the generative model would run here
    # For now, accept generated SMILES via the training_smiles field and return metrics
    raise HTTPException(
        status_code=501,
        detail="Pass generated SMILES directly; use evaluate_generation() in Python code."
    )


# ---------------------------------------------------------------------------
# ADMET prediction endpoint
# ---------------------------------------------------------------------------

class ADMETRequest(BaseModel):
    smiles: List[str] = Field(..., min_length=1, max_length=100, description="List of SMILES")

    @validator("smiles", each_item=True)
    def validate_smiles_len(cls, v: str) -> str:
        if len(v) > 500:
            raise ValueError(f"SMILES too long ({len(v)} > 500 chars)")
        return v


class ADMETEndpointResult(BaseModel):
    smiles: str
    solubility_log_s: float
    bbb_penetration: float
    herg_inhibition: float
    ames_mutagenicity: float
    hepatotoxicity: float
    oral_bioavailability: float
    qed: float
    lipinski_pass: bool
    admet_score: float
    concerns: List[str]


class ADMETResponse(BaseModel):
    results: List[ADMETEndpointResult]
    processing_time_s: float


@app.post("/predict/admet", response_model=ADMETResponse, tags=["ADMET"])
async def predict_admet(request: ADMETRequest):
    """Run ADMET property prediction for a list of SMILES.

    Returns absorption, distribution, metabolism, excretion, and toxicity
    predictions for each compound plus a composite ADMET score.
    """
    t0 = time.time()
    try:
        from admet.predictor import ADMETPredictor
        predictor = ADMETPredictor()
    except ImportError:
        raise HTTPException(status_code=503, detail="ADMET module not available")

    results = []
    for smi in request.smiles:
        try:
            r = predictor.predict(smi)
            results.append(ADMETEndpointResult(
                smiles=smi,
                solubility_log_s=r.solubility_log_s,
                bbb_penetration=r.bbb_penetration,
                herg_inhibition=r.herg_inhibition,
                ames_mutagenicity=r.ames_mutagenicity,
                hepatotoxicity=r.hepatotoxicity,
                oral_bioavailability=r.oral_bioavailability,
                qed=r.qed,
                lipinski_pass=r.lipinski_pass,
                admet_score=r.admet_score,
                concerns=r.flag_concerns(),
            ))
        except Exception as exc:
            logger.warning("ADMET prediction failed for {}: {}", smi, exc)
            results.append(ADMETEndpointResult(
                smiles=smi,
                solubility_log_s=0.0, bbb_penetration=0.0,
                herg_inhibition=0.0, ames_mutagenicity=0.0,
                hepatotoxicity=0.0, oral_bioavailability=0.0,
                qed=0.0, lipinski_pass=False, admet_score=0.0,
                concerns=["prediction_failed"],
            ))

    return ADMETResponse(
        results=results,
        processing_time_s=round(time.time() - t0, 3),
    )


# ---------------------------------------------------------------------------
# Multi-task QSAR endpoint
# ---------------------------------------------------------------------------

class MultiTaskRequest(BaseModel):
    smiles: List[str] = Field(..., min_length=1, max_length=500)
    targets: List[str] = Field(
        default=["InhA", "KatG", "rpoB", "DprE1", "MmpL3"],
        description="TB targets to predict",
    )
    model_path: str = Field(
        default="models/multitask/multitask_qsar.joblib",
        description="Path to trained MultiTaskQSAR model",
    )

    @validator("smiles", each_item=True)
    def validate_smiles_len(cls, v: str) -> str:
        if len(v) > 500:
            raise ValueError(f"SMILES too long")
        return v


class MultiTaskPrediction(BaseModel):
    smiles: str
    predictions: Dict[str, float]
    uncertainties: Dict[str, float]


class MultiTaskResponse(BaseModel):
    results: List[MultiTaskPrediction]
    targets: List[str]
    processing_time_s: float


@app.post("/predict/multitask", response_model=MultiTaskResponse, tags=["Multi-task QSAR"])
async def predict_multitask(request: MultiTaskRequest):
    """Multi-task QSAR prediction across TB protein targets.

    Returns per-target activity probability and epistemic uncertainty
    (MC-Dropout) for each input SMILES.
    """
    t0 = time.time()
    import os

    try:
        from models.multitask_qsar import MultiTaskQSAR
        from data.multitask_loader import compute_descriptors, MultiTaskDataset
    except ImportError:
        raise HTTPException(status_code=503, detail="MultiTaskQSAR module unavailable")

    if not os.path.exists(request.model_path):
        raise HTTPException(
            status_code=404,
            detail=f"Multi-task model not found at {request.model_path}. "
                   "Train it first with scripts/train_multitask.py",
        )

    model = MultiTaskQSAR.load(request.model_path)
    dummy_ds = MultiTaskDataset(request.smiles, {t: np.full(len(request.smiles), np.nan) for t in request.targets})
    X = compute_descriptors(dummy_ds)
    proba = model.predict_proba(X)
    unc = model.predict_uncertainty(X)

    results = []
    for i, smi in enumerate(request.smiles):
        results.append(MultiTaskPrediction(
            smiles=smi,
            predictions={t: float(proba[t][i]) for t in model.targets if t in proba},
            uncertainties={t: float(unc[t][i]) for t in model.targets if t in unc},
        ))

    return MultiTaskResponse(
        results=results,
        targets=model.targets,
        processing_time_s=round(time.time() - t0, 3),
    )


# ---------------------------------------------------------------------------
# SAR cliff analysis endpoint
# ---------------------------------------------------------------------------

class SARRequest(BaseModel):
    smiles: List[str] = Field(..., min_length=2, max_length=1000)
    activities: List[float] = Field(..., description="pIC50 values (NaN for unknown)")
    target: str = Field(default="", description="Target name for provenance")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    activity_threshold: float = Field(default=1.0, ge=0.0)


class SARCliffResult(BaseModel):
    smiles_a: str
    smiles_b: str
    similarity: float
    delta_activity: float
    cliff_score: float


class SARResponse(BaseModel):
    n_cliffs: int
    cliffs: List[SARCliffResult]
    mean_cliff_score: float
    activity_landscape_index: float
    processing_time_s: float


@app.post("/analysis/sar-cliffs", response_model=SARResponse, tags=["SAR Analysis"])
async def sar_cliff_analysis(request: SARRequest):
    """Detect SAR activity cliffs in a compound dataset.

    Returns pairs of structurally similar but activity-different compounds.
    """
    t0 = time.time()
    try:
        from analysis.sar_analysis import (
            SARCliffConfig, detect_sar_cliffs, cliff_summary, activity_landscape_index
        )
    except ImportError:
        raise HTTPException(status_code=503, detail="SAR analysis module unavailable")

    if len(request.smiles) != len(request.activities):
        raise HTTPException(status_code=400, detail="smiles and activities must have equal length")

    acts = np.array(request.activities, dtype=np.float32)
    cfg = SARCliffConfig(
        similarity_threshold=request.similarity_threshold,
        activity_threshold=request.activity_threshold,
    )
    cliffs = detect_sar_cliffs(request.smiles, acts, target=request.target, config=cfg)
    summary = cliff_summary(cliffs)
    ali = activity_landscape_index(request.smiles, acts)

    return SARResponse(
        n_cliffs=len(cliffs),
        cliffs=[
            SARCliffResult(
                smiles_a=c.smiles_a, smiles_b=c.smiles_b,
                similarity=c.similarity, delta_activity=c.delta_activity,
                cliff_score=c.cliff_score,
            )
            for c in cliffs[:50]  # cap response size
        ],
        mean_cliff_score=summary.get("mean_cliff_score", 0.0),
        activity_landscape_index=ali,
        processing_time_s=round(time.time() - t0, 3),
    )
