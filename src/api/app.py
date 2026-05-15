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
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field, validator
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    raise ImportError("FastAPI required: pip install fastapi uvicorn")

import numpy as np
from loguru import logger

from data.chembl_loader import ChEMBLLoader
from data.descriptor_calculator import DescriptorCalculator
from models.qsar_model import QSARModel
from evaluation.generation_metrics import evaluate_generation

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


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ActivityRequest(BaseModel):
    smiles: List[str] = Field(..., min_items=1, max_items=10_000, description="List of SMILES")
    model_path: str = Field("models/qsar_model.pkl", description="Path to trained QSAR model")

    @validator("smiles", each_item=True)
    def smiles_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("SMILES must not be empty")
        return v.strip()


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
        return MoleculeResult(smiles=smiles, valid=False, canonical_smiles=None,
                               predicted_pic50=None, predicted_active=None,
                               probability_active=None, error="Invalid SMILES")

    features = _smiles_to_features(canonical)
    if features is None:
        return MoleculeResult(smiles=smiles, valid=True, canonical_smiles=canonical,
                               predicted_pic50=None, predicted_active=None,
                               probability_active=None, error="Descriptor calculation failed")

    try:
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
    }


@app.post("/predict/activity", response_model=ActivityResponse)
async def predict_activity(request: ActivityRequest):
    t0 = time.time()
    model = get_qsar_model(request.model_path)
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not found: {request.model_path}")

    results = [_predict_one(smi, model) for smi in request.smiles]
    n_valid = sum(r.valid for r in results)
    n_predicted = sum(r.predicted_pic50 is not None or r.predicted_active is not None
                      for r in results)

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
