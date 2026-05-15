"""src.admet — ADMET property prediction for TB drug candidates.

Endpoints:
  Absorption  : Caco-2 permeability, aqueous solubility (ESOL/SLogP)
  Distribution: BBB penetration, plasma protein binding, VD
  Metabolism  : CYP450 inhibition (1A2, 2C9, 2C19, 2D6, 3A4)
  Excretion   : half-life (log t1/2 regression)
  Toxicity    : hERG inhibition, Ames mutagenicity, hepatotoxicity

Classes
-------
ADMETPredictor  : unified predictor returning all endpoints
ADMETResult     : structured dataclass with per-endpoint scores
ADMETFilter     : hard-filter based on configurable thresholds
admet_score     : scalar drug-likeness score (weighted sum)
"""

from .predictor import (
    ADMETFilter,
    ADMETPredictor,
    ADMETResult,
    ADMETThresholds,
    admet_score,
    batch_admet,
)

__all__ = [
    "ADMETPredictor",
    "ADMETResult",
    "ADMETFilter",
    "ADMETThresholds",
    "admet_score",
    "batch_admet",
]
