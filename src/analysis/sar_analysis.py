"""SAR cliff detection and Matched Molecular Pairs (MMP) analysis.

SAR Cliff
---------
Two compounds A and B form an activity cliff when:
  - structural similarity > threshold (e.g. Tanimoto ≥ 0.7)
  - activity difference > threshold (e.g. |ΔpIC50| ≥ 1.0 = 10-fold)

MMP (Matched Molecular Pairs)
------------------------------
A pair (A, B) where B = A with one functional group substitution.
Detected via RDKit's MMPA fragmentation (cut one bond, keep largest fragment).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SARCliff:
    smiles_a: str
    smiles_b: str
    similarity: float
    delta_activity: float
    target: str = ""

    @property
    def cliff_score(self) -> float:
        """Higher = more surprising cliff."""
        return self.similarity * abs(self.delta_activity)


@dataclass
class MMPair:
    smiles_a: str
    smiles_b: str
    core: str          # common scaffold SMILES
    substituent_a: str
    substituent_b: str
    delta_activity: float
    target: str = ""

    @property
    def transformation(self) -> str:
        return f"{self.substituent_a} → {self.substituent_b}"


# ---------------------------------------------------------------------------
# Fingerprint utilities
# ---------------------------------------------------------------------------


def _compute_fps(smiles_list: list[str], n_bits: int = 1024) -> NDArray:
    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        fps = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            fps.append(
                gen.GetFingerprintAsNumPy(mol).astype(np.float32)
                if mol else np.zeros(n_bits, dtype=np.float32)
            )
        return np.stack(fps)
    except ImportError:
        rng = np.random.default_rng(42)
        return rng.random((len(smiles_list), n_bits)).astype(np.float32)


def tanimoto_matrix(fps: NDArray) -> NDArray:
    """Compute pairwise Tanimoto similarity matrix (bit vectors)."""
    dot = fps @ fps.T
    norms = fps.sum(axis=1, keepdims=True)
    union = norms + norms.T - dot
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, dot / union, 0.0)
    return sim.astype(np.float32)


# ---------------------------------------------------------------------------
# SAR Cliff Detection
# ---------------------------------------------------------------------------


@dataclass
class SARCliffConfig:
    similarity_threshold: float = 0.70
    activity_threshold: float = 1.0  # |ΔpIC50| ≥ 1 = 10-fold potency difference
    fp_bits: int = 1024
    max_pairs: int = 10000  # cap to avoid O(n²) blow-up


def detect_sar_cliffs(
    smiles_list: list[str],
    activities: NDArray,
    target: str = "",
    config: SARCliffConfig | None = None,
) -> list[SARCliff]:
    """Detect SAR cliffs in a dataset.

    Parameters
    ----------
    smiles_list : list of SMILES
    activities  : pIC50 values (NaN = unknown, skipped)
    target      : optional target label for provenance
    """
    cfg = config or SARCliffConfig()
    fps = _compute_fps(smiles_list, n_bits=cfg.fp_bits)
    sim_matrix = tanimoto_matrix(fps)
    acts = np.asarray(activities, dtype=np.float32)

    n = len(smiles_list)
    cliffs: list[SARCliff] = []

    for i in range(n):
        if np.isnan(acts[i]):
            continue
        for j in range(i + 1, n):
            if np.isnan(acts[j]):
                continue
            sim = float(sim_matrix[i, j])
            if sim < cfg.similarity_threshold:
                continue
            delta = abs(float(acts[i]) - float(acts[j]))
            if delta < cfg.activity_threshold:
                continue
            cliffs.append(
                SARCliff(
                    smiles_a=smiles_list[i],
                    smiles_b=smiles_list[j],
                    similarity=sim,
                    delta_activity=delta,
                    target=target,
                )
            )
            if len(cliffs) >= cfg.max_pairs:
                logger.warning("SAR cliff limit (%d) reached", cfg.max_pairs)
                return cliffs

    cliffs.sort(key=lambda c: c.cliff_score, reverse=True)
    return cliffs


def cliffs_to_dataframe(cliffs: list[SARCliff]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "smiles_a": c.smiles_a,
            "smiles_b": c.smiles_b,
            "similarity": c.similarity,
            "delta_activity": c.delta_activity,
            "cliff_score": c.cliff_score,
            "target": c.target,
        }
        for c in cliffs
    ])


def cliff_summary(cliffs: list[SARCliff]) -> dict[str, Any]:
    if not cliffs:
        return {"n_cliffs": 0}
    scores = [c.cliff_score for c in cliffs]
    return {
        "n_cliffs": len(cliffs),
        "mean_cliff_score": float(np.mean(scores)),
        "max_cliff_score": float(np.max(scores)),
        "mean_similarity": float(np.mean([c.similarity for c in cliffs])),
        "mean_delta_activity": float(np.mean([c.delta_activity for c in cliffs])),
    }


# ---------------------------------------------------------------------------
# Matched Molecular Pairs
# ---------------------------------------------------------------------------


@dataclass
class MMPConfig:
    max_heavy_cut: int = 10  # max heavy atoms in exchanged fragment
    min_core_atoms: int = 5  # minimum core size
    fp_bits: int = 1024


def _fragment_molecule(smiles: str) -> list[tuple[str, str]]:
    """Cut each non-ring bond and return (core, substituent) SMILES pairs.

    Uses RDKit BRICS fragmentation as a proxy for MMP fragmentation.
    Returns empty list if RDKit is unavailable or parsing fails.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import BRICS, AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        # BRICS fragments
        frags = list(BRICS.BRICSDecompose(mol, returnMols=False))
        results = []
        for frag in frags:
            # Each frag is a SMILES; use largest fragment as core
            parts = frag.split(".")
            parts_sorted = sorted(parts, key=lambda s: len(s), reverse=True)
            if len(parts_sorted) >= 2:
                results.append((parts_sorted[0], parts_sorted[1]))
        return results
    except Exception:
        return []


def find_matched_pairs(
    smiles_list: list[str],
    activities: NDArray,
    target: str = "",
    config: MMPConfig | None = None,
) -> list[MMPair]:
    """Find Matched Molecular Pairs between compounds.

    Two compounds form an MMP if they share a common core and differ
    only in one substituent.
    """
    cfg = config or MMPConfig()

    # Build core → [(smiles, substituent, activity)] index
    core_map: dict[str, list[tuple[str, str, float]]] = {}

    for smi, act in zip(smiles_list, activities):
        if np.isnan(float(act)):
            continue
        for core, sub in _fragment_molecule(smi):
            if not core or not sub:
                continue
            # Normalise by canonical form
            try:
                from rdkit import Chem

                core_mol = Chem.MolFromSmiles(core.replace("[*]", "[H]"))
                if core_mol and core_mol.GetNumHeavyAtoms() >= cfg.min_core_atoms:
                    core_can = Chem.MolToSmiles(core_mol)
                    core_map.setdefault(core_can, []).append((smi, sub, float(act)))
            except Exception:
                core_map.setdefault(core, []).append((smi, sub, float(act)))

    pairs: list[MMPair] = []
    for core, entries in core_map.items():
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                smi_a, sub_a, act_a = entries[i]
                smi_b, sub_b, act_j = entries[j]
                if smi_a == smi_b:
                    continue
                if sub_a == sub_b:
                    continue
                delta = act_b = act_j - act_a
                pairs.append(
                    MMPair(
                        smiles_a=smi_a,
                        smiles_b=smi_b,
                        core=core,
                        substituent_a=sub_a,
                        substituent_b=sub_b,
                        delta_activity=float(act_b),
                        target=target,
                    )
                )

    pairs.sort(key=lambda p: abs(p.delta_activity), reverse=True)
    return pairs


def mmp_to_dataframe(pairs: list[MMPair]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "smiles_a": p.smiles_a,
            "smiles_b": p.smiles_b,
            "core": p.core,
            "substituent_a": p.substituent_a,
            "substituent_b": p.substituent_b,
            "delta_activity": p.delta_activity,
            "transformation": p.transformation,
            "target": p.target,
        }
        for p in pairs
    ])


def transformation_summary(pairs: list[MMPair]) -> pd.DataFrame:
    """Aggregate by transformation and compute mean ΔpIC50."""
    if not pairs:
        return pd.DataFrame()
    df = mmp_to_dataframe(pairs)
    return (
        df.groupby("transformation")["delta_activity"]
        .agg(["mean", "count", "std"])
        .rename(columns={"mean": "mean_delta", "count": "n_pairs", "std": "std_delta"})
        .sort_values("mean_delta", ascending=False)
    )


# ---------------------------------------------------------------------------
# Activity landscape analysis
# ---------------------------------------------------------------------------


def activity_landscape_index(
    smiles_list: list[str],
    activities: NDArray,
    n_bits: int = 1024,
) -> float:
    """Compute the Structure-Activity Landscape Index (SALI).

    SALI = mean(|ΔAct_ij| / (1 - Sim_ij)) over all pairs
    Higher SALI = more activity cliffs = harder dataset for ML.
    """
    fps = _compute_fps(smiles_list, n_bits=n_bits)
    sim = tanimoto_matrix(fps)
    acts = np.asarray(activities, dtype=np.float32)
    n = len(smiles_list)

    total = 0.0
    count = 0
    for i in range(n):
        if np.isnan(acts[i]):
            continue
        for j in range(i + 1, n):
            if np.isnan(acts[j]):
                continue
            s = float(sim[i, j])
            if s >= 1.0:
                continue
            total += abs(float(acts[i]) - float(acts[j])) / (1.0 - s + 1e-8)
            count += 1

    return float(total / count) if count > 0 else 0.0
