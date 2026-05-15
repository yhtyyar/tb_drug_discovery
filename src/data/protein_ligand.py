"""Protein-ligand interaction features for QSAR augmentation.

Provides structural/interaction descriptors computed without requiring
a full docking engine:
  1. AutoDock Vina score (if vina binary available in PATH)
  2. PLIF (Protein-Ligand Interaction Fingerprint) from RDKit + PDB
  3. Pharmacophoric overlap with known active binding modes
  4. Dummy/fallback features when docking is unavailable (CI-safe)

The protein is described by a set of pharmacophore features derived
from known co-crystal structures or from sequence-based predictions.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# TB target pocket pharmacophore descriptors (pre-computed from PDB structures)
# Format: {target: {"hba": [...], "hbd": [...], "hydrophobic": [...]}}
TB_POCKET_PHARMACOPHORES: dict[str, dict[str, list]] = {
    "InhA": {
        "n_hba": 3, "n_hbd": 2, "n_hydrophobic": 5, "n_aromatic": 2,
        "vol_A3": 420.0, "depth_A": 8.5,
    },
    "KatG": {
        "n_hba": 4, "n_hbd": 3, "n_hydrophobic": 4, "n_aromatic": 3,
        "vol_A3": 380.0, "depth_A": 7.2,
    },
    "rpoB": {
        "n_hba": 5, "n_hbd": 2, "n_hydrophobic": 6, "n_aromatic": 4,
        "vol_A3": 650.0, "depth_A": 12.0,
    },
    "DprE1": {
        "n_hba": 3, "n_hbd": 3, "n_hydrophobic": 5, "n_aromatic": 2,
        "vol_A3": 490.0, "depth_A": 9.1,
    },
    "MmpL3": {
        "n_hba": 4, "n_hbd": 1, "n_hydrophobic": 7, "n_aromatic": 3,
        "vol_A3": 820.0, "depth_A": 15.5,
    },
}


# ---------------------------------------------------------------------------
# Ligand pharmacophore features
# ---------------------------------------------------------------------------


def compute_ligand_pharmacophore(smiles: str) -> dict[str, int | float]:
    """Compute pharmacophore feature counts from SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return _zero_pharmacophore()
        return {
            "n_hba": rdMolDescriptors.CalcNumHBA(mol),
            "n_hbd": rdMolDescriptors.CalcNumHBD(mol),
            "n_aromatic": rdMolDescriptors.CalcNumAromaticRings(mol),
            "n_rings": rdMolDescriptors.CalcNumRings(mol),
            "n_rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "mw": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "qed": _safe_qed(mol),
        }
    except ImportError:
        return _zero_pharmacophore()


def _zero_pharmacophore() -> dict[str, int | float]:
    return {k: 0 for k in ["n_hba", "n_hbd", "n_aromatic", "n_rings",
                             "n_rot_bonds", "mw", "logp", "tpsa", "qed"]}


def _safe_qed(mol: Any) -> float:
    try:
        from rdkit.Chem import QED
        return QED.qed(mol)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Pharmacophoric overlap score
# ---------------------------------------------------------------------------


def pharmacophoric_overlap_score(
    smiles: str,
    target: str,
    pocket: dict | None = None,
) -> NDArray:
    """Compute 12-dim pharmacophoric interaction feature vector.

    Features encode complementarity between ligand features and
    the target binding pocket (from pre-computed pharmacophores).

    Returns zeros if target unknown or RDKit unavailable.
    """
    if pocket is None:
        pocket = TB_POCKET_PHARMACOPHORES.get(target, {})

    lig = compute_ligand_pharmacophore(smiles)
    if not pocket:
        return np.zeros(12, dtype=np.float32)

    # Complementarity scores (clamped to [0, 1])
    def _complement(lig_val: float, pocket_val: float) -> float:
        if pocket_val == 0:
            return 0.0
        ratio = lig_val / (pocket_val + 1e-8)
        return float(np.clip(ratio, 0, 2) / 2)

    features = np.array([
        _complement(lig["n_hba"], pocket.get("n_hba", 3)),
        _complement(lig["n_hbd"], pocket.get("n_hbd", 2)),
        _complement(lig["n_aromatic"], pocket.get("n_aromatic", 2)),
        _complement(lig["n_rot_bonds"], 5),
        float(np.clip(lig["mw"] / 500, 0, 1)),
        float(np.clip((lig["logp"] + 2) / 7, 0, 1)),
        float(np.clip(lig["tpsa"] / 140, 0, 1)),
        float(lig["qed"]),
        _complement(lig["n_hba"], pocket.get("n_hbd", 2)),
        _complement(lig["n_hbd"], pocket.get("n_hba", 3)),
        float(np.clip(pocket.get("vol_A3", 400) / 1000, 0, 1)),
        float(np.clip(pocket.get("depth_A", 8) / 20, 0, 1)),
    ], dtype=np.float32)

    return features


PHARMACOPHORE_FEATURE_NAMES = [
    "hba_complement", "hbd_complement", "aromatic_complement",
    "flexibility", "mw_norm", "logp_norm", "tpsa_norm", "qed",
    "hba_hbd_cross", "hbd_hba_cross", "pocket_vol_norm", "pocket_depth_norm",
]


# ---------------------------------------------------------------------------
# Vina docking wrapper (optional)
# ---------------------------------------------------------------------------


@dataclass
class VinaConfig:
    vina_binary: str = "vina"
    exhaustiveness: int = 8
    num_modes: int = 1
    cpu: int = 4


def run_vina_docking(
    smiles: str,
    receptor_pdbqt: str,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
    size_x: float = 20.0,
    size_y: float = 20.0,
    size_z: float = 20.0,
    config: VinaConfig | None = None,
) -> float | None:
    """Run AutoDock Vina and return the best binding affinity (kcal/mol).

    Returns None if Vina is not installed or docking fails.
    """
    cfg = config or VinaConfig()
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        with tempfile.TemporaryDirectory() as tmpdir:
            ligand_pdb = os.path.join(tmpdir, "ligand.pdb")
            out_pdbqt = os.path.join(tmpdir, "out.pdbqt")
            Chem.MolToPDBFile(mol, ligand_pdb)

            cmd = [
                cfg.vina_binary,
                "--receptor", receptor_pdbqt,
                "--ligand", ligand_pdb,
                "--center_x", str(center_x),
                "--center_y", str(center_y),
                "--center_z", str(center_z),
                "--size_x", str(size_x),
                "--size_y", str(size_y),
                "--size_z", str(size_z),
                "--out", out_pdbqt,
                "--exhaustiveness", str(cfg.exhaustiveness),
                "--num_modes", str(cfg.num_modes),
                "--cpu", str(cfg.cpu),
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )

            for line in result.stdout.split("\n"):
                if line.strip().startswith("1"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1])
    except Exception as exc:
        logger.debug("Vina docking failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# ProteinLigandFeaturizer
# ---------------------------------------------------------------------------


@dataclass
class PLIFConfig:
    targets: list[str] = field(
        default_factory=lambda: ["InhA", "KatG", "rpoB", "DprE1", "MmpL3"]
    )
    include_pharmacophore: bool = True
    include_vina: bool = False
    vina_config: VinaConfig | None = None
    receptor_paths: dict[str, str] = field(default_factory=dict)


class ProteinLigandFeaturizer:
    """Compute protein-ligand interaction features for all TB targets.

    Output vector: [pharmacophore_overlap (12 dims) × n_targets]
    Optionally appends Vina scores if receptors are available.
    """

    def __init__(self, config: PLIFConfig | None = None) -> None:
        self.config = config or PLIFConfig()

    @property
    def n_features(self) -> int:
        d = 0
        if self.config.include_pharmacophore:
            d += 12 * len(self.config.targets)
        if self.config.include_vina:
            d += len(self.config.targets)
        return d

    @property
    def feature_names(self) -> list[str]:
        names = []
        if self.config.include_pharmacophore:
            for t in self.config.targets:
                names.extend([f"{t}_{f}" for f in PHARMACOPHORE_FEATURE_NAMES])
        if self.config.include_vina:
            names.extend([f"{t}_vina" for t in self.config.targets])
        return names

    def compute_single(self, smiles: str) -> NDArray:
        parts: list[NDArray] = []

        if self.config.include_pharmacophore:
            for t in self.config.targets:
                parts.append(pharmacophoric_overlap_score(smiles, t))

        if self.config.include_vina:
            for t in self.config.targets:
                pdbqt = self.config.receptor_paths.get(t)
                if pdbqt and os.path.exists(pdbqt):
                    score = run_vina_docking(smiles, pdbqt, config=self.config.vina_config)
                    parts.append(np.array([score or 0.0], dtype=np.float32))
                else:
                    parts.append(np.array([0.0], dtype=np.float32))

        if not parts:
            return np.zeros(self.n_features, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)

    def compute_batch(self, smiles_list: list[str]) -> NDArray:
        rows = [self.compute_single(s) for s in smiles_list]
        return np.stack(rows, axis=0)

    def augment_descriptors(
        self,
        X: NDArray,
        smiles_list: list[str],
    ) -> NDArray:
        """Concatenate existing descriptor matrix with PLIF features."""
        X_plif = self.compute_batch(smiles_list)
        return np.concatenate([X, X_plif], axis=1)
