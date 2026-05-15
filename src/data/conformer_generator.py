"""3D conformer generation and shape/pharmacophoric descriptors.

Workflow
--------
1. Generate low-energy 3D conformers via ETKDG (RDKit)
2. Optimize with MMFF94s force field
3. Compute shape descriptors: PMI ratios (npr1, npr2), asphericity, eccentricity
4. Compute WHIM / autocorrelation 3D descriptors (via RDKit rdMolDescriptors)
5. Compute E3FP fingerprint (extended 3D fingerprint — point cloud on conformer)

All descriptors are returned as numpy float32 arrays compatible with the
QSAR pipeline.  Falls back to zeros when RDKit is unavailable (CI without
full cheminformatics stack).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ConformerConfig:
    n_confs: int = 10
    random_seed: int = 42
    max_iters: int = 2000
    prune_rms_threshold: float = 0.5
    force_field: str = "MMFF94s"
    use_random_coords: bool = False


@dataclass
class Descriptor3DConfig:
    include_pmi: bool = True
    include_whim: bool = True
    include_autocorr: bool = True
    include_e3fp: bool = True
    e3fp_bits: int = 1024
    e3fp_radius_multiplier: float = 1.718
    e3fp_rdkit_bits: int = 64  # bits per shell for RDKit-based E3FP approx


# ---------------------------------------------------------------------------
# Conformer generation
# ---------------------------------------------------------------------------


def generate_conformers(
    smiles: str,
    config: ConformerConfig | None = None,
) -> Any | None:
    """Generate and optimise conformers for a SMILES string.

    Returns the RDKit Mol with embedded conformers, or None on failure.
    """
    cfg = config or ConformerConfig()
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = cfg.random_seed
        params.pruneRmsThresh = cfg.prune_rms_threshold
        params.useRandomCoords = cfg.use_random_coords

        ids = AllChem.EmbedMultipleConfs(mol, numConfs=cfg.n_confs, params=params)
        if len(ids) == 0:
            # fallback: distance geometry without ETKDG
            AllChem.EmbedMolecule(mol, randomSeed=cfg.random_seed)
            ids = list(mol.GetConformers())
            if not ids:
                return None

        # Force-field minimisation
        if cfg.force_field == "MMFF94s":
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            for cid in mol.GetConformers():
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid.GetId())
                if ff is not None:
                    ff.Minimize(maxIts=cfg.max_iters)
        else:
            AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=cfg.max_iters)

        return mol
    except ImportError:
        logger.warning("RDKit not available — skipping conformer generation")
        return None
    except Exception as exc:
        logger.debug("Conformer generation failed for '%s': %s", smiles, exc)
        return None


def get_lowest_energy_conformer_id(mol: Any) -> int:
    """Return the conf ID with the lowest MMFF94s energy."""
    try:
        from rdkit.Chem import AllChem

        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        energies = []
        for conf in mol.GetConformers():
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf.GetId())
            if ff is not None:
                energies.append((ff.CalcEnergy(), conf.GetId()))
        if energies:
            return min(energies)[1]
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# PMI / Shape descriptors
# ---------------------------------------------------------------------------


def compute_pmi_descriptors(mol: Any, conf_id: int = -1) -> NDArray:
    """Principal moments of inertia ratios and derived shape indices.

    Returns 6-element array: [npr1, npr2, asphericity, eccentricity,
                               inertial_shape_factor, spherocity_index]
    Returns zeros on failure.
    """
    try:
        from rdkit.Chem import Descriptors3D

        d = {
            "npr1": Descriptors3D.NPR1(mol, confId=conf_id),
            "npr2": Descriptors3D.NPR2(mol, confId=conf_id),
            "asphericity": Descriptors3D.Asphericity(mol, confId=conf_id),
            "eccentricity": Descriptors3D.Eccentricity(mol, confId=conf_id),
            "isf": Descriptors3D.InertialShapeFactor(mol, confId=conf_id),
            "spherocity": Descriptors3D.SpherocityIndex(mol, confId=conf_id),
        }
        return np.array(list(d.values()), dtype=np.float32)
    except Exception as exc:
        logger.debug("PMI descriptors failed: %s", exc)
        return np.zeros(6, dtype=np.float32)


PMI_FEATURE_NAMES = [
    "pmi_npr1", "pmi_npr2", "pmi_asphericity",
    "pmi_eccentricity", "pmi_inertial_shape_factor", "pmi_spherocity_index",
]


# ---------------------------------------------------------------------------
# WHIM descriptors
# ---------------------------------------------------------------------------


def compute_whim_descriptors(mol: Any, conf_id: int = -1) -> NDArray:
    """114-element WHIM (Weighted Holistic Invariant Molecular) descriptor vector."""
    try:
        from rdkit.Chem import rdMolDescriptors

        whim = rdMolDescriptors.CalcWHIM(mol, confId=conf_id)
        return np.array(whim, dtype=np.float32)
    except Exception as exc:
        logger.debug("WHIM descriptors failed: %s", exc)
        return np.zeros(114, dtype=np.float32)


# ---------------------------------------------------------------------------
# RDF / Autocorrelation 3D
# ---------------------------------------------------------------------------


def compute_autocorr3d(mol: Any, conf_id: int = -1) -> NDArray:
    """80-element 3D autocorrelation descriptor (MOREAU-BROTO)."""
    try:
        from rdkit.Chem import rdMolDescriptors

        ac = rdMolDescriptors.CalcAUTOCORR3D(mol, confId=conf_id)
        return np.array(ac, dtype=np.float32)
    except Exception as exc:
        logger.debug("Autocorr3D failed: %s", exc)
        return np.zeros(80, dtype=np.float32)


# ---------------------------------------------------------------------------
# E3FP-style fingerprint (RDKit approximation)
# ---------------------------------------------------------------------------


def compute_e3fp_fingerprint(
    mol: Any,
    conf_id: int = -1,
    n_bits: int = 1024,
    radius_multiplier: float = 1.718,
    n_shells: int = 5,
) -> NDArray:
    """Approximate E3FP: radial shells of Morgan-like hashing in 3D space.

    True E3FP (e3fp library) uses atom coordinates explicitly.  This
    approximation hashes atom-environment pairs at geometric shells and
    folds into a fixed-length bit-vector — captures 3D shape without the
    full e3fp dependency.

    Returns n_bits-element binary float32 array.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdMolDescriptors

        conf = mol.GetConformer(conf_id)
        positions = conf.GetPositions()

        # centroid
        centroid = positions.mean(axis=0)
        dists = np.linalg.norm(positions - centroid, axis=1)
        max_dist = dists.max() if dists.max() > 0 else 1.0

        fp_bits: set[int] = set()
        for shell in range(1, n_shells + 1):
            r_max = max_dist * (shell / n_shells) * radius_multiplier
            r_min = max_dist * ((shell - 1) / n_shells) * radius_multiplier
            atom_indices = [
                i for i, d in enumerate(dists) if r_min <= d < r_max
            ]
            if not atom_indices:
                continue
            # Morgan hash for atoms in this shell
            sub_mol_env = Chem.RWMol(mol)
            env_fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=n_bits,
                fromAtoms=atom_indices,
            )
            for b in env_fp.GetOnBits():
                fp_bits.add((b + shell * 7919) % n_bits)

        arr = np.zeros(n_bits, dtype=np.float32)
        for b in fp_bits:
            arr[b] = 1.0
        return arr

    except Exception as exc:
        logger.debug("E3FP fingerprint failed: %s", exc)
        return np.zeros(n_bits, dtype=np.float32)


# ---------------------------------------------------------------------------
# Combined 3D descriptor calculator
# ---------------------------------------------------------------------------


class Descriptor3DCalculator:
    """Compute all 3D descriptors for a list of SMILES.

    Descriptor vector layout (default settings, 1214 dims total):
      [0:6]     PMI shape descriptors
      [6:120]   WHIM (114 dims)
      [120:200] Autocorrelation 3D (80 dims)
      [200:1224] E3FP approximation (1024 dims)
    """

    def __init__(
        self,
        conf_config: ConformerConfig | None = None,
        desc_config: Descriptor3DConfig | None = None,
    ) -> None:
        self.conf_config = conf_config or ConformerConfig()
        self.desc_config = desc_config or Descriptor3DConfig()
        self._dim: int | None = None

    @property
    def n_features(self) -> int:
        """Total descriptor dimension."""
        d = 0
        if self.desc_config.include_pmi:
            d += 6
        if self.desc_config.include_whim:
            d += 114
        if self.desc_config.include_autocorr:
            d += 80
        if self.desc_config.include_e3fp:
            d += self.desc_config.e3fp_bits
        return d

    @property
    def feature_names(self) -> list[str]:
        names: list[str] = []
        if self.desc_config.include_pmi:
            names.extend(PMI_FEATURE_NAMES)
        if self.desc_config.include_whim:
            names.extend([f"whim_{i}" for i in range(114)])
        if self.desc_config.include_autocorr:
            names.extend([f"autocorr3d_{i}" for i in range(80)])
        if self.desc_config.include_e3fp:
            names.extend([f"e3fp_{i}" for i in range(self.desc_config.e3fp_bits)])
        return names

    def compute_single(self, smiles: str) -> NDArray:
        """Compute 3D descriptors for one SMILES. Returns zeros on failure."""
        mol = generate_conformers(smiles, self.conf_config)
        if mol is None:
            return np.zeros(self.n_features, dtype=np.float32)

        conf_id = get_lowest_energy_conformer_id(mol)
        parts: list[NDArray] = []

        if self.desc_config.include_pmi:
            parts.append(compute_pmi_descriptors(mol, conf_id))

        if self.desc_config.include_whim:
            parts.append(compute_whim_descriptors(mol, conf_id))

        if self.desc_config.include_autocorr:
            parts.append(compute_autocorr3d(mol, conf_id))

        if self.desc_config.include_e3fp:
            parts.append(
                compute_e3fp_fingerprint(
                    mol,
                    conf_id=conf_id,
                    n_bits=self.desc_config.e3fp_bits,
                    radius_multiplier=self.desc_config.e3fp_radius_multiplier,
                )
            )

        return np.concatenate(parts).astype(np.float32)

    def compute_batch(
        self,
        smiles_list: list[str],
        n_jobs: int = 1,
        show_progress: bool = False,
    ) -> NDArray:
        """Compute 3D descriptors for a list of SMILES.

        Parameters
        ----------
        n_jobs : number of parallel workers (uses joblib if > 1)
        show_progress : show tqdm progress bar
        """
        if show_progress:
            try:
                from tqdm import tqdm

                smiles_list = list(tqdm(smiles_list, desc="3D descriptors"))
            except ImportError:
                pass

        if n_jobs == 1:
            rows = [self.compute_single(smi) for smi in smiles_list]
        else:
            try:
                from joblib import Parallel, delayed

                rows = Parallel(n_jobs=n_jobs)(
                    delayed(self.compute_single)(smi) for smi in smiles_list
                )
            except ImportError:
                rows = [self.compute_single(smi) for smi in smiles_list]

        return np.stack(rows, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience: augment existing 2D descriptors with 3D
# ---------------------------------------------------------------------------


def augment_with_3d(
    X_2d: NDArray,
    smiles_list: list[str],
    conf_config: ConformerConfig | None = None,
    desc_config: Descriptor3DConfig | None = None,
    n_jobs: int = 1,
) -> NDArray:
    """Concatenate 2D descriptor matrix with 3D shape descriptors.

    Parameters
    ----------
    X_2d : (n, d_2d) existing descriptor matrix
    smiles_list : corresponding SMILES strings

    Returns
    -------
    (n, d_2d + d_3d) combined descriptor matrix
    """
    calc = Descriptor3DCalculator(conf_config, desc_config)
    X_3d = calc.compute_batch(smiles_list, n_jobs=n_jobs)
    return np.concatenate([X_2d, X_3d], axis=1)
