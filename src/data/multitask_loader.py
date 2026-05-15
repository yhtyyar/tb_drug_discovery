"""Multi-target dataset loader for TB drug discovery.

Loads and aligns activity data across five M. tuberculosis targets:
  InhA  (CHEMBL1849) — enoyl-ACP reductase
  KatG  (CHEMBL1790) — catalase-peroxidase (isoniazid activator)
  rpoB  (CHEMBL1916) — RNA polymerase beta (rifampicin target)
  DprE1 (CHEMBL3622) — decaprenylphosphoryl-β-D-ribose 2'-oxidase
  MmpL3 (CHEMBL4296) — mycolic acid transport protein

Data may be loaded from:
  1. A pre-existing CSV with columns [smiles, InhA, KatG, rpoB, DprE1, MmpL3]
  2. Individual per-target CSVs
  3. In-memory DataFrames

Missing values (NaN) indicate the compound was not tested against that target.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

TARGET_NAMES = ["InhA", "KatG", "rpoB", "DprE1", "MmpL3"]
CHEMBL_IDS = {
    "InhA": "CHEMBL1849",
    "KatG": "CHEMBL1790",
    "rpoB": "CHEMBL1916",
    "DprE1": "CHEMBL3622",
    "MmpL3": "CHEMBL4296",
}

# Approximate pIC50 activity threshold (IC50 ≤ 10 µM → pIC50 ≥ 5.0 is common;
# we use 6.5 = ~300 nM for selectivity-relevant hits)
ACTIVITY_THRESHOLD = 6.5


# ---------------------------------------------------------------------------
# MultiTaskDataset
# ---------------------------------------------------------------------------


class MultiTaskDataset:
    """Holds aligned SMILES + multi-target pIC50 / activity labels.

    Attributes
    ----------
    smiles   : list of canonical SMILES strings
    y_dict   : {target_name: np.ndarray (n,) float32, NaN = not tested}
    metadata : optional DataFrame with additional columns
    """

    def __init__(
        self,
        smiles: list[str],
        y_dict: dict[str, NDArray],
        metadata: pd.DataFrame | None = None,
    ) -> None:
        n = len(smiles)
        for t, arr in y_dict.items():
            if len(arr) != n:
                raise ValueError(f"y_dict['{t}'] length {len(arr)} != smiles length {n}")
        self.smiles = smiles
        self.y_dict = y_dict
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.smiles)

    @property
    def n_targets(self) -> int:
        return len(self.y_dict)

    @property
    def targets(self) -> list[str]:
        return list(self.y_dict.keys())

    def coverage(self) -> pd.DataFrame:
        """Return per-target data coverage statistics."""
        rows = []
        for t, arr in self.y_dict.items():
            mask = ~np.isnan(arr)
            binary = (arr[mask] >= ACTIVITY_THRESHOLD).astype(int) if mask.any() else np.array([])
            rows.append(
                {
                    "target": t,
                    "n_total": int(mask.sum()),
                    "n_active": int(binary.sum()) if len(binary) else 0,
                    "n_inactive": int((1 - binary).sum()) if len(binary) else 0,
                    "pct_coverage": 100.0 * mask.mean(),
                    "pct_active": 100.0 * binary.mean() if len(binary) else float("nan"),
                }
            )
        return pd.DataFrame(rows).set_index("target")

    def train_val_test_split(
        self,
        frac_train: float = 0.70,
        frac_val: float = 0.10,
        frac_test: float = 0.20,
        scaffold_split: bool = True,
        seed: int = 42,
    ) -> tuple["MultiTaskDataset", "MultiTaskDataset", "MultiTaskDataset"]:
        """Split dataset preserving scaffold diversity across splits."""
        n = len(self)
        if scaffold_split:
            indices = _scaffold_indices(self.smiles, frac_train, frac_val, seed)
        else:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n)
            n_train = int(n * frac_train)
            n_val = int(n * frac_val)
            indices = (
                perm[:n_train].tolist(),
                perm[n_train : n_train + n_val].tolist(),
                perm[n_train + n_val :].tolist(),
            )

        splits = []
        for idx in indices:
            idx_arr = list(idx)
            sub_smiles = [self.smiles[i] for i in idx_arr]
            sub_y = {t: arr[idx_arr] for t, arr in self.y_dict.items()}
            sub_meta = (
                self.metadata.iloc[idx_arr].reset_index(drop=True)
                if self.metadata is not None
                else None
            )
            splits.append(MultiTaskDataset(sub_smiles, sub_y, sub_meta))

        return splits[0], splits[1], splits[2]

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame({"smiles": self.smiles})
        for t, arr in self.y_dict.items():
            df[t] = arr
        if self.metadata is not None:
            for col in self.metadata.columns:
                if col not in df.columns:
                    df[col] = self.metadata[col].values
        return df

    def filter_min_targets(self, min_targets: int = 1) -> "MultiTaskDataset":
        """Keep only compounds tested in at least `min_targets` assays."""
        coverage_arr = np.stack(
            [~np.isnan(arr) for arr in self.y_dict.values()], axis=1
        ).sum(axis=1)
        keep = np.where(coverage_arr >= min_targets)[0].tolist()
        return MultiTaskDataset(
            [self.smiles[i] for i in keep],
            {t: arr[keep] for t, arr in self.y_dict.items()},
            self.metadata.iloc[keep].reset_index(drop=True) if self.metadata is not None else None,
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_multitask_csv(
    path: str | Path,
    smiles_col: str = "smiles",
    target_cols: list[str] | None = None,
    standardize: bool = True,
) -> MultiTaskDataset:
    """Load multi-target dataset from a single CSV.

    Expected columns: smiles, [InhA, KatG, rpoB, DprE1, MmpL3] (any subset).
    Activity values should be pIC50 (float). Missing → NaN.
    """
    df = pd.read_csv(path)
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in {path}")

    targets = target_cols or [t for t in TARGET_NAMES if t in df.columns]
    if not targets:
        raise ValueError("No target columns found. Expected one of: " + str(TARGET_NAMES))

    smiles_list = df[smiles_col].astype(str).tolist()

    if standardize:
        smiles_list = _standardize_smiles(smiles_list)

    y_dict: dict[str, NDArray] = {}
    for t in targets:
        y_dict[t] = df[t].astype(float).values.astype(np.float32)

    meta_cols = [c for c in df.columns if c not in [smiles_col] + targets]
    meta = df[meta_cols].reset_index(drop=True) if meta_cols else None

    logger.info(
        "Loaded %d compounds × %d targets from %s", len(smiles_list), len(targets), path
    )
    return MultiTaskDataset(smiles_list, y_dict, meta)


def load_multitask_from_dict(
    data: dict[str, pd.DataFrame],
    smiles_col: str = "smiles",
    activity_col: str = "pic50",
    standardize: bool = True,
) -> MultiTaskDataset:
    """Build a MultiTaskDataset from {target_name: per-target DataFrame}.

    Each DataFrame must have columns [smiles_col, activity_col].
    Molecules are aligned by canonical SMILES.
    """
    from rdkit import Chem

    all_smiles: set[str] = set()
    target_maps: dict[str, dict[str, float]] = {}

    for target, df in data.items():
        if smiles_col not in df.columns or activity_col not in df.columns:
            raise ValueError(f"DataFrame for {target} missing '{smiles_col}' or '{activity_col}'")
        mapping: dict[str, float] = {}
        for _, row in df.iterrows():
            smi = str(row[smiles_col])
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            can = Chem.MolToSmiles(mol)
            mapping[can] = float(row[activity_col])
            all_smiles.add(can)
        target_maps[target] = mapping
        logger.info("Target %s: %d compounds loaded", target, len(mapping))

    sorted_smiles = sorted(all_smiles)
    if standardize:
        sorted_smiles = _standardize_smiles(sorted_smiles)

    y_dict: dict[str, NDArray] = {}
    for target, mapping in target_maps.items():
        arr = np.array(
            [mapping.get(s, np.nan) for s in sorted_smiles], dtype=np.float32
        )
        y_dict[target] = arr

    logger.info(
        "Aligned dataset: %d unique molecules × %d targets",
        len(sorted_smiles),
        len(target_maps),
    )
    return MultiTaskDataset(sorted_smiles, y_dict)


def make_synthetic_multitask_dataset(
    n_compounds: int = 500,
    targets: list[str] | None = None,
    coverage: float = 0.6,
    seed: int = 42,
) -> MultiTaskDataset:
    """Generate a synthetic dataset for unit testing (no RDKit required).

    Uses simple bit-vector fingerprints and random pIC50 values.
    """
    rng = np.random.default_rng(seed)
    targets = targets or TARGET_NAMES

    # Generate fake SMILES using alkane chain variants
    smiles_list = [f"C{'C' * (i % 10 + 1)}N" for i in range(n_compounds)]

    y_dict: dict[str, NDArray] = {}
    for i, t in enumerate(targets):
        arr = rng.normal(6.0, 1.5, n_compounds).astype(np.float32)
        # Randomly mask ~(1-coverage) entries
        mask = rng.random(n_compounds) > coverage
        arr[mask] = np.nan
        y_dict[t] = arr

    return MultiTaskDataset(smiles_list, y_dict)


# ---------------------------------------------------------------------------
# Descriptor computation for MultiTaskDataset
# ---------------------------------------------------------------------------


def compute_descriptors(
    dataset: MultiTaskDataset,
    descriptor_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
) -> NDArray:
    """Compute fingerprint descriptors for all compounds in a MultiTaskDataset.

    Parameters
    ----------
    descriptor_type : 'morgan' | 'rdkit' | 'maccs'
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

        fps = []
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

        for smi in dataset.smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fps.append(np.zeros(n_bits, dtype=np.float32))
                continue
            if descriptor_type == "morgan":
                fp = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
            elif descriptor_type == "rdkit":
                from rdkit.Chem import RDKFingerprint

                fp_obj = RDKFingerprint(mol, fpSize=n_bits)
                arr = np.zeros(n_bits, dtype=np.float32)
                for bit in fp_obj.GetOnBits():
                    arr[bit] = 1.0
                fp = arr
            elif descriptor_type == "maccs":
                fp_obj = MACCSkeys.GenMACCSKeys(mol)
                fp = np.array(fp_obj, dtype=np.float32)
            else:
                raise ValueError(f"Unknown descriptor_type: {descriptor_type}")
            fps.append(fp)

        return np.stack(fps, axis=0)

    except ImportError:
        logger.warning("RDKit not available — using random descriptors for testing")
        rng = np.random.default_rng(42)
        return rng.random((len(dataset), n_bits)).astype(np.float32)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _standardize_smiles(smiles_list: list[str]) -> list[str]:
    """Canonicalize SMILES; keep original on failure."""
    try:
        from rdkit import Chem

        out = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            out.append(Chem.MolToSmiles(mol) if mol else smi)
        return out
    except ImportError:
        return smiles_list


def _scaffold_indices(
    smiles: list[str],
    frac_train: float,
    frac_val: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Scaffold-based split returning index lists (train, val, test)."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        scaffold_map: dict[str, list[int]] = {}
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                key = "__invalid__"
            else:
                key = MurckoScaffold.MurckoScaffoldSmilesFromMol(mol)
            scaffold_map.setdefault(key, []).append(i)

        groups = list(scaffold_map.values())
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)

        n = len(smiles)
        n_train = int(n * frac_train)
        n_val = int(n * frac_val)

        train_idx, val_idx, test_idx = [], [], []
        for g in groups:
            if len(train_idx) < n_train:
                train_idx.extend(g)
            elif len(val_idx) < n_val:
                val_idx.extend(g)
            else:
                test_idx.extend(g)

        return train_idx, val_idx, test_idx

    except ImportError:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(smiles)).tolist()
        n_train = int(len(smiles) * frac_train)
        n_val = int(len(smiles) * frac_val)
        return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]
