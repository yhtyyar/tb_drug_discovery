"""Scaffold-based train/test splitting for QSAR models.

Random splitting inflates QSAR metrics because structurally similar
molecules end up in both train and test sets. Scaffold split groups
molecules by their Bemis-Murcko scaffold, then assigns whole scaffold
groups to a single split — giving a realistic estimate of generalization
to chemically novel compounds.

References:
    Bemis & Murcko (1996) J. Med. Chem. 39(15):2887-2893
    Wu et al. (2018) MoleculeNet: A Benchmark for Molecular ML
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def get_scaffold(
    smiles: str,
    generic: bool = False,
    include_chirality: bool = True,
) -> Optional[str]:
    """Compute Bemis-Murcko scaffold for a SMILES string.

    Args:
        smiles: Input SMILES.
        generic: If True, return generic scaffold (all atoms → C).
        include_chirality: If True, treat enantiomers as distinct scaffolds.
            Chiral centres in the core ring system change binding geometry,
            so the default (True) is the chemically correct choice. Set False
            only when the dataset does not encode stereo information.

    Returns:
        Scaffold SMILES, or None if the molecule is invalid.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit required for scaffold split")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold_mol is None:
        return smiles  # acyclic molecules keep themselves

    if generic:
        scaffold_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)

    return Chem.MolToSmiles(scaffold_mol, isomericSmiles=include_chirality) or smiles


def scaffold_split(
    smiles_list: List[str],
    frac_train: float = 0.7,
    frac_val: float = 0.15,
    frac_test: float = 0.15,
    random_seed: int = 42,
    balanced: bool = True,
) -> Tuple[List[int], List[int], List[int]]:
    """Split dataset indices by Bemis-Murcko scaffold.

    Scaffolds are sorted by frequency (largest first) and greedily
    assigned to train, then val, then test until each reaches its
    target size.  This keeps structurally similar molecules together
    and avoids data leakage between splits.

    Args:
        smiles_list: List of SMILES strings.
        frac_train: Fraction for training set.
        frac_val: Fraction for validation set.
        frac_test: Fraction for test set.
        random_seed: Random seed for tie-breaking.
        balanced: Randomize scaffold assignment order within same-size groups.

    Returns:
        Tuple of (train_indices, val_indices, test_indices).

    Example:
        >>> train_idx, val_idx, test_idx = scaffold_split(smiles, 0.7, 0.15, 0.15)
        >>> print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6, \
        "Fractions must sum to 1.0"

    n = len(smiles_list)
    rng = random.Random(random_seed)

    # Build scaffold → molecule index mapping
    scaffold_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaffold = get_scaffold(smi)
        if scaffold is None:
            scaffold = f"__invalid_{i}__"  # isolate broken molecules
        scaffold_to_indices[scaffold].append(i)

    # Sort scaffolds by cluster size descending for greedy assignment
    scaffold_groups = list(scaffold_to_indices.values())
    if balanced:
        # Shuffle within same-size groups to avoid ordering bias
        scaffold_groups = sorted(scaffold_groups, key=lambda g: (-len(g), rng.random()))
    else:
        scaffold_groups = sorted(scaffold_groups, key=lambda g: -len(g))

    train_cutoff = frac_train * n
    val_cutoff = (frac_train + frac_val) * n

    train_idx, val_idx, test_idx = [], [], []

    for group in scaffold_groups:
        if len(train_idx) + len(group) <= train_cutoff:
            train_idx.extend(group)
        elif len(train_idx) + len(val_idx) + len(group) <= val_cutoff:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    logger.info(
        f"Scaffold split: {len(train_idx)} train, {len(val_idx)} val, "
        f"{len(test_idx)} test | "
        f"{len(scaffold_to_indices)} unique scaffolds"
    )

    return train_idx, val_idx, test_idx


def scaffold_split_df(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    frac_train: float = 0.7,
    frac_val: float = 0.15,
    frac_test: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scaffold split on a DataFrame.

    Args:
        df: DataFrame with a SMILES column.
        smiles_col: Name of the SMILES column.
        frac_train: Training fraction.
        frac_val: Validation fraction.
        frac_test: Test fraction.
        random_seed: Random seed.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    smiles = df[smiles_col].tolist()
    train_idx, val_idx, test_idx = scaffold_split(
        smiles, frac_train, frac_val, frac_test, random_seed
    )

    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def scaffold_k_fold(
    smiles_list: List[str],
    n_folds: int = 5,
    random_seed: int = 42,
) -> List[Tuple[List[int], List[int]]]:
    """Scaffold-aware k-fold cross-validation.

    Assigns scaffold groups round-robin across k folds to ensure each
    fold has a disjoint set of scaffolds.

    Args:
        smiles_list: List of SMILES strings.
        n_folds: Number of folds.
        random_seed: Random seed.

    Returns:
        List of (train_indices, test_indices) tuples, one per fold.
    """
    rng = random.Random(random_seed)

    scaffold_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaffold = get_scaffold(smi) or f"__invalid_{i}__"
        scaffold_to_indices[scaffold].append(i)

    # Shuffle and sort by size for balanced folds
    groups = sorted(scaffold_to_indices.values(), key=lambda g: (-len(g), rng.random()))

    # Assign groups round-robin to folds
    fold_buckets: List[List[int]] = [[] for _ in range(n_folds)]
    for i, group in enumerate(groups):
        fold_buckets[i % n_folds].extend(group)

    splits = []
    for fold_idx in range(n_folds):
        test_idx = fold_buckets[fold_idx]
        train_idx = [
            idx for i, bucket in enumerate(fold_buckets)
            if i != fold_idx
            for idx in bucket
        ]
        splits.append((train_idx, test_idx))

    fold_sizes = [len(fold_buckets[i]) for i in range(n_folds)]
    logger.info(
        f"Scaffold {n_folds}-fold: sizes = {fold_sizes}, "
        f"scaffolds = {len(scaffold_to_indices)}"
    )

    return splits
