"""Tests for scaffold-based data splitting.

Ensures that scaffold split properly separates structurally similar molecules
to prevent data leakage and inflated metrics.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.scaffold_split import get_scaffold, scaffold_split, scaffold_k_fold


# Test data: benzene derivatives should share same scaffold
BENZENE_DERIVATIVES = [
    "c1ccccc1",           # benzene
    "c1ccc(F)cc1",        # fluorobenzene
    "c1ccc(Cl)cc1",       # chlorobenzene
    "c1ccc(Br)cc1",       # bromobenzene
    "c1ccc(C)cc1",        # toluene
]

# Ethanol derivatives
ETHANOL_DERIVATIVES = [
    "CCO",                # ethanol
    "CC(=O)O",            # acetic acid
    "CCN",                # ethylamine
    "CCCl",               # chloroethane
]

# Diverse scaffolds
DIVERSE_SMILES = BENZENE_DERIVATIVES + ETHANOL_DERIVATIVES + [
    "C1CCCCC1",           # cyclohexane
    "C1CCOCC1",           # tetrahydropyran
]


def test_get_scaffold_basic():
    """Test basic scaffold extraction."""
    scaffold = get_scaffold("c1ccccc1")
    assert scaffold is not None
    assert isinstance(scaffold, str)


def test_benzene_family_same_scaffold():
    """Benzene derivatives should have the same scaffold."""
    scaffolds = {get_scaffold(smi) for smi in BENZENE_DERIVATIVES}
    assert len(scaffolds) == 1, f"Expected 1 scaffold, got {len(scaffolds)}: {scaffolds}"


def test_ethanol_family_same_scaffold():
    """Ethanol derivatives should have the same scaffold."""
    scaffolds = {get_scaffold(smi) for smi in ETHANOL_DERIVATIVES}
    assert len(scaffolds) == 1, f"Expected 1 scaffold, got {len(scaffolds)}: {scaffolds}"


def test_different_families_different_scaffolds():
    """Benzene and ethanol should have different scaffolds."""
    benzene_scaffold = get_scaffold("c1ccccc1")
    ethanol_scaffold = get_scaffold("CCO")
    assert benzene_scaffold != ethanol_scaffold


def test_scaffold_split_sizes():
    """Test that split produces expected approximate sizes."""
    # Repeat each SMILES 10 times to get enough samples
    smiles_list = DIVERSE_SMILES * 10

    train_idx, val_idx, test_idx = scaffold_split(
        smiles_list,
        frac_train=0.7,
        frac_val=0.15,
        frac_test=0.15,
        random_seed=42,
    )

    n_total = len(smiles_list)

    # Check approximate sizes (allowing margin for discrete scaffold groups)
    assert abs(len(train_idx) / n_total - 0.7) < 0.15
    assert abs(len(val_idx) / n_total - 0.15) < 0.15
    assert abs(len(test_idx) / n_total - 0.15) < 0.15

    # Check no overlap
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0

    # Check all indices are covered
    all_idx = set(train_idx) | set(val_idx) | set(test_idx)
    assert len(all_idx) == n_total


def test_scaffold_split_no_data_leakage():
    """Ensure structurally similar molecules don't leak across splits."""
    # Create dataset with multiple benzene derivatives
    smiles_list = [
        "c1ccccc1", "c1ccc(F)cc1", "c1ccc(Cl)cc1",  # benzene family
        "CCO", "CC(=O)O", "CCN",                      # ethanol family
        "C1CCCCC1", "C1CCCCC1C",                      # cyclohexane family
    ] * 5

    train_idx, val_idx, test_idx = scaffold_split(
        smiles_list,
        frac_train=0.6,
        frac_val=0.2,
        frac_test=0.2,
        random_seed=42,
    )

    # Get scaffolds for each split
    train_scaffolds = {get_scaffold(smiles_list[i]) for i in train_idx}
    val_scaffolds = {get_scaffold(smiles_list[i]) for i in val_idx}
    test_scaffolds = {get_scaffold(smiles_list[i]) for i in test_idx}

    # No scaffold should appear in multiple splits
    assert len(train_scaffolds & val_scaffolds) == 0, "Data leakage: scaffolds in both train and val"
    assert len(train_scaffolds & test_scaffolds) == 0, "Data leakage: scaffolds in both train and test"
    assert len(val_scaffolds & test_scaffolds) == 0, "Data leakage: scaffolds in both val and test"


def test_scaffold_k_fold():
    """Test k-fold scaffold split."""
    smiles_list = DIVERSE_SMILES * 5

    splits = scaffold_k_fold(smiles_list, n_folds=3, random_seed=42)

    assert len(splits) == 3

    # Check that all indices are covered across all folds
    all_test_indices = set()
    for train_idx, test_idx in splits:
        # No overlap within fold
        assert len(set(train_idx) & set(test_idx)) == 0
        all_test_indices.update(test_idx)

    # All samples should appear in test at least once
    assert all_test_indices == set(range(len(smiles_list)))


def test_scaffold_k_fold_no_scaffold_leakage():
    """Ensure k-fold doesn't leak scaffolds across folds."""
    smiles_list = BENZENE_DERIVATIVES * 3 + ETHANOL_DERIVATIVES * 3

    splits = scaffold_k_fold(smiles_list, n_folds=2, random_seed=42)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_scaffolds = {get_scaffold(smiles_list[i]) for i in train_idx}
        test_scaffolds = {get_scaffold(smiles_list[i]) for i in test_idx}

        # No scaffold overlap in each fold
        overlap = train_scaffolds & test_scaffolds
        assert len(overlap) == 0, f"Fold {fold_idx}: scaffolds leaked: {overlap}"


def test_scaffold_split_reproducibility():
    """Test that same seed gives same split."""
    smiles_list = DIVERSE_SMILES * 10

    train1, val1, test1 = scaffold_split(
        smiles_list, frac_train=0.7, frac_val=0.15, frac_test=0.15, random_seed=42
    )
    train2, val2, test2 = scaffold_split(
        smiles_list, frac_train=0.7, frac_val=0.15, frac_test=0.15, random_seed=42
    )

    assert train1 == train2
    assert val1 == val2
    assert test1 == test2


def test_scaffold_split_different_seeds():
    """Test that different seeds give different splits."""
    smiles_list = DIVERSE_SMILES * 10

    train1, val1, test1 = scaffold_split(
        smiles_list, frac_train=0.7, frac_val=0.15, frac_test=0.15, random_seed=42
    )
    train2, val2, test2 = scaffold_split(
        smiles_list, frac_train=0.7, frac_val=0.15, frac_test=0.15, random_seed=123
    )

    # At least one split should differ
    assert train1 != train2 or val1 != val2 or test1 != test2


def test_invalid_smiles_handling():
    """Test handling of invalid SMILES."""
    # Include some invalid SMILES
    smiles_list = ["c1ccccc1", "INVALID", "CCO", "BAD_SMILES", "c1ccc(F)cc1"]

    # Should not raise exception
    train_idx, val_idx, test_idx = scaffold_split(
        smiles_list,
        frac_train=0.6,
        frac_val=0.2,
        frac_test=0.2,
    )

    # All valid indices should be assigned
    valid_indices = {0, 2, 4}  # indices of valid SMILES
    all_assigned = set(train_idx) | set(val_idx) | set(test_idx)
    assert valid_indices.issubset(all_assigned)
