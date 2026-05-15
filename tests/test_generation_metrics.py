"""Tests for molecular generation quality metrics."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")

VALID_MOLS = [
    "CCO", "c1ccccc1", "CC(=O)O", "CCN", "c1ccc(O)cc1",
    "CC(=O)Nc1ccc(O)cc1", "CN1CCC[C@H]1c1cccnc1",
    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
]
INVALID_MOLS = ["INVALID", "!!!!", "", "X#Y"]


class TestValidity:
    def test_all_valid(self):
        from evaluation.generation_metrics import compute_validity
        validity, valid = compute_validity(VALID_MOLS)
        assert validity == 1.0
        assert len(valid) == len(VALID_MOLS)

    def test_all_invalid(self):
        from evaluation.generation_metrics import compute_validity
        validity, valid = compute_validity(INVALID_MOLS)
        assert validity == 0.0
        assert valid == []

    def test_mixed(self):
        from evaluation.generation_metrics import compute_validity
        mixed = VALID_MOLS[:4] + INVALID_MOLS[:4]
        validity, valid = compute_validity(mixed)
        assert 0.0 < validity < 1.0
        assert len(valid) == 4

    def test_empty_list(self):
        from evaluation.generation_metrics import compute_validity
        validity, valid = compute_validity([])
        assert validity == 0.0
        assert valid == []


class TestUniqueness:
    def test_all_unique(self):
        from evaluation.generation_metrics import compute_uniqueness
        uniqueness, unique = compute_uniqueness(VALID_MOLS)
        assert uniqueness == 1.0

    def test_duplicates_counted(self):
        from evaluation.generation_metrics import compute_uniqueness
        duped = ["CCO", "CCO", "CCO", "c1ccccc1"]
        uniqueness, unique = compute_uniqueness(duped)
        assert uniqueness == pytest.approx(2 / 4)
        assert len(unique) == 2


class TestNovelty:
    def test_all_novel(self):
        from evaluation.generation_metrics import compute_novelty
        generated = ["CCO", "c1ccccc1"]
        training = {"CCCCO", "CCCCN"}
        novelty, novel = compute_novelty(generated, training)
        assert novelty == 1.0

    def test_none_novel(self):
        from evaluation.generation_metrics import compute_novelty
        generated = ["CCO", "c1ccccc1"]
        training = {"CCO", "c1ccccc1"}
        novelty, novel = compute_novelty(generated, training)
        assert novelty == 0.0
        assert novel == []


class TestQED:
    def test_qed_in_range(self):
        from evaluation.generation_metrics import compute_qed_distribution
        stats = compute_qed_distribution(VALID_MOLS)
        assert 0.0 <= stats["qed_mean"] <= 1.0
        assert 0.0 <= stats["qed_median"] <= 1.0

    def test_qed_paracetamol_reasonable(self):
        from evaluation.generation_metrics import compute_qed_distribution
        stats = compute_qed_distribution(["CC(=O)Nc1ccc(O)cc1"])  # Paracetamol
        assert stats["qed_mean"] > 0.5, "Paracetamol QED should be > 0.5"


class TestDiversity:
    def test_identical_molecules_zero_diversity(self):
        from evaluation.generation_metrics import compute_diversity
        diversity = compute_diversity(["CCO", "CCO", "CCO"])
        assert diversity == pytest.approx(0.0, abs=0.05)

    def test_diverse_molecules_high_score(self):
        from evaluation.generation_metrics import compute_diversity
        diverse = [
            "CCO", "c1ccccc1CC(=O)O",
            "CC(C)(C)c1ccc(O)cc1",
            "O=C1CCCCC1", "c1ccc2ncccc2c1",
        ]
        diversity = compute_diversity(diverse)
        assert diversity > 0.3, f"Expected diversity > 0.3, got {diversity:.3f}"


class TestLipinski:
    def test_ethanol_passes_ro5(self):
        from evaluation.generation_metrics import compute_lipinski_compliance
        result = compute_lipinski_compliance(["CCO"])
        assert result["ro5_pass_rate"] == 1.0

    def test_pass_rate_in_range(self):
        from evaluation.generation_metrics import compute_lipinski_compliance
        result = compute_lipinski_compliance(VALID_MOLS)
        assert 0.0 <= result["ro5_pass_rate"] <= 1.0


class TestEvaluateGeneration:
    def test_full_pipeline(self):
        from evaluation.generation_metrics import evaluate_generation
        training = VALID_MOLS[:3]
        generated = VALID_MOLS + INVALID_MOLS
        metrics = evaluate_generation(generated, training, verbose=False)

        assert "validity" in metrics
        assert "uniqueness" in metrics
        assert "novelty" in metrics
        assert "qed_mean" in metrics
        assert "diversity" in metrics
        assert "ro5_pass_rate" in metrics

        assert 0.0 <= metrics["validity"] <= 1.0
        assert metrics["n_generated"] == len(generated)

    def test_no_training_set(self):
        from evaluation.generation_metrics import evaluate_generation
        metrics = evaluate_generation(VALID_MOLS, training_smiles=None, verbose=False)
        assert np.isnan(metrics["novelty"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
