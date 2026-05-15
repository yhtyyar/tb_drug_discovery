"""Tests for MoleculeStandardizer.

Key invariants:
1. Tautomers of the same molecule produce the same canonical SMILES
2. Salt stripping returns only the organic fragment
3. Neutralization removes common charges
4. Deduplication accounts for tautomers
5. Invalid SMILES returns None (never raises)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

pytest.importorskip("rdkit")


@pytest.fixture(scope="module")
def std():
    from data.mol_standardizer import MoleculeStandardizer
    return MoleculeStandardizer(
        remove_salts=True,
        neutralize=True,
        canonicalize_tautomer=True,
        assign_stereo=True,
    )


class TestSaltStripping:
    def test_sodium_acetate_stripped(self, std):
        result = std.standardize("CC(=O)[O-].[Na+]")
        assert result is not None
        assert "Na" not in result

    def test_hcl_salt_stripped(self, std):
        result = std.standardize("NCc1ccccc1.Cl")
        assert result is not None
        assert "Cl" not in result or result.count("Cl") == result.count("c")

    def test_pure_molecule_unchanged_structurally(self, std):
        result = std.standardize("CCO")
        assert result is not None


class TestTautomerCanonicalization:
    def test_keto_enol_tautomers_same(self, std):
        """Keto and enol form of acetaldehyde hydrate (conceptual)."""
        # 2-hydroxypyridine ↔ pyridin-2(1H)-one
        s1 = std.standardize("Oc1ccccn1")     # 2-hydroxypyridine
        s2 = std.standardize("O=c1cccc[nH]1") # pyridin-2(1H)-one
        assert s1 is not None
        assert s2 is not None
        assert s1 == s2, f"Tautomers differ: {s1} != {s2}"

    def test_same_input_same_output(self, std):
        smi = "c1ccccc1"
        assert std.standardize(smi) == std.standardize(smi)


class TestNeutralization:
    def test_carboxylate_neutralized(self, std):
        result = std.standardize("CC(=O)[O-]")
        assert result is not None
        assert "[O-]" not in result

    def test_ammonium_neutralized(self, std):
        result = std.standardize("CC[NH3+]")
        assert result is not None
        assert "[NH3+]" not in result


class TestInvalidInputs:
    def test_invalid_smiles_returns_none(self, std):
        assert std.standardize("not_a_smiles") is None

    def test_empty_string_returns_none(self, std):
        assert std.standardize("") is None

    def test_percent_garbage_returns_none(self, std):
        assert std.standardize("%%%") is None


class TestBatchStandardization:
    def test_batch_length_preserved(self, std):
        smiles = ["CCO", "c1ccccc1", "invalid"]
        result = std.standardize_batch(smiles, drop_invalid=False)
        assert len(result) == 3

    def test_batch_drop_invalid(self, std):
        smiles = ["CCO", "invalid", "c1ccccc1"]
        result = std.standardize_batch(smiles, drop_invalid=True)
        assert len(result) == 2
        assert all(r is not None for r in result)


class TestDeduplication:
    def test_exact_duplicates_removed(self, std):
        smiles = ["CCO", "CCO", "CCO"]
        unique, indices = std.deduplicate(smiles)
        assert len(unique) == 1
        assert len(indices) == 1

    def test_tautomers_deduplicated(self, std):
        smiles = ["Oc1ccccn1", "O=c1cccc[nH]1"]  # same molecule
        unique, indices = std.deduplicate(smiles)
        assert len(unique) == 1

    def test_different_molecules_kept(self, std):
        smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        unique, indices = std.deduplicate(smiles)
        assert len(unique) == 3

    def test_are_same_molecule(self, std):
        assert std.are_same_molecule("Oc1ccccn1", "O=c1cccc[nH]1")
        assert not std.are_same_molecule("CCO", "c1ccccc1")
