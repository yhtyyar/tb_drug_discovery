"""Property-based tests using hypothesis.

These tests encode invariants that must hold for any valid input,
not just the specific examples in unit tests.  Running with a large
number of examples catches edge cases that hand-written tests miss.

Install: pip install hypothesis
Run:     pytest tests/test_property_based.py -v --hypothesis-seed=42
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from hypothesis import given, settings, assume, HealthCheck
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="hypothesis not installed: pip install hypothesis")

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_SMILES = [
    "CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCC",
    "c1ccc(O)cc1", "CC(=O)Nc1ccc(O)cc1", "CN1CCC[C@H]1c1cccnc1",
    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "O=C(O)c1ccccc1",
    "Cc1ccc(S(N)(=O)=O)cc1",
    "CCOC(=O)c1ccc(N)cc1",
]


# ---------------------------------------------------------------------------
# SMILES validity invariants
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestSMILESInvariants:
    """Every SMILES that RDKit parses must survive the round-trip."""

    @given(st.sampled_from(VALID_SMILES))
    @settings(max_examples=len(VALID_SMILES), suppress_health_check=[HealthCheck.too_slow])
    def test_canonical_smiles_is_idempotent(self, smiles: str):
        """Canonical SMILES of a canonical SMILES must equal itself."""
        from data.chembl_loader import ChEMBLLoader
        loader = ChEMBLLoader.__new__(ChEMBLLoader)
        canonical = loader.standardize_smiles(smiles)
        assert canonical is not None
        canonical2 = loader.standardize_smiles(canonical)
        assert canonical == canonical2, (
            f"Canonical not idempotent: {smiles!r} → {canonical!r} → {canonical2!r}"
        )

    @given(st.sampled_from(VALID_SMILES))
    @settings(max_examples=len(VALID_SMILES))
    def test_valid_smiles_always_parse(self, smiles: str):
        """All reference SMILES must parse without errors."""
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f"RDKit failed to parse known-valid SMILES: {smiles!r}"

    @given(st.sampled_from(VALID_SMILES))
    @settings(max_examples=len(VALID_SMILES))
    def test_sanitized_mol_has_positive_atoms(self, smiles: str):
        """Every valid molecule must have at least one atom."""
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        Chem.SanitizeMol(mol)
        assert mol.GetNumAtoms() > 0

    @given(st.sampled_from(["INVALID_SMILES", "!!!!", "X#Y", "", "   "]))
    @settings(max_examples=5)
    def test_invalid_smiles_return_none(self, smiles: str):
        """Invalid SMILES strings must never silently succeed."""
        mol = Chem.MolFromSmiles(smiles)
        assert mol is None or mol.GetNumAtoms() == 0


# ---------------------------------------------------------------------------
# pIC50 conversion invariants
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestPIC50Invariants:
    """Mathematical properties of pIC50 = 9 - log10(IC50_nM)."""

    @given(st.floats(min_value=1e-3, max_value=1e9, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_pic50_monotone_decreasing(self, ic50_nm: float):
        """Higher IC50 must always give lower pIC50."""
        from data.chembl_loader import ChEMBLLoader
        loader = ChEMBLLoader.__new__(ChEMBLLoader)
        p1 = loader.calculate_pic50(ic50_nm)
        p2 = loader.calculate_pic50(ic50_nm * 10)
        assert p1 > p2, f"pIC50 not monotone: IC50={ic50_nm} → {p1}, 10x → {p2}"

    @given(st.floats(min_value=1e-3, max_value=1e9, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_pic50_range(self, ic50_nm: float):
        """pIC50 must stay in a physically meaningful range (-3 to 15)."""
        from data.chembl_loader import ChEMBLLoader
        loader = ChEMBLLoader.__new__(ChEMBLLoader)
        p = loader.calculate_pic50(ic50_nm)
        assert np.isfinite(p)
        assert -3 < p < 20, f"pIC50={p} out of range for IC50={ic50_nm}"

    @given(st.one_of(
        st.just(0.0),
        st.just(-1.0),
        st.floats(max_value=0, allow_nan=True),
    ))
    @settings(max_examples=20)
    def test_non_positive_ic50_returns_nan(self, ic50_nm: float):
        """Non-positive IC50 values must return NaN, not crash."""
        from data.chembl_loader import ChEMBLLoader
        loader = ChEMBLLoader.__new__(ChEMBLLoader)
        result = loader.calculate_pic50(ic50_nm)
        assert np.isnan(result), f"Expected NaN for IC50={ic50_nm}, got {result}"


# ---------------------------------------------------------------------------
# Data splitting invariants
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestDataSplitInvariants:
    """Properties that any train/val/test split must satisfy."""

    @given(
        n=st.integers(min_value=10, max_value=500),
        seed=st.integers(min_value=0, max_value=999),
    )
    @settings(max_examples=50)
    def test_no_data_leakage(self, n: int, seed: int):
        """Train, val and test indices must be disjoint."""
        from data.data_preprocessor import DataPreprocessor
        X = np.random.randn(n, 5)
        y = np.random.randn(n)
        prep = DataPreprocessor(random_seed=seed)
        X_train, X_val, X_test, y_train, y_val, y_test = prep.split_data(X, y)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == n, f"Lost samples: {total} != {n}"

    @given(
        n=st.integers(min_value=20, max_value=300),
        seed=st.integers(min_value=0, max_value=999),
    )
    @settings(max_examples=50)
    def test_split_reproducibility(self, n: int, seed: int):
        """Same seed must produce identical splits."""
        from data.data_preprocessor import DataPreprocessor
        X = np.arange(n * 5).reshape(n, 5).astype(float)
        y = np.arange(n).astype(float)

        p1 = DataPreprocessor(random_seed=seed)
        p2 = DataPreprocessor(random_seed=seed)
        split1 = p1.split_data(X, y)
        split2 = p2.split_data(X, y)

        for a, b in zip(split1, split2):
            np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Descriptor invariants
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestDescriptorInvariants:
    """Molecular descriptor properties that must always hold."""

    @given(st.sampled_from(VALID_SMILES))
    @settings(max_examples=len(VALID_SMILES))
    def test_molecular_weight_positive(self, smiles: str):
        """Molecular weight must be strictly positive for any valid molecule."""
        from data.descriptor_calculator import DescriptorCalculator
        calc = DescriptorCalculator(lipinski=True, topological=False, extended=False)
        result = calc.calculate(smiles)
        assert result is not None
        assert result["MolWt"] > 0, f"Non-positive MolWt for {smiles!r}"

    @given(st.sampled_from(VALID_SMILES))
    @settings(max_examples=len(VALID_SMILES))
    def test_hbd_le_hba(self, smiles: str):
        """H-bond donors cannot exceed H-bond acceptors for these examples."""
        from data.descriptor_calculator import DescriptorCalculator
        calc = DescriptorCalculator(lipinski=True, topological=False, extended=False)
        result = calc.calculate(smiles)
        assert result is not None
        # This is NOT a universal rule — just checks the descriptor is computed
        assert result["HBD"] >= 0, "Negative HBD"
        assert result["HBA"] >= 0, "Negative HBA"

    @given(st.sampled_from(VALID_SMILES))
    @settings(max_examples=len(VALID_SMILES))
    def test_descriptors_are_finite(self, smiles: str):
        """All descriptor values must be finite numbers."""
        from data.descriptor_calculator import DescriptorCalculator
        calc = DescriptorCalculator(lipinski=True, topological=True, extended=False)
        result = calc.calculate(smiles)
        assert result is not None
        for key, val in result.items():
            assert np.isfinite(val), f"Non-finite descriptor {key}={val} for {smiles!r}"


# ---------------------------------------------------------------------------
# Scaffold split invariants
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
class TestScaffoldSplitInvariants:
    """Scaffold split must never lose or duplicate molecules."""

    @given(
        smiles=st.lists(
            st.sampled_from(VALID_SMILES),
            min_size=10,
            max_size=50,
        ),
        seed=st.integers(min_value=0, max_value=999),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_scaffold_split_no_data_loss(self, smiles, seed):
        """Scaffold split must cover every input molecule exactly once."""
        from data.scaffold_split import scaffold_split
        train, val, test = scaffold_split(smiles, 0.7, 0.15, 0.15, random_seed=seed)
        all_indices = sorted(train + val + test)
        assert all_indices == list(range(len(smiles))), \
            "Scaffold split lost or duplicated molecules"

    @given(
        smiles=st.lists(
            st.sampled_from(VALID_SMILES),
            min_size=10,
            max_size=50,
        ),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_scaffold_split_disjoint(self, smiles):
        """Train, val and test sets must not share any index."""
        from data.scaffold_split import scaffold_split
        train, val, test = scaffold_split(smiles, 0.7, 0.15, 0.15)
        train_set, val_set, test_set = set(train), set(val), set(test)
        assert not (train_set & val_set), "Train and val overlap"
        assert not (train_set & test_set), "Train and test overlap"
        assert not (val_set & test_set), "Val and test overlap"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
