"""Tests for 3D conformer generation and descriptor calculation."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.conformer_generator import (
    ConformerConfig,
    Descriptor3DCalculator,
    Descriptor3DConfig,
    augment_with_3d,
    compute_autocorr3d,
    compute_e3fp_fingerprint,
    compute_pmi_descriptors,
    compute_whim_descriptors,
    generate_conformers,
    get_lowest_energy_conformer_id,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(c1=O)C"
INVALID_SMILES = "this_is_not_smiles"

# Skip all tests if RDKit is not installed
rdkit = pytest.importorskip("rdkit", reason="RDKit not available")


# ---------------------------------------------------------------------------
# Conformer generation
# ---------------------------------------------------------------------------


class TestConformerGeneration:
    def test_valid_smiles_returns_mol(self) -> None:
        mol = generate_conformers(ASPIRIN, ConformerConfig(n_confs=1, random_seed=0))
        assert mol is not None

    def test_mol_has_conformers(self) -> None:
        mol = generate_conformers(ASPIRIN, ConformerConfig(n_confs=3, random_seed=0))
        assert mol is not None
        assert mol.GetNumConformers() >= 1

    def test_invalid_smiles_returns_none(self) -> None:
        mol = generate_conformers(INVALID_SMILES)
        assert mol is None

    def test_lowest_energy_conf_id(self) -> None:
        mol = generate_conformers(ASPIRIN, ConformerConfig(n_confs=3, random_seed=0))
        assert mol is not None
        conf_id = get_lowest_energy_conformer_id(mol)
        assert conf_id >= 0


# ---------------------------------------------------------------------------
# PMI descriptors
# ---------------------------------------------------------------------------


class TestPMIDescriptors:
    @pytest.fixture()
    def aspirin_mol(self):
        return generate_conformers(ASPIRIN, ConformerConfig(n_confs=1, random_seed=0))

    def test_shape(self, aspirin_mol) -> None:
        desc = compute_pmi_descriptors(aspirin_mol)
        assert desc.shape == (6,)

    def test_dtype(self, aspirin_mol) -> None:
        desc = compute_pmi_descriptors(aspirin_mol)
        assert desc.dtype == np.float32

    def test_no_nan(self, aspirin_mol) -> None:
        desc = compute_pmi_descriptors(aspirin_mol)
        assert not np.isnan(desc).any()

    def test_npr_range(self, aspirin_mol) -> None:
        desc = compute_pmi_descriptors(aspirin_mol)
        # npr1, npr2 must be in [0, 1]
        assert 0 <= desc[0] <= 1.0
        assert 0 <= desc[1] <= 1.0


# ---------------------------------------------------------------------------
# WHIM descriptors
# ---------------------------------------------------------------------------


class TestWHIMDescriptors:
    @pytest.fixture()
    def caffeine_mol(self):
        return generate_conformers(CAFFEINE, ConformerConfig(n_confs=1, random_seed=0))

    def test_shape(self, caffeine_mol) -> None:
        desc = compute_whim_descriptors(caffeine_mol)
        assert desc.shape == (114,)

    def test_dtype(self, caffeine_mol) -> None:
        desc = compute_whim_descriptors(caffeine_mol)
        assert desc.dtype == np.float32

    def test_not_all_zero(self, caffeine_mol) -> None:
        desc = compute_whim_descriptors(caffeine_mol)
        assert desc.sum() != 0.0


# ---------------------------------------------------------------------------
# Autocorr3D
# ---------------------------------------------------------------------------


class TestAutocorr3D:
    @pytest.fixture()
    def mol(self):
        return generate_conformers(ASPIRIN, ConformerConfig(n_confs=1, random_seed=0))

    def test_shape(self, mol) -> None:
        desc = compute_autocorr3d(mol)
        assert desc.shape == (80,)

    def test_dtype(self, mol) -> None:
        desc = compute_autocorr3d(mol)
        assert desc.dtype == np.float32


# ---------------------------------------------------------------------------
# E3FP fingerprint
# ---------------------------------------------------------------------------


class TestE3FP:
    @pytest.fixture()
    def mol(self):
        return generate_conformers(CAFFEINE, ConformerConfig(n_confs=1, random_seed=0))

    def test_shape(self, mol) -> None:
        fp = compute_e3fp_fingerprint(mol, n_bits=512)
        assert fp.shape == (512,)

    def test_binary(self, mol) -> None:
        fp = compute_e3fp_fingerprint(mol, n_bits=512)
        assert set(np.unique(fp)).issubset({0.0, 1.0})

    def test_not_all_zero(self, mol) -> None:
        fp = compute_e3fp_fingerprint(mol, n_bits=512)
        assert fp.sum() > 0


# ---------------------------------------------------------------------------
# Descriptor3DCalculator
# ---------------------------------------------------------------------------


class TestDescriptor3DCalculator:
    @pytest.fixture()
    def calc(self) -> Descriptor3DCalculator:
        return Descriptor3DCalculator(
            conf_config=ConformerConfig(n_confs=1, random_seed=0),
            desc_config=Descriptor3DConfig(e3fp_bits=256),
        )

    def test_n_features(self, calc: Descriptor3DCalculator) -> None:
        expected = 6 + 114 + 80 + 256
        assert calc.n_features == expected

    def test_feature_names_length(self, calc: Descriptor3DCalculator) -> None:
        assert len(calc.feature_names) == calc.n_features

    def test_compute_single_shape(self, calc: Descriptor3DCalculator) -> None:
        desc = calc.compute_single(ASPIRIN)
        assert desc.shape == (calc.n_features,)

    def test_compute_single_dtype(self, calc: Descriptor3DCalculator) -> None:
        desc = calc.compute_single(ASPIRIN)
        assert desc.dtype == np.float32

    def test_invalid_smiles_returns_zeros(self, calc: Descriptor3DCalculator) -> None:
        desc = calc.compute_single(INVALID_SMILES)
        assert desc.shape == (calc.n_features,)
        assert (desc == 0).all()

    def test_compute_batch_shape(self, calc: Descriptor3DCalculator) -> None:
        smiles = [ASPIRIN, CAFFEINE, INVALID_SMILES]
        X = calc.compute_batch(smiles)
        assert X.shape == (3, calc.n_features)

    def test_pmi_only(self) -> None:
        calc = Descriptor3DCalculator(
            conf_config=ConformerConfig(n_confs=1),
            desc_config=Descriptor3DConfig(
                include_pmi=True,
                include_whim=False,
                include_autocorr=False,
                include_e3fp=False,
            ),
        )
        assert calc.n_features == 6
        desc = calc.compute_single(ASPIRIN)
        assert desc.shape == (6,)

    def test_different_mols_differ(self, calc: Descriptor3DCalculator) -> None:
        d1 = calc.compute_single(ASPIRIN)
        d2 = calc.compute_single(CAFFEINE)
        assert not np.allclose(d1, d2)


# ---------------------------------------------------------------------------
# augment_with_3d
# ---------------------------------------------------------------------------


class TestAugmentWith3D:
    def test_shape(self) -> None:
        smiles = [ASPIRIN, CAFFEINE]
        X_2d = np.random.rand(2, 10).astype(np.float32)
        calc_dims = 6 + 114 + 80 + 1024
        X_aug = augment_with_3d(X_2d, smiles)
        assert X_aug.shape == (2, 10 + calc_dims)

    def test_2d_part_preserved(self) -> None:
        smiles = [ASPIRIN]
        X_2d = np.ones((1, 5), dtype=np.float32) * 3.14
        X_aug = augment_with_3d(X_2d, smiles)
        np.testing.assert_allclose(X_aug[:, :5], X_2d)
