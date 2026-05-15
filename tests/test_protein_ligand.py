"""Tests for protein-ligand interaction features."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.protein_ligand import (
    PLIFConfig,
    ProteinLigandFeaturizer,
    compute_ligand_pharmacophore,
    pharmacophoric_overlap_score,
    PHARMACOPHORE_FEATURE_NAMES,
    TB_POCKET_PHARMACOPHORES,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2N(C)C(=O)N(C)C(=O)c12"
INVALID = "not_a_smiles"


class TestLigandPharmacophore:
    def test_returns_dict(self) -> None:
        d = compute_ligand_pharmacophore(ASPIRIN)
        assert isinstance(d, dict)
        assert "n_hba" in d

    def test_invalid_returns_zeros(self) -> None:
        d = compute_ligand_pharmacophore(INVALID)
        assert d["n_hba"] == 0


class TestPharmacophoreOverlap:
    def test_shape(self) -> None:
        feat = pharmacophoric_overlap_score(ASPIRIN, "InhA")
        assert feat.shape == (12,)

    def test_dtype(self) -> None:
        feat = pharmacophoric_overlap_score(ASPIRIN, "InhA")
        assert feat.dtype == np.float32

    def test_range(self) -> None:
        feat = pharmacophoric_overlap_score(ASPIRIN, "InhA")
        assert (feat >= 0).all() and (feat <= 2).all()

    def test_unknown_target_zeros(self) -> None:
        feat = pharmacophoric_overlap_score(ASPIRIN, "UNKNOWN_TARGET")
        assert (feat == 0).all()

    def test_different_targets_differ(self) -> None:
        f1 = pharmacophoric_overlap_score(ASPIRIN, "InhA")
        f2 = pharmacophoric_overlap_score(ASPIRIN, "rpoB")
        assert not np.allclose(f1, f2)


class TestProteinLigandFeaturizer:
    @pytest.fixture()
    def featurizer(self) -> ProteinLigandFeaturizer:
        cfg = PLIFConfig(targets=["InhA", "KatG"], include_vina=False)
        return ProteinLigandFeaturizer(config=cfg)

    def test_n_features(self, featurizer: ProteinLigandFeaturizer) -> None:
        assert featurizer.n_features == 12 * 2

    def test_feature_names_len(self, featurizer: ProteinLigandFeaturizer) -> None:
        assert len(featurizer.feature_names) == featurizer.n_features

    def test_compute_single_shape(self, featurizer: ProteinLigandFeaturizer) -> None:
        feat = featurizer.compute_single(ASPIRIN)
        assert feat.shape == (featurizer.n_features,)

    def test_compute_batch_shape(self, featurizer: ProteinLigandFeaturizer) -> None:
        X = featurizer.compute_batch([ASPIRIN, CAFFEINE, INVALID])
        assert X.shape == (3, featurizer.n_features)

    def test_augment_descriptors(self, featurizer: ProteinLigandFeaturizer) -> None:
        X_2d = np.random.rand(2, 10).astype(np.float32)
        X_aug = featurizer.augment_descriptors(X_2d, [ASPIRIN, CAFFEINE])
        assert X_aug.shape == (2, 10 + featurizer.n_features)
        np.testing.assert_allclose(X_aug[:, :10], X_2d)
