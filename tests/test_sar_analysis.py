"""Tests for SAR cliff detection and MMP analysis."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import (
    SARCliffConfig,
    MMPConfig,
    activity_landscape_index,
    cliff_summary,
    cliffs_to_dataframe,
    detect_sar_cliffs,
    find_matched_pairs,
    mmp_to_dataframe,
    tanimoto_matrix,
    transformation_summary,
)

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",   # aspirin
    "CC(=O)Oc1ccccc1C(=O)N",   # aspirin amide analogue (similar)
    "Cn1cnc2N(C)C(=O)N(C)C(=O)c12",  # caffeine (dissimilar)
    "c1ccc(N)cc1",              # aniline
]
ACTIVITIES = np.array([7.0, 5.0, 6.0, 8.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# tanimoto_matrix
# ---------------------------------------------------------------------------


class TestTanimotoMatrix:
    def test_shape(self) -> None:
        fps = np.random.randint(0, 2, (4, 64)).astype(np.float32)
        sim = tanimoto_matrix(fps)
        assert sim.shape == (4, 4)

    def test_diagonal_ones(self) -> None:
        fps = np.random.randint(0, 2, (4, 64)).astype(np.float32)
        sim = tanimoto_matrix(fps)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)

    def test_symmetric(self) -> None:
        fps = np.random.randint(0, 2, (5, 64)).astype(np.float32)
        sim = tanimoto_matrix(fps)
        np.testing.assert_allclose(sim, sim.T, atol=1e-5)

    def test_range(self) -> None:
        fps = np.random.randint(0, 2, (5, 64)).astype(np.float32)
        sim = tanimoto_matrix(fps)
        assert (sim >= 0).all() and (sim <= 1).all()


# ---------------------------------------------------------------------------
# SAR cliff detection
# ---------------------------------------------------------------------------


class TestSARCliffDetection:
    def test_returns_list(self) -> None:
        cliffs = detect_sar_cliffs(SMILES, ACTIVITIES)
        assert isinstance(cliffs, list)

    def test_cliffs_sorted_by_score(self) -> None:
        cliffs = detect_sar_cliffs(SMILES, ACTIVITIES, config=SARCliffConfig(similarity_threshold=0.3))
        if len(cliffs) >= 2:
            assert cliffs[0].cliff_score >= cliffs[1].cliff_score

    def test_cliff_attributes(self) -> None:
        cliffs = detect_sar_cliffs(SMILES, ACTIVITIES, config=SARCliffConfig(similarity_threshold=0.3))
        for c in cliffs:
            assert hasattr(c, "smiles_a")
            assert hasattr(c, "smiles_b")
            assert c.similarity >= 0
            assert c.delta_activity >= 0

    def test_nan_activities_ignored(self) -> None:
        acts = np.array([7.0, np.nan, 5.0, np.nan], dtype=np.float32)
        cliffs = detect_sar_cliffs(SMILES, acts)
        for c in cliffs:
            assert c.smiles_a not in [SMILES[1], SMILES[3]]
            assert c.smiles_b not in [SMILES[1], SMILES[3]]

    def test_high_threshold_fewer_cliffs(self) -> None:
        low = detect_sar_cliffs(SMILES, ACTIVITIES, config=SARCliffConfig(similarity_threshold=0.1))
        high = detect_sar_cliffs(SMILES, ACTIVITIES, config=SARCliffConfig(similarity_threshold=0.9))
        assert len(low) >= len(high)

    def test_cliffs_to_dataframe(self) -> None:
        cliffs = detect_sar_cliffs(SMILES, ACTIVITIES, config=SARCliffConfig(similarity_threshold=0.3))
        df = cliffs_to_dataframe(cliffs)
        if len(cliffs) > 0:
            assert "smiles_a" in df.columns
            assert "cliff_score" in df.columns

    def test_cliff_summary_empty(self) -> None:
        summary = cliff_summary([])
        assert summary["n_cliffs"] == 0

    def test_cliff_summary_populated(self) -> None:
        cliffs = detect_sar_cliffs(SMILES, ACTIVITIES, config=SARCliffConfig(similarity_threshold=0.3))
        summary = cliff_summary(cliffs)
        assert "n_cliffs" in summary
        assert summary["n_cliffs"] == len(cliffs)


# ---------------------------------------------------------------------------
# MMP analysis
# ---------------------------------------------------------------------------


rdkit = pytest.importorskip("rdkit", reason="RDKit required for MMP")


class TestMMP:
    def test_returns_list(self) -> None:
        pairs = find_matched_pairs(SMILES, ACTIVITIES)
        assert isinstance(pairs, list)

    def test_pair_attributes(self) -> None:
        pairs = find_matched_pairs(SMILES, ACTIVITIES)
        for p in pairs:
            assert hasattr(p, "smiles_a")
            assert hasattr(p, "smiles_b")
            assert hasattr(p, "core")
            assert hasattr(p, "transformation")

    def test_sorted_by_delta_activity(self) -> None:
        pairs = find_matched_pairs(SMILES, ACTIVITIES)
        if len(pairs) >= 2:
            assert abs(pairs[0].delta_activity) >= abs(pairs[1].delta_activity)

    def test_mmp_to_dataframe(self) -> None:
        pairs = find_matched_pairs(SMILES, ACTIVITIES)
        df = mmp_to_dataframe(pairs)
        if len(pairs) > 0:
            assert "transformation" in df.columns
            assert "delta_activity" in df.columns

    def test_transformation_summary(self) -> None:
        pairs = find_matched_pairs(SMILES, ACTIVITIES)
        df = transformation_summary(pairs)
        if len(pairs) > 0:
            assert "mean_delta" in df.columns


# ---------------------------------------------------------------------------
# Activity Landscape Index
# ---------------------------------------------------------------------------


class TestActivityLandscapeIndex:
    def test_returns_float(self) -> None:
        sali = activity_landscape_index(SMILES, ACTIVITIES)
        assert isinstance(sali, float)
        assert sali >= 0.0

    def test_all_nan_returns_zero(self) -> None:
        acts = np.full(4, np.nan, dtype=np.float32)
        sali = activity_landscape_index(SMILES, acts)
        assert sali == 0.0
