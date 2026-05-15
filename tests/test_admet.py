"""Tests for ADMET prediction module."""

from __future__ import annotations

import pytest

from src.admet import (
    ADMETFilter,
    ADMETPredictor,
    ADMETResult,
    ADMETThresholds,
    admet_score,
    batch_admet,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(c1=O)C"
RIFAMPICIN = "CC1C=CC=C(C(=O)NC2=C(C(=O)C3=CC=CC=C3C2=O)OC)C(=O)OC1"
INVALID = "not_a_smiles_$$"

rdkit = pytest.importorskip("rdkit", reason="RDKit not available")


# ---------------------------------------------------------------------------
# ADMETResult
# ---------------------------------------------------------------------------


class TestADMETResult:
    def test_to_dict_has_expected_keys(self) -> None:
        r = ADMETResult(smiles=ASPIRIN)
        d = r.to_dict()
        assert "smiles" in d
        assert "herg_inhibition" in d
        assert "ames_mutagenicity" in d
        assert "admet_score" in d

    def test_flag_concerns_clean_molecule(self) -> None:
        r = ADMETResult(
            smiles=ASPIRIN,
            herg_inhibition=0.1,
            ames_mutagenicity=0.1,
            hepatotoxicity=0.1,
            solubility_log_s=-3.0,
            lipinski_pass=True,
        )
        r.admet_score = admet_score(r)
        assert r.flag_concerns() == []

    def test_flag_concerns_toxic_molecule(self) -> None:
        r = ADMETResult(
            smiles=ASPIRIN,
            herg_inhibition=0.9,
            ames_mutagenicity=0.8,
            hepatotoxicity=0.85,
            solubility_log_s=-8.0,
            lipinski_pass=False,
        )
        concerns = r.flag_concerns()
        assert any("hERG" in c for c in concerns)
        assert any("Ames" in c for c in concerns)
        assert any("Hepatotox" in c for c in concerns)


# ---------------------------------------------------------------------------
# admet_score
# ---------------------------------------------------------------------------


class TestAdmetScore:
    def test_range(self) -> None:
        r = ADMETResult(smiles=ASPIRIN, qed=0.7, bbb_penetration=0.6)
        s = admet_score(r)
        assert 0.0 <= s <= 1.0

    def test_better_compound_higher_score(self) -> None:
        r_good = ADMETResult(
            smiles=ASPIRIN,
            qed=0.9, herg_inhibition=0.1, ames_mutagenicity=0.1,
            hepatotoxicity=0.1, oral_bioavailability=0.9,
            solubility_log_s=-2.0, bbb_penetration=0.8,
        )
        r_bad = ADMETResult(
            smiles=ASPIRIN,
            qed=0.3, herg_inhibition=0.9, ames_mutagenicity=0.9,
            hepatotoxicity=0.9, oral_bioavailability=0.1,
            solubility_log_s=-8.0, bbb_penetration=0.1,
        )
        assert admet_score(r_good) > admet_score(r_bad)


# ---------------------------------------------------------------------------
# ADMETPredictor
# ---------------------------------------------------------------------------


class TestADMETPredictor:
    @pytest.fixture()
    def predictor(self) -> ADMETPredictor:
        return ADMETPredictor()

    def test_predict_returns_result(self, predictor: ADMETPredictor) -> None:
        r = predictor.predict(ASPIRIN)
        assert isinstance(r, ADMETResult)

    def test_predict_smiles_stored(self, predictor: ADMETPredictor) -> None:
        r = predictor.predict(CAFFEINE)
        assert r.smiles == CAFFEINE

    def test_invalid_smiles_returns_default(self, predictor: ADMETPredictor) -> None:
        r = predictor.predict(INVALID)
        assert isinstance(r, ADMETResult)
        assert r.smiles == INVALID

    def test_all_probabilities_in_range(self, predictor: ADMETPredictor) -> None:
        r = predictor.predict(ASPIRIN)
        for field in [
            r.herg_inhibition, r.ames_mutagenicity, r.hepatotoxicity,
            r.caco2_permeability, r.oral_bioavailability,
            r.cyp1a2_inhibition, r.cyp2c9_inhibition,
            r.cyp2c19_inhibition, r.cyp2d6_inhibition, r.cyp3a4_inhibition,
        ]:
            assert 0.0 <= field <= 1.0, f"Out of range: {field}"

    def test_admet_score_in_range(self, predictor: ADMETPredictor) -> None:
        r = predictor.predict(CAFFEINE)
        assert 0.0 <= r.admet_score <= 1.0

    def test_lipinski_flag_aspirin(self, predictor: ADMETPredictor) -> None:
        r = predictor.predict(ASPIRIN)
        # Aspirin (MW=180, logP=1.2) should pass Ro5
        assert r.lipinski_pass is True

    def test_predict_batch(self, predictor: ADMETPredictor) -> None:
        results = predictor.predict_batch([ASPIRIN, CAFFEINE, INVALID])
        assert len(results) == 3
        for r in results:
            assert isinstance(r, ADMETResult)

    def test_different_molecules_differ(self, predictor: ADMETPredictor) -> None:
        r1 = predictor.predict(ASPIRIN)
        r2 = predictor.predict(CAFFEINE)
        # At least one property should differ between molecules
        diff = (
            abs(r1.solubility_log_s - r2.solubility_log_s) > 0.01
            or abs(r1.bbb_penetration - r2.bbb_penetration) > 0.01
            or abs(r1.qed - r2.qed) > 0.01
        )
        assert diff


# ---------------------------------------------------------------------------
# ADMETFilter
# ---------------------------------------------------------------------------


class TestADMETFilter:
    @pytest.fixture()
    def admet_filter(self) -> ADMETFilter:
        t = ADMETThresholds(
            max_herg=0.7, max_ames=0.7, max_hepatotox=0.7,
            min_solubility_log_s=-7.0,
        )
        return ADMETFilter(thresholds=t)

    def test_returns_three_lists(self, admet_filter: ADMETFilter) -> None:
        passed, results, reasons = admet_filter.filter([ASPIRIN, CAFFEINE])
        assert isinstance(passed, list)
        assert len(results) == 2
        assert len(reasons) == 2

    def test_all_smiles_processed(self, admet_filter: ADMETFilter) -> None:
        smiles = [ASPIRIN, CAFFEINE, INVALID]
        passed, results, reasons = admet_filter.filter(smiles)
        assert len(results) == 3
        assert len(reasons) == 3

    def test_passed_is_subset_of_input(self, admet_filter: ADMETFilter) -> None:
        smiles = [ASPIRIN, CAFFEINE]
        passed, _, _ = admet_filter.filter(smiles)
        for s in passed:
            assert s in smiles


# ---------------------------------------------------------------------------
# batch_admet
# ---------------------------------------------------------------------------


class TestBatchAdmet:
    def test_returns_list_of_dicts(self) -> None:
        results = batch_admet([ASPIRIN, CAFFEINE])
        assert len(results) == 2
        for r in results:
            assert isinstance(r, dict)
            assert "admet_score" in r
