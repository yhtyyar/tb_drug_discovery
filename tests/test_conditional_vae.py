"""Tests for Conditional VAE and REINVENT fine-tuner."""

from __future__ import annotations

import numpy as np
import pytest

from src.generation.conditional_vae import (
    CVAEConfig,
    ConditionalVAE,
    REINVENTFinetuner,
    RewardConfig,
    RewardFunction,
    smiles_to_tokens,
    tokens_to_smiles,
    VOCAB_SIZE,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2N(C)C(=O)N(C)C(=O)c12"


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


class TestTokenisation:
    def test_round_trip(self) -> None:
        tokens = smiles_to_tokens(ASPIRIN, max_len=80)
        smi = tokens_to_smiles(tokens)
        assert isinstance(smi, str)
        assert len(smi) > 0

    def test_fixed_length(self) -> None:
        tokens = smiles_to_tokens(ASPIRIN, max_len=50)
        assert len(tokens) == 50

    def test_starts_with_sos(self) -> None:
        tokens = smiles_to_tokens(ASPIRIN, max_len=50)
        assert tokens[0] == SOS_IDX

    def test_all_indices_in_vocab(self) -> None:
        tokens = smiles_to_tokens(ASPIRIN, max_len=50)
        assert all(0 <= t < VOCAB_SIZE for t in tokens)


# ---------------------------------------------------------------------------
# ConditionalVAE
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_cvae() -> ConditionalVAE:
    cfg = CVAEConfig(latent_dim=16, hidden_dim=32, embed_dim=16, condition_dim=4, max_len=30)
    return ConditionalVAE(config=cfg)


class TestConditionalVAE:
    def test_encode_shape(self, tiny_cvae: ConditionalVAE) -> None:
        z = tiny_cvae.encode(ASPIRIN)
        assert z.shape == (16,)

    def test_encode_dtype(self, tiny_cvae: ConditionalVAE) -> None:
        z = tiny_cvae.encode(ASPIRIN)
        assert z.dtype == np.float32

    def test_encode_no_nan(self, tiny_cvae: ConditionalVAE) -> None:
        z = tiny_cvae.encode(ASPIRIN)
        assert not np.isnan(z).any()

    def test_decode_returns_string(self, tiny_cvae: ConditionalVAE) -> None:
        z = np.random.rand(16).astype(np.float32)
        smi = tiny_cvae.decode(z)
        assert isinstance(smi, str)

    def test_generate_count(self, tiny_cvae: ConditionalVAE) -> None:
        smiles = tiny_cvae.generate(5)
        assert len(smiles) == 5

    def test_generate_all_strings(self, tiny_cvae: ConditionalVAE) -> None:
        smiles = tiny_cvae.generate(3)
        for s in smiles:
            assert isinstance(s, str)

    def test_with_condition(self, tiny_cvae: ConditionalVAE) -> None:
        cond = np.array([0.5, 0.3, 0.8, 0.1], dtype=np.float32)
        z = tiny_cvae.encode(ASPIRIN, condition=cond)
        assert z.shape == (16,)

    def test_interpolate_length(self, tiny_cvae: ConditionalVAE) -> None:
        interp = tiny_cvae.interpolate(ASPIRIN, CAFFEINE, n_steps=4)
        assert len(interp) == 4
        for s in interp:
            assert isinstance(s, str)

    def test_save_load(self, tiny_cvae: ConditionalVAE, tmp_path) -> None:
        path = str(tmp_path / "cvae.joblib")
        tiny_cvae.save(path)
        loaded = ConditionalVAE.load(path)
        z = loaded.encode(ASPIRIN)
        assert z.shape == (16,)


# ---------------------------------------------------------------------------
# RewardFunction
# ---------------------------------------------------------------------------


class TestRewardFunction:
    def test_returns_array(self) -> None:
        fn = RewardFunction()
        rewards = fn([ASPIRIN, CAFFEINE])
        assert rewards.shape == (2,)

    def test_range(self) -> None:
        fn = RewardFunction()
        rewards = fn([ASPIRIN, CAFFEINE, "not_smiles"])
        assert (rewards >= 0).all() and (rewards <= 1).all()

    def test_invalid_smiles_zero_reward(self) -> None:
        fn = RewardFunction()
        rewards = fn(["not_a_smiles_$$$$$$$$$$"])
        assert rewards[0] == 0.0

    def test_with_reference(self) -> None:
        fn = RewardFunction(reference_smiles=[ASPIRIN])
        rewards = fn([ASPIRIN, CAFFEINE])
        assert rewards.shape == (2,)

    def test_weights_sum_to_one(self) -> None:
        cfg = RewardConfig()
        total = (
            cfg.activity_weight + cfg.admet_weight
            + cfg.diversity_weight + cfg.novelty_weight
        )
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# REINVENTFinetuner
# ---------------------------------------------------------------------------


class TestREINVENTFinetuner:
    @pytest.fixture()
    def finetuner(self, tiny_cvae: ConditionalVAE) -> REINVENTFinetuner:
        reward_fn = RewardFunction()
        return REINVENTFinetuner(
            model=tiny_cvae,
            reward_fn=reward_fn,
            n_steps=3,
            batch_size=8,
        )

    def test_step_returns_metrics(self, finetuner: REINVENTFinetuner) -> None:
        metrics = finetuner.step()
        assert "mean_reward" in metrics
        assert "max_reward" in metrics
        assert "n_valid" in metrics

    def test_run_returns_history(self, finetuner: REINVENTFinetuner) -> None:
        history = finetuner.run(n_steps=2)
        assert len(history) == 2
        assert all("step" in m for m in history)

    def test_generate_optimised(self, finetuner: REINVENTFinetuner) -> None:
        smiles = finetuner.generate_optimised(n=5)
        assert len(smiles) == 5
