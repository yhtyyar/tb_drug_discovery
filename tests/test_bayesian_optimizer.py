"""Tests for Bayesian Optimization over Latent Space (BOLS)."""

from __future__ import annotations

import numpy as np
import pytest

from src.generation.bayesian_optimizer import (
    BOLSConfig,
    BayesianLatentOptimizer,
    GaussianProcess,
    ei_acquisition,
    pi_acquisition,
    ucb_acquisition,
)
from src.generation.conditional_vae import CVAEConfig, ConditionalVAE


# ---------------------------------------------------------------------------
# GaussianProcess
# ---------------------------------------------------------------------------


class TestGaussianProcess:
    @pytest.fixture()
    def gp(self) -> GaussianProcess:
        return GaussianProcess(length_scale=1.0, signal_var=1.0, noise_var=1e-3, optimize_hyperparams=False)

    def test_fit_predict_shape(self, gp: GaussianProcess) -> None:
        X_train = np.random.rand(10, 4).astype(np.float32)
        y_train = np.random.rand(10).astype(np.float32)
        gp.fit(X_train, y_train)
        X_test = np.random.rand(5, 4).astype(np.float32)
        mu, var = gp.predict(X_test)
        assert mu.shape == (5,)
        assert var.shape == (5,)

    def test_variance_non_negative(self, gp: GaussianProcess) -> None:
        X_train = np.random.rand(8, 3).astype(np.float32)
        y_train = np.random.rand(8).astype(np.float32)
        gp.fit(X_train, y_train)
        X_test = np.random.rand(5, 3).astype(np.float32)
        _, var = gp.predict(X_test)
        assert (var >= 0).all()

    def test_predict_without_fit(self, gp: GaussianProcess) -> None:
        X = np.random.rand(3, 4).astype(np.float32)
        mu, var = gp.predict(X)
        assert mu.shape == (3,)

    def test_hyperparams_optimize(self) -> None:
        gp = GaussianProcess(optimize_hyperparams=True)
        X = np.random.rand(15, 4).astype(np.float32)
        y = np.random.rand(15).astype(np.float32)
        gp.fit(X, y)
        assert gp.length_scale > 0
        assert gp.signal_var > 0


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------


class TestAcquisitionFunctions:
    def test_ucb_shape(self) -> None:
        mu = np.array([0.5, 0.7, 0.3])
        var = np.array([0.1, 0.2, 0.05])
        acq = ucb_acquisition(mu, var, beta=2.0)
        assert acq.shape == (3,)

    def test_ucb_higher_var_higher_acq(self) -> None:
        mu = np.array([0.5, 0.5])
        var = np.array([0.01, 1.0])
        acq = ucb_acquisition(mu, var, beta=2.0)
        assert acq[1] > acq[0]

    def test_ei_shape(self) -> None:
        mu = np.array([0.5, 0.7])
        var = np.array([0.1, 0.2])
        acq = ei_acquisition(mu, var, y_best=0.4)
        assert acq.shape == (2,)

    def test_ei_non_negative(self) -> None:
        mu = np.array([0.5, 0.3, 0.8])
        var = np.array([0.1, 0.05, 0.2])
        acq = ei_acquisition(mu, var, y_best=0.4)
        assert (acq >= 0).all()

    def test_pi_range(self) -> None:
        mu = np.random.rand(10).astype(np.float32)
        var = np.abs(np.random.rand(10)).astype(np.float32) + 0.01
        acq = pi_acquisition(mu, var, y_best=0.5)
        assert (acq >= 0).all() and (acq <= 1).all()


# ---------------------------------------------------------------------------
# BayesianLatentOptimizer
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_vae() -> ConditionalVAE:
    cfg = CVAEConfig(latent_dim=8, hidden_dim=16, embed_dim=8, condition_dim=2, max_len=20)
    return ConditionalVAE(config=cfg)


def dummy_objective(smiles_list: list[str]) -> np.ndarray:
    """Toy objective: reward = fraction of 'C' characters / 10."""
    return np.array([s.count("C") / 10.0 for s in smiles_list], dtype=np.float32)


class TestBayesianLatentOptimizer:
    @pytest.fixture()
    def optimizer(self, tiny_vae: ConditionalVAE) -> BayesianLatentOptimizer:
        cfg = BOLSConfig(
            n_initial=5, n_iterations=3, n_candidates=20,
            batch_size=2, latent_dim=8, random_seed=0,
        )
        return BayesianLatentOptimizer(tiny_vae, dummy_objective, config=cfg)

    def test_initialize_populates_history(self, optimizer: BayesianLatentOptimizer) -> None:
        optimizer.initialize()
        assert len(optimizer.z_history) == 5
        assert len(optimizer.smiles_history) == 5
        assert len(optimizer.reward_history) == 5

    def test_step_returns_log(self, optimizer: BayesianLatentOptimizer) -> None:
        optimizer.initialize()
        log = optimizer.step()
        assert "best_reward" in log
        assert "batch_mean_reward" in log
        assert "n_evaluated" in log

    def test_run_correct_iterations(self, optimizer: BayesianLatentOptimizer) -> None:
        logs = optimizer.run(n_iterations=3)
        assert len(logs) == 3

    def test_best_molecules_count(self, optimizer: BayesianLatentOptimizer) -> None:
        optimizer.run(n_iterations=2)
        best = optimizer.best_molecules(top_k=3)
        assert len(best) <= 3
        for smi, r in best:
            assert isinstance(smi, str)
            assert isinstance(r, float)

    def test_best_reward_monotone(self, optimizer: BayesianLatentOptimizer) -> None:
        logs = optimizer.run(n_iterations=3)
        rewards = [log["best_reward"] for log in logs]
        for i in range(1, len(rewards)):
            assert rewards[i] >= rewards[i - 1] - 1e-8

    def test_convergence_data(self, optimizer: BayesianLatentOptimizer) -> None:
        optimizer.run(n_iterations=2)
        data = optimizer.convergence_plot_data()
        assert "cumulative_best_reward" in data
        assert "all_rewards" in data
        assert len(data["cumulative_best_reward"]) == len(data["all_rewards"])

    def test_different_acquisitions(self, tiny_vae: ConditionalVAE) -> None:
        for acq in ["ucb", "ei", "pi"]:
            cfg = BOLSConfig(
                n_initial=4, n_iterations=1, n_candidates=10,
                batch_size=2, latent_dim=8, acquisition=acq,
            )
            opt = BayesianLatentOptimizer(tiny_vae, dummy_objective, config=cfg)
            logs = opt.run(n_iterations=1)
            assert len(logs) == 1

    def test_with_seed_smiles(
        self, optimizer: BayesianLatentOptimizer
    ) -> None:
        seeds = ["CC(=O)O", "c1ccccc1", "CCN"]
        optimizer.initialize(seed_smiles=seeds)
        assert len(optimizer.smiles_history) >= len(seeds)
