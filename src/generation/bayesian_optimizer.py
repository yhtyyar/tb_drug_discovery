"""Bayesian Optimization over VAE Latent Space (BOLS).

Strategy
--------
1. Encode known actives into latent space z ∈ ℝ^d
2. Fit a GP surrogate f(z) ≈ activity
3. Maximise acquisition function (UCB / EI / PI) to select next z
4. Decode z_next → SMILES → evaluate → update GP

This enables sample-efficient exploration of chemical space guided by
the GP uncertainty, combining the generative power of the CVAE with
principled Bayesian decision-making.

All heavy computation uses scipy.stats (available in standard Python)
and numpy — no torch required for the BO loop itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gaussian Process (squared-exponential kernel, numpy implementation)
# ---------------------------------------------------------------------------


class GaussianProcess:
    """Exact GP regression with squared-exponential (RBF) kernel.

    Parameters
    ----------
    length_scale : RBF kernel length scale
    signal_var   : signal variance σ_f²
    noise_var    : observation noise σ_n²
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        signal_var: float = 1.0,
        noise_var: float = 1e-3,
        optimize_hyperparams: bool = True,
    ) -> None:
        self.length_scale = length_scale
        self.signal_var = signal_var
        self.noise_var = noise_var
        self.optimize_hyperparams = optimize_hyperparams
        self._X_train: NDArray | None = None
        self._y_train: NDArray | None = None
        self._K_inv: NDArray | None = None
        self._alpha: NDArray | None = None

    def _rbf(self, X1: NDArray, X2: NDArray) -> NDArray:
        """Squared-exponential kernel K(X1, X2)."""
        diff = X1[:, None, :] - X2[None, :, :]
        sq_dist = (diff ** 2).sum(axis=2)
        return self.signal_var * np.exp(-0.5 * sq_dist / self.length_scale ** 2)

    def fit(self, X: NDArray, y: NDArray) -> "GaussianProcess":
        """Fit GP to training data (X, y)."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._X_train = X
        self._y_train = y

        if self.optimize_hyperparams and len(X) >= 5:
            self._optimize_hyperparams(X, y)

        K = self._rbf(X, X) + self.noise_var * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K + 1e-8 * np.eye(len(K)))
            self._alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            self._K_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(len(K))))
        except np.linalg.LinAlgError:
            self._K_inv = np.linalg.pinv(K)
            self._alpha = self._K_inv @ y
        return self

    def predict(self, X: NDArray) -> tuple[NDArray, NDArray]:
        """Predict mean and variance at new points.

        Returns (mu, sigma²) arrays of shape (n,).
        """
        X = np.asarray(X, dtype=np.float64)
        if self._X_train is None:
            return np.zeros(len(X)), np.ones(len(X))

        K_s = self._rbf(self._X_train, X)  # (n_train, n_test)
        mu = K_s.T @ self._alpha
        K_ss = self._rbf(X, X)  # (n_test, n_test)
        cov = K_ss - K_s.T @ self._K_inv @ K_s
        var = np.diag(cov).clip(0.0)
        return mu.astype(np.float32), var.astype(np.float32)

    def _optimize_hyperparams(self, X: NDArray, y: NDArray) -> None:
        """Simple grid search over length_scale and signal_var."""
        best_ll = -np.inf
        for ls in [0.1, 0.5, 1.0, 2.0, 5.0]:
            for sv in [0.1, 0.5, 1.0, 2.0]:
                self.length_scale, self.signal_var = ls, sv
                K = self._rbf(X, X) + self.noise_var * np.eye(len(X))
                try:
                    sign, logdet = np.linalg.slogdet(K)
                    if sign <= 0:
                        continue
                    ll = -0.5 * y @ np.linalg.solve(K, y) - 0.5 * logdet
                    if ll > best_ll:
                        best_ll = ll
                        best_ls, best_sv = ls, sv
                except Exception:
                    continue
        self.length_scale = best_ls
        self.signal_var = best_sv


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------


def ucb_acquisition(
    mu: NDArray, sigma: NDArray, beta: float = 2.0
) -> NDArray:
    """Upper Confidence Bound: μ + β·σ."""
    return mu + beta * np.sqrt(sigma)


def ei_acquisition(
    mu: NDArray, sigma: NDArray, y_best: float, xi: float = 0.01
) -> NDArray:
    """Expected Improvement."""
    sigma = np.sqrt(sigma) + 1e-9
    z = (mu - y_best - xi) / sigma
    return (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)


def pi_acquisition(
    mu: NDArray, sigma: NDArray, y_best: float, xi: float = 0.01
) -> NDArray:
    """Probability of Improvement."""
    sigma = np.sqrt(sigma) + 1e-9
    z = (mu - y_best - xi) / sigma
    return norm.cdf(z)


# ---------------------------------------------------------------------------
# BOLS — Bayesian Optimization over Latent Space
# ---------------------------------------------------------------------------


@dataclass
class BOLSConfig:
    n_initial: int = 20
    n_iterations: int = 50
    n_candidates: int = 500
    acquisition: str = "ucb"     # 'ucb' | 'ei' | 'pi'
    beta: float = 2.0            # UCB exploration coefficient
    xi: float = 0.01             # EI/PI exploration parameter
    latent_dim: int = 128
    random_seed: int = 42
    batch_size: int = 5          # molecules to evaluate per iteration


class BayesianLatentOptimizer:
    """Bayesian optimization loop over the VAE latent space.

    Parameters
    ----------
    vae : ConditionalVAE (or any object with encode/decode methods)
    objective : callable (smiles_list) -> NDArray of scalar rewards
    config : BOLSConfig
    """

    def __init__(
        self,
        vae: Any,
        objective: Callable[[list[str]], NDArray],
        config: BOLSConfig | None = None,
        gp: GaussianProcess | None = None,
    ) -> None:
        self.vae = vae
        self.objective = objective
        self.config = config or BOLSConfig()
        self.gp = gp or GaussianProcess(optimize_hyperparams=True)
        self._rng = np.random.default_rng(self.config.random_seed)

        # History
        self.z_history: list[NDArray] = []
        self.smiles_history: list[str] = []
        self.reward_history: list[float] = []
        self.iteration_logs: list[dict] = []

    @property
    def y_best(self) -> float:
        return float(max(self.reward_history)) if self.reward_history else -np.inf

    def _sample_candidates(self, n: int) -> NDArray:
        """Sample candidate latent vectors."""
        cfg = self.config
        # Combine random sampling + perturbation of best known points
        n_random = max(n // 2, 1)
        z_rand = self._rng.normal(0, 1, (n_random, cfg.latent_dim)).astype(np.float32)

        if len(self.z_history) > 0:
            n_perturb = n - n_random
            best_idx = int(np.argmax(self.reward_history))
            z_best = self.z_history[best_idx]
            noise = self._rng.normal(0, 0.5, (n_perturb, cfg.latent_dim)).astype(np.float32)
            z_perturb = z_best + noise
            return np.vstack([z_rand, z_perturb])
        return z_rand

    def _acquisition(self, mu: NDArray, var: NDArray) -> NDArray:
        cfg = self.config
        if cfg.acquisition == "ucb":
            return ucb_acquisition(mu, var, beta=cfg.beta)
        elif cfg.acquisition == "ei":
            return ei_acquisition(mu, var, y_best=self.y_best, xi=cfg.xi)
        elif cfg.acquisition == "pi":
            return pi_acquisition(mu, var, y_best=self.y_best, xi=cfg.xi)
        else:
            raise ValueError(f"Unknown acquisition: {cfg.acquisition}")

    def initialize(self, seed_smiles: list[str] | None = None) -> None:
        """Seed the optimizer with random or provided molecules."""
        cfg = self.config
        n = cfg.n_initial

        if seed_smiles:
            z_init = np.stack([self.vae.encode(s) for s in seed_smiles[:n]])
            smiles_init = seed_smiles[:n]
        else:
            z_init = self._rng.normal(0, 1, (n, cfg.latent_dim)).astype(np.float32)
            smiles_init = [self.vae.decode(z) for z in z_init]

        rewards = self.objective(smiles_init)

        for z, smi, r in zip(z_init, smiles_init, rewards):
            self.z_history.append(z)
            self.smiles_history.append(smi)
            self.reward_history.append(float(r))

        logger.info(
            "BOLS initialized: %d points, best_reward=%.4f", n, self.y_best
        )

    def step(self) -> dict[str, Any]:
        """Run one BO iteration: fit GP → maximize acquisition → evaluate."""
        cfg = self.config

        if not self.z_history:
            self.initialize()

        # Fit GP
        Z = np.stack(self.z_history)
        y = np.array(self.reward_history, dtype=np.float32)
        self.gp.fit(Z, y)

        # Sample candidates and score via acquisition
        Z_cand = self._sample_candidates(cfg.n_candidates)
        mu, var = self.gp.predict(Z_cand)
        acq = self._acquisition(mu, var)

        # Select top-k candidates (batch)
        top_k = min(cfg.batch_size, len(Z_cand))
        top_indices = np.argsort(acq)[::-1][:top_k]
        z_next = Z_cand[top_indices]
        smiles_next = [self.vae.decode(z) for z in z_next]

        # Evaluate objective
        rewards_next = self.objective(smiles_next)

        # Update history
        for z, smi, r in zip(z_next, smiles_next, rewards_next):
            self.z_history.append(z)
            self.smiles_history.append(smi)
            self.reward_history.append(float(r))

        log = {
            "iteration": len(self.iteration_logs),
            "best_reward": self.y_best,
            "batch_mean_reward": float(rewards_next.mean()),
            "batch_max_reward": float(rewards_next.max()),
            "n_evaluated": len(self.reward_history),
            "acquisition": cfg.acquisition,
        }
        self.iteration_logs.append(log)

        return log

    def run(self, n_iterations: int | None = None) -> list[dict]:
        """Run the full BO loop."""
        n = n_iterations or self.config.n_iterations
        if not self.z_history:
            self.initialize()

        for i in range(n):
            log = self.step()
            if (i + 1) % 10 == 0:
                logger.info(
                    "BOLS iteration %d/%d — best_reward=%.4f",
                    i + 1, n, log["best_reward"],
                )

        return self.iteration_logs

    def best_molecules(self, top_k: int = 10) -> list[tuple[str, float]]:
        """Return top-k (smiles, reward) pairs found so far."""
        if not self.reward_history:
            return []
        idx = np.argsort(self.reward_history)[::-1][:top_k]
        return [
            (self.smiles_history[i], self.reward_history[i]) for i in idx
        ]

    def convergence_plot_data(self) -> dict[str, list]:
        """Return data for a convergence plot (cumulative best reward)."""
        cummax = []
        best = -np.inf
        for r in self.reward_history:
            best = max(best, r)
            cummax.append(best)
        return {
            "iteration": list(range(len(cummax))),
            "cumulative_best_reward": cummax,
            "all_rewards": list(self.reward_history),
        }
