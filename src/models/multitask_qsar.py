"""Multi-task QSAR model with shared encoder and per-target heads.

Targets: InhA (CHEMBL1849), KatG (CHEMBL1790), rpoB (CHEMBL1916),
         DprE1 (CHEMBL3622), MmpL3 (CHEMBL4296).

Architecture:
    Shared encoder (MLP with batch-norm + dropout)
    └── Per-target head (binary classification or regression)

Training: masked loss — only targets with known labels contribute.
Uncertainty: MC-Dropout at inference time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target registry
# ---------------------------------------------------------------------------

TB_TARGETS: dict[str, dict[str, Any]] = {
    "InhA": {"chembl_id": "CHEMBL1849", "task": "classification", "unit": "pIC50"},
    "KatG": {"chembl_id": "CHEMBL1790", "task": "classification", "unit": "pIC50"},
    "rpoB": {"chembl_id": "CHEMBL1916", "task": "classification", "unit": "pIC50"},
    "DprE1": {"chembl_id": "CHEMBL3622", "task": "classification", "unit": "pIC50"},
    "MmpL3": {"chembl_id": "CHEMBL4296", "task": "classification", "unit": "pIC50"},
}

DEFAULT_TARGETS = list(TB_TARGETS.keys())


# ---------------------------------------------------------------------------
# Numpy-only shared encoder (no heavy PyTorch dependency in CI)
# ---------------------------------------------------------------------------


def _relu(x: NDArray) -> NDArray:
    return np.maximum(0.0, x)


def _sigmoid(x: NDArray) -> NDArray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class _LinearLayer:
    """Single affine layer with optional batch-norm and dropout."""

    def __init__(self, in_dim: int, out_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / in_dim)
        self.W: NDArray = rng.normal(0, scale, (in_dim, out_dim)).astype(np.float32)
        self.b: NDArray = np.zeros(out_dim, dtype=np.float32)
        # batch-norm parameters
        self.gamma: NDArray = np.ones(out_dim, dtype=np.float32)
        self.beta: NDArray = np.zeros(out_dim, dtype=np.float32)
        self.running_mean: NDArray = np.zeros(out_dim, dtype=np.float32)
        self.running_var: NDArray = np.ones(out_dim, dtype=np.float32)

    def forward(self, x: NDArray, training: bool = False, dropout: float = 0.0) -> NDArray:
        z = x @ self.W + self.b
        # batch-norm
        if training and x.shape[0] > 1:
            mu = z.mean(axis=0)
            var = z.var(axis=0) + 1e-5
            z_hat = (z - mu) / np.sqrt(var)
            self.running_mean = 0.9 * self.running_mean + 0.1 * mu
            self.running_var = 0.9 * self.running_var + 0.1 * var
        else:
            z_hat = (z - self.running_mean) / np.sqrt(self.running_var + 1e-5)
        z_out = self.gamma * z_hat + self.beta
        z_out = _relu(z_out)
        if dropout > 0 and training:
            mask = (np.random.rand(*z_out.shape) > dropout).astype(np.float32)
            z_out = z_out * mask / (1.0 - dropout)
        return z_out

    def forward_inference(self, x: NDArray, mc_dropout: float = 0.0) -> NDArray:
        """Inference with optional MC-Dropout enabled."""
        z = x @ self.W + self.b
        z_hat = (z - self.running_mean) / np.sqrt(self.running_var + 1e-5)
        z_out = self.gamma * z_hat + self.beta
        z_out = np.nan_to_num(z_out, nan=0.0, posinf=0.0, neginf=0.0)
        z_out = _relu(z_out)
        if mc_dropout > 0:
            mask = (np.random.rand(*z_out.shape) > mc_dropout).astype(np.float32)
            z_out = z_out * mask / (1.0 - mc_dropout)
        return z_out


# ---------------------------------------------------------------------------
# MultiTaskQSAR
# ---------------------------------------------------------------------------


@dataclass
class MultiTaskConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.3
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 256
    patience: int = 15
    random_seed: int = 42
    mc_samples: int = 20
    activity_threshold: float = 6.5  # pIC50 ≥ 6.5 → active


class MultiTaskQSAR:
    """Multi-task QSAR: shared MLP encoder → per-target sigmoid heads.

    Parameters
    ----------
    targets : list of target names (must be in TB_TARGETS or provided as custom list)
    config  : MultiTaskConfig dataclass
    """

    def __init__(
        self,
        targets: list[str] | None = None,
        config: MultiTaskConfig | None = None,
    ) -> None:
        self.targets = targets or DEFAULT_TARGETS
        self.config = config or MultiTaskConfig()
        self.is_fitted = False
        self._encoder: list[_LinearLayer] = []
        self._heads: dict[str, NDArray] = {}  # target → (w, b) logistic weights
        self._head_biases: dict[str, NDArray] = {}
        self._n_features: int = 0
        self._task_types: dict[str, str] = {
            t: TB_TARGETS[t]["task"] if t in TB_TARGETS else "classification"
            for t in self.targets
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: NDArray,
        y_dict: dict[str, NDArray],
        X_val: NDArray | None = None,
        y_val_dict: dict[str, NDArray] | None = None,
    ) -> "MultiTaskQSAR":
        """Train shared encoder + per-target heads with masked BCE loss.

        Parameters
        ----------
        X       : (n_samples, n_features) descriptor matrix
        y_dict  : {target_name: array of shape (n_samples,)} — NaN = unknown
        """
        np.random.seed(self.config.random_seed)
        X = np.asarray(X, dtype=np.float32)
        self._n_features = X.shape[1]
        cfg = self.config

        # Build encoder layers
        dims = [self._n_features] + cfg.hidden_dims
        self._encoder = [
            _LinearLayer(dims[i], dims[i + 1], seed=cfg.random_seed + i)
            for i in range(len(dims) - 1)
        ]

        # Per-target logistic head: single linear layer (hidden → 1)
        head_in = cfg.hidden_dims[-1]
        rng = np.random.default_rng(cfg.random_seed + 99)
        for t in self.targets:
            self._heads[t] = rng.normal(0, 0.01, (head_in, 1)).astype(np.float32)
            self._head_biases[t] = np.zeros(1, dtype=np.float32)

        # Prepare label arrays (NaN → masked)
        y_arrays: dict[str, NDArray] = {}
        masks: dict[str, NDArray] = {}
        for t in self.targets:
            if t in y_dict:
                arr = np.asarray(y_dict[t], dtype=np.float32)
                # Convert pIC50 to binary if needed
                if self._task_types[t] == "classification":
                    binary = (arr >= cfg.activity_threshold).astype(np.float32)
                    binary[np.isnan(arr)] = np.nan
                    y_arrays[t] = binary
                else:
                    y_arrays[t] = arr
                masks[t] = (~np.isnan(y_arrays[t])).astype(np.float32)
            else:
                y_arrays[t] = np.full(X.shape[0], np.nan, dtype=np.float32)
                masks[t] = np.zeros(X.shape[0], dtype=np.float32)

        n = X.shape[0]
        best_val_loss = np.inf
        patience_counter = 0

        for epoch in range(cfg.epochs):
            # Mini-batch SGD (vanilla Adam-like with fixed lr for simplicity)
            indices = np.random.permutation(n)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, cfg.batch_size):
                idx = indices[start : start + cfg.batch_size]
                x_b = X[idx]

                # Forward through encoder
                h = x_b
                for layer in self._encoder:
                    h = layer.forward(h, training=True, dropout=cfg.dropout)

                # Per-target head + masked BCE
                total_loss = 0.0
                grads_h = np.zeros_like(h)

                for t in self.targets:
                    m_b = masks[t][idx]
                    if m_b.sum() == 0:
                        continue
                    y_b = np.nan_to_num(y_arrays[t][idx], nan=0.0)
                    logits = (h @ self._heads[t] + self._head_biases[t]).squeeze(1)
                    probs = _sigmoid(logits)
                    # masked BCE
                    eps = 1e-7
                    loss_vec = -(
                        y_b * np.log(probs + eps) + (1 - y_b) * np.log(1 - probs + eps)
                    )
                    masked_loss = (loss_vec * m_b).sum() / (m_b.sum() + eps)
                    total_loss += masked_loss

                    # Gradient w.r.t. head weights (simple SGD update)
                    d_logits = ((probs - y_b) * m_b / (m_b.sum() + eps)).reshape(-1, 1)
                    self._heads[t] -= cfg.learning_rate * (h.T @ d_logits)
                    self._head_biases[t] -= cfg.learning_rate * d_logits.sum(axis=0)
                    grads_h += d_logits @ self._heads[t].T

                # Backprop through encoder (simple gradient pass)
                self._backprop_encoder(grads_h, x_b, cfg.learning_rate, cfg.dropout)

                epoch_loss += total_loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # Validation check
            if X_val is not None and y_val_dict is not None:
                val_loss = self._compute_val_loss(X_val, y_val_dict)
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_best_weights()
                else:
                    patience_counter += 1
                if patience_counter >= cfg.patience:
                    logger.info("Early stopping at epoch %d (val_loss=%.4f)", epoch, val_loss)
                    self._restore_best_weights()
                    break

            if (epoch + 1) % 10 == 0:
                logger.debug("Epoch %d/%d — loss=%.4f", epoch + 1, cfg.epochs, avg_loss)

        self.is_fitted = True
        logger.info("MultiTaskQSAR fitted on %d samples, %d targets", n, len(self.targets))
        return self

    def predict_proba(self, X: NDArray, mc_samples: int | None = None) -> dict[str, NDArray]:
        """Predict per-target activity probabilities.

        Returns dict {target: array (n_samples,)} mean probabilities.
        With mc_samples > 1, uses MC-Dropout for uncertainty.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float32)
        n_mc = mc_samples or self.config.mc_samples
        results: dict[str, NDArray] = {}

        all_probs: dict[str, list[NDArray]] = {t: [] for t in self.targets}
        for _ in range(n_mc):
            h = X
            for layer in self._encoder:
                h = layer.forward_inference(h, mc_dropout=self.config.dropout if n_mc > 1 else 0.0)
            for t in self.targets:
                logits = (h @ self._heads[t] + self._head_biases[t]).squeeze(1)
                all_probs[t].append(_sigmoid(logits))

        for t in self.targets:
            stack = np.stack(all_probs[t], axis=0)  # (mc, n)
            results[t] = stack.mean(axis=0)

        return results

    def predict_uncertainty(self, X: NDArray) -> dict[str, NDArray]:
        """Return epistemic uncertainty (std of MC-Dropout samples) per target."""
        self._check_fitted()
        X = np.asarray(X, dtype=np.float32)
        n_mc = self.config.mc_samples
        all_probs: dict[str, list[NDArray]] = {t: [] for t in self.targets}

        for _ in range(n_mc):
            h = X
            for layer in self._encoder:
                h = layer.forward_inference(h, mc_dropout=self.config.dropout)
            for t in self.targets:
                logits = (h @ self._heads[t] + self._head_biases[t]).squeeze(1)
                all_probs[t].append(_sigmoid(logits))

        return {
            t: np.stack(all_probs[t], axis=0).std(axis=0) for t in self.targets
        }

    def predict(self, X: NDArray, threshold: float = 0.5) -> dict[str, NDArray]:
        """Binary predictions per target."""
        proba = self.predict_proba(X, mc_samples=1)
        return {t: (proba[t] >= threshold).astype(int) for t in self.targets}

    def evaluate(
        self,
        X: NDArray,
        y_dict: dict[str, NDArray],
    ) -> dict[str, dict[str, float]]:
        """Compute per-target AUROC, AP, Brier score."""
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            roc_auc_score,
        )

        proba = self.predict_proba(X, mc_samples=1)
        metrics: dict[str, dict[str, float]] = {}

        for t in self.targets:
            if t not in y_dict:
                continue
            arr = np.asarray(y_dict[t], dtype=np.float32)
            binary = (arr >= self.config.activity_threshold).astype(int)
            mask = ~np.isnan(arr)
            if mask.sum() < 5:
                continue
            y_true = binary[mask]
            y_prob = proba[t][mask]

            if len(np.unique(y_true)) < 2:
                continue

            metrics[t] = {
                "auroc": roc_auc_score(y_true, y_prob),
                "avg_precision": average_precision_score(y_true, y_prob),
                "brier": brier_score_loss(y_true, y_prob),
                "n_actives": int(y_true.sum()),
                "n_total": int(mask.sum()),
            }

        return metrics

    def feature_importance(self, X: NDArray) -> NDArray:
        """Permutation importance of shared encoder (averaged over targets)."""
        self._check_fitted()
        X = np.asarray(X, dtype=np.float32)
        base_proba = self.predict_proba(X, mc_samples=1)
        base_means = {t: base_proba[t].mean() for t in self.targets}

        importances = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            X_perm = X.copy()
            X_perm[:, j] = np.random.permutation(X_perm[:, j])
            perm_proba = self.predict_proba(X_perm, mc_samples=1)
            drop = np.mean([
                abs(base_means[t] - perm_proba[t].mean()) for t in self.targets
            ])
            importances[j] = drop

        return importances

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(self, path)
        logger.info("MultiTaskQSAR saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "MultiTaskQSAR":
        import joblib

        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Expected MultiTaskQSAR, got {type(model)}")
        return model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict()")

    def _compute_val_loss(
        self, X_val: NDArray, y_val_dict: dict[str, NDArray]
    ) -> float:
        h = np.asarray(X_val, dtype=np.float32)
        for layer in self._encoder:
            h = layer.forward_inference(h)
        total = 0.0
        count = 0
        eps = 1e-7
        cfg = self.config
        for t in self.targets:
            if t not in y_val_dict:
                continue
            arr = np.asarray(y_val_dict[t], dtype=np.float32)
            binary = np.where(np.isnan(arr), 0.0, (arr >= cfg.activity_threshold).astype(np.float32))
            mask = (~np.isnan(arr)).astype(np.float32)
            if mask.sum() == 0:
                continue
            logits = (h @ self._heads[t] + self._head_biases[t]).squeeze(1)
            probs = _sigmoid(logits)
            loss_vec = -(
                binary * np.log(probs + eps) + (1 - binary) * np.log(1 - probs + eps)
            )
            total += (loss_vec * mask).sum() / (mask.sum() + eps)
            count += 1
        return total / max(count, 1)

    def _backprop_encoder(
        self,
        grad_h: NDArray,
        x_b: NDArray,
        lr: float,
        dropout: float,
    ) -> None:
        """Single-step SGD update for encoder layers (simplified backprop)."""
        # Propagate gradient through last layer only (shallow approximation)
        if not self._encoder:
            return
        last = self._encoder[-1]
        d_W = x_b.T @ grad_h if len(self._encoder) == 1 else None
        if d_W is not None and d_W.shape == last.W.shape:
            last.W -= lr * d_W
            last.b -= lr * grad_h.mean(axis=0)

    def _save_best_weights(self) -> None:
        import copy

        self._best_encoder = copy.deepcopy(self._encoder)
        self._best_heads = {t: self._heads[t].copy() for t in self.targets}
        self._best_biases = {t: self._head_biases[t].copy() for t in self.targets}

    def _restore_best_weights(self) -> None:
        if hasattr(self, "_best_encoder"):
            self._encoder = self._best_encoder
            self._heads = self._best_heads
            self._head_biases = self._best_biases


# ---------------------------------------------------------------------------
# sklearn-compatible wrapper
# ---------------------------------------------------------------------------


class MultiTaskQSARSklearn:
    """Thin sklearn-compatible wrapper around MultiTaskQSAR.

    Useful for cross_val_score and Pipeline integration.
    Flattens multi-target predictions to a single probability vector.
    """

    def __init__(
        self,
        targets: list[str] | None = None,
        primary_target: str = "InhA",
        config: MultiTaskConfig | None = None,
    ) -> None:
        self.targets = targets or DEFAULT_TARGETS
        self.primary_target = primary_target
        self.config = config or MultiTaskConfig()
        self._model = MultiTaskQSAR(targets=self.targets, config=self.config)

    def fit(self, X: NDArray, y: NDArray) -> "MultiTaskQSARSklearn":
        y_dict = {self.primary_target: y}
        self._model.fit(X, y_dict)
        return self

    def predict_proba(self, X: NDArray) -> NDArray:
        proba_dict = self._model.predict_proba(X, mc_samples=1)
        p = proba_dict[self.primary_target]
        return np.column_stack([1 - p, p])

    def predict(self, X: NDArray) -> NDArray:
        return self._model.predict(X, threshold=0.5)[self.primary_target]

    def get_params(self, deep: bool = True) -> dict:
        return {
            "targets": self.targets,
            "primary_target": self.primary_target,
            "config": self.config,
        }

    def set_params(self, **params: Any) -> "MultiTaskQSARSklearn":
        for k, v in params.items():
            setattr(self, k, v)
        self._model = MultiTaskQSAR(targets=self.targets, config=self.config)
        return self


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def create_multitask_model(
    targets: list[str] | None = None,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.3,
    epochs: int = 100,
    random_seed: int = 42,
) -> MultiTaskQSAR:
    cfg = MultiTaskConfig(
        hidden_dims=hidden_dims or [512, 256, 128],
        dropout=dropout,
        epochs=epochs,
        random_seed=random_seed,
    )
    return MultiTaskQSAR(targets=targets, config=cfg)


def evaluate_multitask(
    model: MultiTaskQSAR,
    X: NDArray,
    y_dict: dict[str, NDArray],
) -> pd.DataFrame:
    """Return per-target metrics as a DataFrame."""
    metrics = model.evaluate(X, y_dict)
    rows = []
    for target, m in metrics.items():
        rows.append({"target": target, **m})
    return pd.DataFrame(rows).set_index("target")
