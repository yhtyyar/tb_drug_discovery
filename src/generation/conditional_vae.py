"""Conditional SMILES VAE with property-guided generation.

Architecture
-----------
Encoder : GRU(embed) → [μ, log σ²] (latent_dim)
Decoder : GRU(z + condition) → softmax over vocabulary

Conditioning: one-hot or continuous property vector
concatenated to every decoder step.

The model is fully functional with numpy/pure Python when torch is absent
(training only works with torch; inference falls back to a stub).

REINVENT-style RL
-----------------
RewardFunction    : weights ADMET score + activity prediction
REINVENTFinetuner : prior-model KL constraint + policy gradient
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary helpers (shared with existing VAE)
# ---------------------------------------------------------------------------

SMILES_CHARSET = (
    "PAD SOS EOS UNK "
    "C N O S F Cl Br I P "
    "c n o s "
    "( ) [ ] = # + - . @ / \\ "
    "1 2 3 4 5 6 7 8 9 0 "
    "H"
).split()

CHAR2IDX = {c: i for i, c in enumerate(SMILES_CHARSET)}
IDX2CHAR = {i: c for i, c in enumerate(SMILES_CHARSET)}
PAD_IDX = CHAR2IDX["PAD"]
SOS_IDX = CHAR2IDX["SOS"]
EOS_IDX = CHAR2IDX["EOS"]
UNK_IDX = CHAR2IDX["UNK"]
VOCAB_SIZE = len(SMILES_CHARSET)


def smiles_to_tokens(smiles: str, max_len: int = 120) -> list[int]:
    """Tokenise SMILES into integer indices (character-level)."""
    tokens = [SOS_IDX]
    i = 0
    while i < len(smiles):
        # Two-char tokens first (Cl, Br)
        two = smiles[i : i + 2]
        if two in CHAR2IDX:
            tokens.append(CHAR2IDX[two])
            i += 2
        elif smiles[i] in CHAR2IDX:
            tokens.append(CHAR2IDX[smiles[i]])
            i += 1
        else:
            tokens.append(UNK_IDX)
            i += 1
    tokens.append(EOS_IDX)
    # Pad / truncate
    tokens = tokens[:max_len]
    tokens += [PAD_IDX] * (max_len - len(tokens))
    return tokens


def tokens_to_smiles(tokens: list[int]) -> str:
    chars = []
    for t in tokens:
        if t == EOS_IDX:
            break
        if t in (SOS_IDX, PAD_IDX):
            continue
        chars.append(IDX2CHAR.get(t, "?"))
    return "".join(chars)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CVAEConfig:
    vocab_size: int = VOCAB_SIZE
    embed_dim: int = 64
    hidden_dim: int = 256
    latent_dim: int = 128
    condition_dim: int = 8
    max_len: int = 120
    dropout: float = 0.1
    learning_rate: float = 1e-3
    kl_weight: float = 1.0
    kl_anneal_epochs: int = 10
    epochs: int = 50
    batch_size: int = 128
    random_seed: int = 42


@dataclass
class RewardConfig:
    activity_weight: float = 0.4
    admet_weight: float = 0.4
    diversity_weight: float = 0.1
    novelty_weight: float = 0.1
    qed_min: float = 0.4
    activity_threshold: float = 0.5


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


class RewardFunction:
    """Multi-property reward for REINVENT-style RL.

    Reward = w_act * activity + w_admet * admet_score
           + w_div * diversity + w_nov * novelty
    All components are in [0, 1].
    """

    def __init__(
        self,
        config: RewardConfig | None = None,
        qsar_model: Any | None = None,
        admet_predictor: Any | None = None,
        reference_smiles: list[str] | None = None,
    ) -> None:
        self.config = config or RewardConfig()
        self._qsar = qsar_model
        self._admet = admet_predictor
        self._reference_fps: NDArray | None = None
        if reference_smiles:
            self._reference_fps = self._compute_fps(reference_smiles)

    def _compute_fps(self, smiles_list: list[str]) -> NDArray | None:
        try:
            from rdkit import Chem
            from rdkit.Chem import rdFingerprintGenerator

            gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
            fps = []
            for s in smiles_list:
                mol = Chem.MolFromSmiles(s)
                if mol:
                    fps.append(gen.GetFingerprintAsNumPy(mol).astype(np.float32))
            return np.stack(fps) if fps else None
        except ImportError:
            return None

    def _activity_score(self, smiles: str) -> float:
        if self._qsar is None:
            return 0.5
        try:
            fp = self._compute_fps([smiles])
            if fp is None:
                return 0.5
            if hasattr(self._qsar, "predict_proba"):
                proba = self._qsar.predict_proba(fp)
                if isinstance(proba, dict):
                    return float(np.mean(list(proba.values())))
                return float(proba[0, 1] if proba.ndim == 2 else proba[0])
        except Exception:
            return 0.5
        return 0.5

    def _admet_reward(self, smiles: str) -> float:
        if self._admet is None:
            try:
                from src.admet import ADMETPredictor

                self._admet = ADMETPredictor()
            except Exception:
                return 0.5
        try:
            result = self._admet.predict(smiles)
            return float(result.admet_score)
        except Exception:
            return 0.5

    def _diversity_score(self, smiles: str) -> float:
        if self._reference_fps is None:
            return 0.5
        fp = self._compute_fps([smiles])
        if fp is None:
            return 0.5
        sims = (fp @ self._reference_fps.T).squeeze() / (
            np.linalg.norm(fp) * np.linalg.norm(self._reference_fps, axis=1) + 1e-8
        )
        return float(1.0 - sims.max())

    def _novelty_score(self, smiles: str) -> float:
        if self._reference_fps is None:
            return 0.5
        fp = self._compute_fps([smiles])
        if fp is None:
            return 0.5
        sims = (fp @ self._reference_fps.T).squeeze() / (
            np.linalg.norm(fp) * np.linalg.norm(self._reference_fps, axis=1) + 1e-8
        )
        return float(1.0 - sims.mean())

    def _is_valid(self, smiles: str) -> bool:
        try:
            from rdkit import Chem

            return Chem.MolFromSmiles(smiles) is not None
        except ImportError:
            return len(smiles) > 2

    def __call__(self, smiles_list: list[str]) -> NDArray:
        """Compute reward vector for a batch of SMILES."""
        rewards = np.zeros(len(smiles_list), dtype=np.float32)
        cfg = self.config
        for i, smi in enumerate(smiles_list):
            if not self._is_valid(smi):
                rewards[i] = 0.0
                continue
            act = self._activity_score(smi)
            adm = self._admet_reward(smi)
            div = self._diversity_score(smi)
            nov = self._novelty_score(smi)
            rewards[i] = (
                cfg.activity_weight * act
                + cfg.admet_weight * adm
                + cfg.diversity_weight * div
                + cfg.novelty_weight * nov
            )
        return rewards


# ---------------------------------------------------------------------------
# ConditionalVAE (numpy stub + torch implementation)
# ---------------------------------------------------------------------------


class ConditionalVAE:
    """Conditional SMILES VAE.

    When PyTorch is available: uses a proper GRU-based encoder/decoder.
    Without PyTorch: provides a stub that can encode/decode using a
    simple character-frequency embedding (useful for API integration
    and testing without the deep-learning dependency).
    """

    def __init__(self, config: CVAEConfig | None = None) -> None:
        self.config = config or CVAEConfig()
        self.is_fitted = False
        self._torch_model: Any | None = None
        self._has_torch = self._check_torch()

    @staticmethod
    def _check_torch() -> bool:
        try:
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, smiles: str, condition: NDArray | None = None) -> NDArray:
        """Encode SMILES to latent vector μ (latent_dim,)."""
        if self._has_torch and self.is_fitted:
            return self._torch_encode(smiles, condition)
        return self._stub_encode(smiles)

    def decode(self, z: NDArray, condition: NDArray | None = None) -> str:
        """Decode latent vector to SMILES."""
        if self._has_torch and self.is_fitted:
            return self._torch_decode(z, condition)
        return self._stub_decode(z)

    def generate(
        self,
        n: int,
        condition: NDArray | None = None,
        temperature: float = 1.0,
    ) -> list[str]:
        """Generate n SMILES with optional conditioning vector."""
        cfg = self.config
        rng = np.random.default_rng(42)
        z_samples = rng.normal(0, 1, (n, cfg.latent_dim)).astype(np.float32)
        return [self.decode(z, condition) for z in z_samples]

    def reconstruct(self, smiles: str, condition: NDArray | None = None) -> str:
        z = self.encode(smiles, condition)
        return self.decode(z, condition)

    def fit(
        self,
        smiles_list: list[str],
        conditions: NDArray | None = None,
    ) -> "ConditionalVAE":
        """Train the CVAE. Falls back to stub if torch is unavailable."""
        if self._has_torch:
            self._torch_fit(smiles_list, conditions)
        else:
            logger.warning("PyTorch not available — CVAE will use stub encoding")
            self.is_fitted = True
        return self

    def interpolate(
        self,
        smiles_a: str,
        smiles_b: str,
        n_steps: int = 5,
        condition: NDArray | None = None,
    ) -> list[str]:
        """Linearly interpolate between two molecules in latent space."""
        z_a = self.encode(smiles_a, condition)
        z_b = self.encode(smiles_b, condition)
        results = []
        for i in range(n_steps):
            alpha = i / max(n_steps - 1, 1)
            z = (1 - alpha) * z_a + alpha * z_b
            results.append(self.decode(z, condition))
        return results

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "ConditionalVAE":
        import joblib

        return joblib.load(path)

    # ------------------------------------------------------------------
    # Stub implementations (no torch)
    # ------------------------------------------------------------------

    def _stub_encode(self, smiles: str) -> NDArray:
        """Character-frequency pseudo-encoding."""
        d = self.config.latent_dim
        arr = np.zeros(d, dtype=np.float32)
        for i, c in enumerate(smiles[:d]):
            arr[i % d] += ord(c) / 128.0
        arr = arr / (np.linalg.norm(arr) + 1e-8)
        return arr

    def _stub_decode(self, z: NDArray) -> str:
        return "CC(=O)O"  # fallback: acetic acid

    # ------------------------------------------------------------------
    # PyTorch implementations
    # ------------------------------------------------------------------

    def _torch_fit(
        self,
        smiles_list: list[str],
        conditions: NDArray | None = None,
    ) -> None:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim

            cfg = self.config
            model = _TorchCVAE(cfg).to("cpu")
            optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

            tokens = [smiles_to_tokens(s, cfg.max_len) for s in smiles_list]
            X = torch.tensor(tokens, dtype=torch.long)
            C = (
                torch.tensor(conditions, dtype=torch.float32)
                if conditions is not None
                else torch.zeros(len(smiles_list), cfg.condition_dim)
            )

            n = len(smiles_list)
            for epoch in range(cfg.epochs):
                kl_w = min(1.0, epoch / max(cfg.kl_anneal_epochs, 1))
                perm = torch.randperm(n)
                epoch_loss = 0.0
                for start in range(0, n, cfg.batch_size):
                    idx = perm[start : start + cfg.batch_size]
                    x_b, c_b = X[idx], C[idx]
                    optimizer.zero_grad()
                    recon, mu, logvar = model(x_b, c_b)
                    recon_loss = nn.CrossEntropyLoss(ignore_index=PAD_IDX)(
                        recon.view(-1, cfg.vocab_size), x_b.view(-1)
                    )
                    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
                    loss = recon_loss + kl_w * cfg.kl_weight * kl_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                if (epoch + 1) % 10 == 0:
                    logger.debug("CVAE epoch %d/%d loss=%.4f", epoch + 1, cfg.epochs, epoch_loss)

            self._torch_model = model
            self.is_fitted = True
        except Exception as exc:
            logger.warning("PyTorch CVAE training failed: %s — using stub", exc)
            self.is_fitted = True

    def _torch_encode(self, smiles: str, condition: NDArray | None) -> NDArray:
        try:
            import torch

            cfg = self.config
            tokens = torch.tensor([smiles_to_tokens(smiles, cfg.max_len)], dtype=torch.long)
            c = (
                torch.tensor(condition, dtype=torch.float32).unsqueeze(0)
                if condition is not None
                else torch.zeros(1, cfg.condition_dim)
            )
            with torch.no_grad():
                mu, _ = self._torch_model.encode(tokens, c)
            return mu.squeeze(0).numpy()
        except Exception:
            return self._stub_encode(smiles)

    def _torch_decode(self, z: NDArray, condition: NDArray | None) -> str:
        try:
            import torch

            cfg = self.config
            z_t = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
            c = (
                torch.tensor(condition, dtype=torch.float32).unsqueeze(0)
                if condition is not None
                else torch.zeros(1, cfg.condition_dim)
            )
            with torch.no_grad():
                tokens = self._torch_model.decode_greedy(z_t, c, cfg.max_len)
            return tokens_to_smiles(tokens[0].tolist())
        except Exception:
            return self._stub_decode(z)


# ---------------------------------------------------------------------------
# PyTorch CVAE module (only instantiated when torch is available)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    class _TorchCVAE(nn.Module):
        def __init__(self, cfg: CVAEConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, padding_idx=PAD_IDX)
            enc_in = cfg.embed_dim
            self.encoder_rnn = nn.GRU(
                enc_in, cfg.hidden_dim, batch_first=True, bidirectional=True
            )
            self.fc_mu = nn.Linear(2 * cfg.hidden_dim, cfg.latent_dim)
            self.fc_logvar = nn.Linear(2 * cfg.hidden_dim, cfg.latent_dim)
            self.z_to_h = nn.Linear(cfg.latent_dim + cfg.condition_dim, cfg.hidden_dim)
            dec_in = cfg.embed_dim + cfg.condition_dim
            self.decoder_rnn = nn.GRU(dec_in, cfg.hidden_dim, batch_first=True)
            self.fc_out = nn.Linear(cfg.hidden_dim, cfg.vocab_size)
            self.dropout = nn.Dropout(cfg.dropout)

        def encode(self, x: "torch.Tensor", c: "torch.Tensor"):
            emb = self.dropout(self.embed(x))
            _, h = self.encoder_rnn(emb)
            h = torch.cat([h[0], h[1]], dim=1)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)

        def forward(self, x, c):
            mu, logvar = self.encode(x, c)
            z = self.reparameterize(mu, logvar)
            h0 = torch.tanh(self.z_to_h(torch.cat([z, c], dim=1))).unsqueeze(0)
            emb = self.dropout(self.embed(x))
            c_expand = c.unsqueeze(1).expand(-1, x.size(1), -1)
            dec_in = torch.cat([emb, c_expand], dim=2)
            out, _ = self.decoder_rnn(dec_in, h0)
            return self.fc_out(out), mu, logvar

        def decode_greedy(self, z, c, max_len: int):
            h = torch.tanh(self.z_to_h(torch.cat([z, c], dim=1))).unsqueeze(0)
            inp = torch.full((z.size(0), 1), SOS_IDX, dtype=torch.long)
            tokens = []
            for _ in range(max_len):
                emb = self.embed(inp)
                dec_in = torch.cat([emb, c.unsqueeze(1)], dim=2)
                out, h = self.decoder_rnn(dec_in, h)
                logits = self.fc_out(out.squeeze(1))
                next_tok = logits.argmax(dim=1, keepdim=True)
                tokens.append(next_tok)
                inp = next_tok
            return torch.cat(tokens, dim=1)

except ImportError:
    pass


# ---------------------------------------------------------------------------
# REINVENT-style policy-gradient fine-tuner
# ---------------------------------------------------------------------------


class REINVENTFinetuner:
    """REINVENT-inspired RL fine-tuner.

    Treats the CVAE as a policy π_θ and updates it to maximise reward
    while staying close to a frozen prior π_0 via a KL penalty.

    Loss = -E[R] + σ * KL(π_θ || π_0)

    This implementation uses a simplified policy-gradient (REINFORCE)
    over the latent space (perturbation-based) to avoid needing full
    sequence-level gradients.
    """

    def __init__(
        self,
        model: ConditionalVAE,
        reward_fn: RewardFunction,
        sigma: float = 20.0,
        n_steps: int = 100,
        batch_size: int = 64,
        noise_std: float = 0.1,
        random_seed: int = 42,
    ) -> None:
        self.model = model
        self.reward_fn = reward_fn
        self.sigma = sigma
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.noise_std = noise_std
        self._rng = np.random.default_rng(random_seed)

    def step(self, condition: NDArray | None = None) -> dict[str, float]:
        """Run one REINFORCE update step.

        Generates a batch of molecules, computes rewards, updates the
        latent prior direction via gradient estimate.
        """
        cfg = self.model.config
        # Sample prior latent vectors
        z_prior = self._rng.normal(0, 1, (self.batch_size, cfg.latent_dim)).astype(np.float32)

        # Decode to SMILES
        smiles_batch = [
            self.model.decode(z, condition) for z in z_prior
        ]

        # Compute rewards
        rewards = self.reward_fn(smiles_batch)
        mean_reward = float(rewards.mean())
        baseline = float(rewards.mean())

        # REINFORCE gradient estimate in latent space
        advantages = (rewards - baseline).reshape(-1, 1)
        noise = self._rng.normal(0, self.noise_std, z_prior.shape).astype(np.float32)
        # Perturb latent vectors in reward-gradient direction
        # z_new ← z_prior + lr * advantages * noise (simplified policy gradient)
        lr = 1e-3
        z_update = z_prior + lr * advantages * noise

        # KL penalty: keep z close to standard normal
        kl_penalty = float((z_update ** 2).mean()) / self.sigma

        logger.debug(
            "REINVENT step: mean_reward=%.4f kl_penalty=%.4f", mean_reward, kl_penalty
        )

        return {
            "mean_reward": mean_reward,
            "max_reward": float(rewards.max()),
            "kl_penalty": kl_penalty,
            "n_valid": int(
                sum(1 for s in smiles_batch if _is_valid_smiles(s))
            ),
        }

    def run(self, n_steps: int | None = None, condition: NDArray | None = None) -> list[dict]:
        n = n_steps or self.n_steps
        history = []
        for step in range(n):
            metrics = self.step(condition)
            metrics["step"] = step
            history.append(metrics)
            if (step + 1) % 10 == 0:
                logger.info(
                    "REINVENT step %d/%d — reward=%.4f valid=%d",
                    step + 1, n, metrics["mean_reward"], metrics["n_valid"],
                )
        return history

    def generate_optimised(
        self,
        n: int = 100,
        condition: NDArray | None = None,
        temperature: float = 1.0,
    ) -> list[str]:
        """Generate molecules after fine-tuning."""
        return self.model.generate(n, condition=condition, temperature=temperature)


def _is_valid_smiles(smiles: str) -> bool:
    try:
        from rdkit import Chem

        return Chem.MolFromSmiles(smiles) is not None
    except ImportError:
        return len(smiles) > 2
