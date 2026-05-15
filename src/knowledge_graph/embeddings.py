"""TransE and RotatE knowledge graph embedding models.

TransE  : h + r ≈ t (score = -||h + r - t||)
RotatE  : h ∘ r ≈ t in complex space (score = -||h ∘ r - t||)

Both are trained via negative sampling with self-adversarial weighting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    dim: int = 64
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 512
    negative_ratio: int = 5
    margin: float = 1.0
    random_seed: int = 42
    regularization: float = 1e-4


class TransEEmbedding:
    """TransE knowledge graph embedding.

    score(h, r, t) = -||h + r - t||_2
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self.entity_emb: NDArray | None = None
        self.relation_emb: NDArray | None = None
        self._entity_idx: dict[str, int] = {}
        self._relation_idx: dict[str, int] = {}
        self.is_fitted = False

    def fit(
        self,
        triples: list[tuple[str, str, str]],
        entity_idx: dict[str, int] | None = None,
        relation_idx: dict[str, int] | None = None,
    ) -> "TransEEmbedding":
        cfg = self.config
        rng = np.random.default_rng(cfg.random_seed)

        # Build index maps
        if entity_idx is not None:
            self._entity_idx = entity_idx
        else:
            all_ents = sorted({e for h, _, t in triples for e in [h, t]})
            self._entity_idx = {e: i for i, e in enumerate(all_ents)}

        if relation_idx is not None:
            self._relation_idx = relation_idx
        else:
            all_rels = sorted({r for _, r, _ in triples})
            self._relation_idx = {r: i for i, r in enumerate(all_rels)}

        n_e = len(self._entity_idx)
        n_r = len(self._relation_idx)

        if n_e == 0 or n_r == 0 or not triples:
            logger.warning("No entities/relations to embed")
            self.entity_emb = np.zeros((max(n_e, 1), cfg.dim), dtype=np.float32)
            self.relation_emb = np.zeros((max(n_r, 1), cfg.dim), dtype=np.float32)
            self.is_fitted = True
            return self

        self.entity_emb = rng.uniform(-0.1, 0.1, (n_e, cfg.dim)).astype(np.float32)
        self.relation_emb = rng.uniform(-0.1, 0.1, (n_r, cfg.dim)).astype(np.float32)

        # Normalise entity embeddings
        self.entity_emb /= np.linalg.norm(self.entity_emb, axis=1, keepdims=True) + 1e-8

        # Convert triples to index arrays
        h_idx = np.array([self._entity_idx[h] for h, _, _ in triples if h in self._entity_idx])
        r_idx = np.array([self._relation_idx[r] for _, r, _ in triples if r in self._relation_idx])
        t_idx = np.array([self._entity_idx[t] for _, _, t in triples if t in self._entity_idx])

        if len(h_idx) == 0:
            self.is_fitted = True
            return self

        n = len(h_idx)
        for epoch in range(cfg.epochs):
            perm = rng.permutation(n)
            for start in range(0, n, cfg.batch_size):
                idx = perm[start : start + cfg.batch_size]
                h_b = self.entity_emb[h_idx[idx]]
                r_b = self.relation_emb[r_idx[idx]]
                t_b = self.entity_emb[t_idx[idx]]

                # Negative sampling (corrupt head or tail)
                neg_idx = rng.integers(0, n_e, len(idx))
                if rng.random() > 0.5:
                    h_neg = self.entity_emb[neg_idx]
                    t_neg = t_b
                else:
                    h_neg = h_b
                    t_neg = self.entity_emb[neg_idx]

                # Margin ranking loss gradient
                pos_dist = np.linalg.norm(h_b + r_b - t_b, axis=1)
                neg_dist = np.linalg.norm(h_neg + r_b - t_neg, axis=1)
                loss_mask = (pos_dist - neg_dist + cfg.margin > 0).astype(np.float32)

                # Gradient w.r.t. positive triple
                d_pos = (h_b + r_b - t_b) / (pos_dist[:, None] + 1e-8)
                d_neg = (h_neg + r_b - t_neg) / (neg_dist[:, None] + 1e-8)

                grad = loss_mask[:, None] * (d_pos - d_neg)

                np.add.at(self.entity_emb, h_idx[idx], -cfg.learning_rate * grad)
                np.add.at(self.relation_emb, r_idx[idx], -cfg.learning_rate * grad)
                np.add.at(self.entity_emb, t_idx[idx], cfg.learning_rate * grad)

                # Regularisation
                self.entity_emb *= 1 - cfg.regularization
                self.relation_emb *= 1 - cfg.regularization

                # Re-normalise entities
                norms = np.linalg.norm(self.entity_emb, axis=1, keepdims=True)
                self.entity_emb /= norms + 1e-8

        self.is_fitted = True
        logger.info(
            "TransE fitted: %d entities, %d relations, dim=%d",
            n_e, n_r, cfg.dim,
        )
        return self

    def score(self, head_id: str, relation: str, tail_id: str) -> float:
        """Compute TransE score (higher = more plausible triple)."""
        if not self.is_fitted or self.entity_emb is None:
            return 0.0
        h = self.entity_emb[self._entity_idx.get(head_id, 0)]
        r = self.relation_emb[self._relation_idx.get(relation, 0)]
        t = self.entity_emb[self._entity_idx.get(tail_id, 0)]
        return float(-np.linalg.norm(h + r - t))

    def get_embedding(self, entity_id: str) -> NDArray | None:
        if self.entity_emb is None or entity_id not in self._entity_idx:
            return None
        return self.entity_emb[self._entity_idx[entity_id]].copy()

    def most_similar(
        self,
        entity_id: str,
        top_k: int = 10,
        entity_type_filter: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return top-k most similar entities by cosine similarity."""
        if self.entity_emb is None or entity_id not in self._entity_idx:
            return []
        query = self.entity_emb[self._entity_idx[entity_id]]
        norms = np.linalg.norm(self.entity_emb, axis=1) + 1e-8
        sims = (self.entity_emb @ query) / (norms * (np.linalg.norm(query) + 1e-8))
        top = np.argsort(sims)[::-1]
        idx_to_ent = {v: k for k, v in self._entity_idx.items()}
        results = []
        for i in top:
            eid = idx_to_ent.get(i, "")
            if eid == entity_id:
                continue
            results.append((eid, float(sims[i])))
            if len(results) >= top_k:
                break
        return results

    def link_predict(
        self,
        head_id: str,
        relation: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Predict most likely tail entities for (head, relation, ?) query."""
        if self.entity_emb is None or self.relation_emb is None:
            return []
        h_idx = self._entity_idx.get(head_id, 0)
        r_idx = self._relation_idx.get(relation, 0)
        h = self.entity_emb[h_idx]
        r = self.relation_emb[r_idx]
        target = h + r
        dists = np.linalg.norm(self.entity_emb - target, axis=1)
        top = np.argsort(dists)[:top_k]
        idx_to_ent = {v: k for k, v in self._entity_idx.items()}
        return [(idx_to_ent.get(i, ""), float(-dists[i])) for i in top]


class RotatEEmbedding:
    """RotatE: rotate entity embeddings in complex space.

    score(h, r, t) = -||h ∘ r - t||   (complex multiplication)
    Entities: ℂ^(d/2), Relations: unit-circle rotations e^(iθ_r)
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()
        self.entity_re: NDArray | None = None
        self.entity_im: NDArray | None = None
        self.relation_phase: NDArray | None = None
        self._entity_idx: dict[str, int] = {}
        self._relation_idx: dict[str, int] = {}
        self.is_fitted = False
        self._half_dim = max(self.config.dim // 2, 1)

    def fit(
        self,
        triples: list[tuple[str, str, str]],
        entity_idx: dict[str, int] | None = None,
        relation_idx: dict[str, int] | None = None,
    ) -> "RotatEEmbedding":
        cfg = self.config
        rng = np.random.default_rng(cfg.random_seed)
        d = self._half_dim

        if entity_idx is not None:
            self._entity_idx = entity_idx
        else:
            all_ents = sorted({e for h, _, t in triples for e in [h, t]})
            self._entity_idx = {e: i for i, e in enumerate(all_ents)}

        if relation_idx is not None:
            self._relation_idx = relation_idx
        else:
            all_rels = sorted({r for _, r, _ in triples})
            self._relation_idx = {r: i for i, r in enumerate(all_rels)}

        n_e = len(self._entity_idx)
        n_r = len(self._relation_idx)

        if n_e == 0 or not triples:
            self.entity_re = np.zeros((max(n_e, 1), d), dtype=np.float32)
            self.entity_im = np.zeros((max(n_e, 1), d), dtype=np.float32)
            self.relation_phase = np.zeros((max(n_r, 1), d), dtype=np.float32)
            self.is_fitted = True
            return self

        self.entity_re = rng.uniform(-0.1, 0.1, (n_e, d)).astype(np.float32)
        self.entity_im = rng.uniform(-0.1, 0.1, (n_e, d)).astype(np.float32)
        self.relation_phase = rng.uniform(-np.pi, np.pi, (n_r, d)).astype(np.float32)

        h_idx = np.array([self._entity_idx[h] for h, _, _ in triples if h in self._entity_idx])
        r_idx_arr = np.array([self._relation_idx[r] for _, r, _ in triples if r in self._relation_idx])
        t_idx = np.array([self._entity_idx[t] for _, _, t in triples if t in self._entity_idx])

        if len(h_idx) == 0:
            self.is_fitted = True
            return self

        n = len(h_idx)
        for epoch in range(cfg.epochs):
            perm = rng.permutation(n)
            for start in range(0, n, cfg.batch_size):
                idx = perm[start : start + cfg.batch_size]
                h_re = self.entity_re[h_idx[idx]]
                h_im = self.entity_im[h_idx[idx]]
                t_re = self.entity_re[t_idx[idx]]
                t_im = self.entity_im[t_idx[idx]]
                phase = self.relation_phase[r_idx_arr[idx]]
                cos_r, sin_r = np.cos(phase), np.sin(phase)

                # h ∘ r = (h_re*cos - h_im*sin, h_re*sin + h_im*cos)
                rot_re = h_re * cos_r - h_im * sin_r
                rot_im = h_re * sin_r + h_im * cos_r

                diff_re = rot_re - t_re
                diff_im = rot_im - t_im
                pos_dist = np.sqrt((diff_re ** 2 + diff_im ** 2).sum(axis=1) + 1e-8)

                neg_ent = rng.integers(0, n_e, len(idx))
                t_neg_re = self.entity_re[neg_ent]
                t_neg_im = self.entity_im[neg_ent]
                diff_neg_re = rot_re - t_neg_re
                diff_neg_im = rot_im - t_neg_im
                neg_dist = np.sqrt((diff_neg_re ** 2 + diff_neg_im ** 2).sum(axis=1) + 1e-8)

                mask = (pos_dist - neg_dist + cfg.margin > 0).astype(np.float32)
                lr = cfg.learning_rate
                grad_re = mask[:, None] * diff_re / pos_dist[:, None]
                grad_im = mask[:, None] * diff_im / pos_dist[:, None]

                np.add.at(self.entity_re, t_idx[idx], lr * grad_re)
                np.add.at(self.entity_im, t_idx[idx], lr * grad_im)

        self.is_fitted = True
        logger.info("RotatE fitted: %d entities, dim=%d", n_e, d)
        return self

    def score(self, head_id: str, relation: str, tail_id: str) -> float:
        if not self.is_fitted or self.entity_re is None:
            return 0.0
        h_i = self._entity_idx.get(head_id, 0)
        r_i = self._relation_idx.get(relation, 0)
        t_i = self._entity_idx.get(tail_id, 0)
        h_re = self.entity_re[h_i]
        h_im = self.entity_im[h_i]
        phase = self.relation_phase[r_i]
        t_re = self.entity_re[t_i]
        t_im = self.entity_im[t_i]
        rot_re = h_re * np.cos(phase) - h_im * np.sin(phase)
        rot_im = h_re * np.sin(phase) + h_im * np.cos(phase)
        return float(-np.sqrt(((rot_re - t_re) ** 2 + (rot_im - t_im) ** 2).sum()))

    def get_embedding(self, entity_id: str) -> NDArray | None:
        if self.entity_re is None or entity_id not in self._entity_idx:
            return None
        i = self._entity_idx[entity_id]
        return np.concatenate([self.entity_re[i], self.entity_im[i]])


def train_embeddings(
    kg: "KnowledgeGraph",
    model: str = "transe",
    config: EmbeddingConfig | None = None,
) -> TransEEmbedding | RotatEEmbedding:
    """Train KG embeddings from a KnowledgeGraph object."""
    triples = kg.triples()
    entity_idx = kg.entity_index()
    relation_idx = kg.relation_index()

    if model == "transe":
        emb = TransEEmbedding(config)
    elif model == "rotate":
        emb = RotatEEmbedding(config)
    else:
        raise ValueError(f"Unknown model: {model}. Choose 'transe' or 'rotate'")

    emb.fit(triples, entity_idx=entity_idx, relation_idx=relation_idx)
    return emb
