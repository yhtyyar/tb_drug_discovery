"""Tests for Knowledge Graph, embeddings, and drug repurposing."""

from __future__ import annotations

import numpy as np
import pytest

from src.knowledge_graph import (
    DrugRepurposingEngine,
    Entity,
    EntityType,
    KnowledgeGraph,
    Relation,
    RelationType,
    RotatEEmbedding,
    TransEEmbedding,
    train_embeddings,
)
from src.knowledge_graph.embeddings import EmbeddingConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_entity(Entity("M1", EntityType.MOLECULE, "DrugA"))
    kg.add_entity(Entity("M2", EntityType.MOLECULE, "DrugB"))
    kg.add_entity(Entity("P1", EntityType.PROTEIN, "InhA"))
    kg.add_entity(Entity("P2", EntityType.PROTEIN, "KatG"))
    kg.add_entity(Entity("D1", EntityType.DISEASE, "TB"))
    kg.add_relation(Relation("M1", RelationType.INHIBITS, "P1", weight=1.0))
    kg.add_relation(Relation("M2", RelationType.INHIBITS, "P2", weight=0.8))
    kg.add_relation(Relation("M1", RelationType.TREATS, "D1", weight=1.0))
    kg.add_relation(Relation("P1", RelationType.ASSOCIATED_WITH, "D1", weight=1.0))
    return kg


@pytest.fixture()
def tb_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    kg.add_tb_default_entities()
    return kg


@pytest.fixture()
def tiny_cfg() -> EmbeddingConfig:
    return EmbeddingConfig(dim=8, epochs=5, batch_size=4, random_seed=0)


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------


class TestKnowledgeGraph:
    def test_add_entity(self, small_kg: KnowledgeGraph) -> None:
        assert small_kg.get_entity("M1") is not None
        assert small_kg.get_entity("P1").name == "InhA"

    def test_add_relation(self, small_kg: KnowledgeGraph) -> None:
        assert len(small_kg.triples()) == 4

    def test_neighbours_out(self, small_kg: KnowledgeGraph) -> None:
        nbrs = small_kg.neighbours("M1", direction="out")
        nbr_ids = {e.entity_id for e in nbrs}
        assert "P1" in nbr_ids or "D1" in nbr_ids

    def test_neighbours_filtered(self, small_kg: KnowledgeGraph) -> None:
        nbrs = small_kg.neighbours("M1", relation_type=RelationType.INHIBITS)
        assert all(isinstance(e, Entity) for e in nbrs)

    def test_entities_by_type(self, small_kg: KnowledgeGraph) -> None:
        mols = small_kg.entities_by_type(EntityType.MOLECULE)
        assert len(mols) == 2

    def test_stats(self, small_kg: KnowledgeGraph) -> None:
        s = small_kg.stats()
        assert s["n_entities"] == 5
        assert s["n_relations"] == 4

    def test_to_dataframe(self, small_kg: KnowledgeGraph) -> None:
        df = small_kg.to_dataframe()
        assert len(df) == 4
        assert "relation" in df.columns

    def test_round_trip_serialisation(self, small_kg: KnowledgeGraph) -> None:
        d = small_kg.to_dict()
        kg2 = KnowledgeGraph.from_dict(d)
        assert kg2.stats()["n_entities"] == small_kg.stats()["n_entities"]
        assert kg2.stats()["n_relations"] == small_kg.stats()["n_relations"]

    def test_default_tb_entities(self, tb_kg: KnowledgeGraph) -> None:
        proteins = tb_kg.entities_by_type(EntityType.PROTEIN)
        names = {p.name for p in proteins}
        assert "InhA" in names
        assert "KatG" in names

    def test_save_load(self, small_kg: KnowledgeGraph, tmp_path) -> None:
        from src.knowledge_graph import save_graph, load_graph

        path = str(tmp_path / "kg.joblib")
        save_graph(small_kg, path)
        kg2 = load_graph(path)
        assert kg2.stats()["n_entities"] == 5


# ---------------------------------------------------------------------------
# TransE
# ---------------------------------------------------------------------------


class TestTransEEmbedding:
    def test_fit(self, small_kg: KnowledgeGraph, tiny_cfg: EmbeddingConfig) -> None:
        emb = train_embeddings(small_kg, model="transe", config=tiny_cfg)
        assert emb.is_fitted

    def test_entity_embedding_shape(
        self, small_kg: KnowledgeGraph, tiny_cfg: EmbeddingConfig
    ) -> None:
        emb = train_embeddings(small_kg, model="transe", config=tiny_cfg)
        e = emb.get_embedding("M1")
        assert e is not None
        assert e.shape == (tiny_cfg.dim,)

    def test_unknown_entity_returns_none(
        self, small_kg: KnowledgeGraph, tiny_cfg: EmbeddingConfig
    ) -> None:
        emb = train_embeddings(small_kg, model="transe", config=tiny_cfg)
        assert emb.get_embedding("NONEXISTENT") is None

    def test_score_returns_float(
        self, small_kg: KnowledgeGraph, tiny_cfg: EmbeddingConfig
    ) -> None:
        emb = train_embeddings(small_kg, model="transe", config=tiny_cfg)
        s = emb.score("M1", "inhibits", "P1")
        assert isinstance(s, float)

    def test_most_similar(
        self, small_kg: KnowledgeGraph, tiny_cfg: EmbeddingConfig
    ) -> None:
        emb = train_embeddings(small_kg, model="transe", config=tiny_cfg)
        similar = emb.most_similar("M1", top_k=3)
        assert isinstance(similar, list)
        assert all(isinstance(sim, float) for _, sim in similar)

    def test_link_predict(
        self, small_kg: KnowledgeGraph, tiny_cfg: EmbeddingConfig
    ) -> None:
        emb = train_embeddings(small_kg, model="transe", config=tiny_cfg)
        preds = emb.link_predict("M1", "inhibits", top_k=3)
        assert isinstance(preds, list)


# ---------------------------------------------------------------------------
# RotatE
# ---------------------------------------------------------------------------


class TestRotatEEmbedding:
    def test_fit(self, small_kg: KnowledgeGraph, tiny_cfg: EmbeddingConfig) -> None:
        emb = train_embeddings(small_kg, model="rotate", config=tiny_cfg)
        assert emb.is_fitted

    def test_embedding_shape(
        self, small_kg: KnowledgeGraph, tiny_cfg: EmbeddingConfig
    ) -> None:
        emb = train_embeddings(small_kg, model="rotate", config=tiny_cfg)
        e = emb.get_embedding("M1")
        assert e is not None
        # RotatE returns re + im concatenated = 2 * half_dim
        assert e.shape[0] == tiny_cfg.dim


# ---------------------------------------------------------------------------
# DrugRepurposingEngine
# ---------------------------------------------------------------------------


class TestDrugRepurposing:
    @pytest.fixture()
    def engine(
        self, tb_kg: KnowledgeGraph, tiny_cfg: EmbeddingConfig
    ) -> DrugRepurposingEngine:
        emb = train_embeddings(tb_kg, model="transe", config=tiny_cfg)
        return DrugRepurposingEngine(tb_kg, emb)

    def test_find_treats_candidates(self, engine: DrugRepurposingEngine) -> None:
        candidates = engine.find_treats_candidates("MESH:D014397", top_k=5)
        assert isinstance(candidates, list)

    def test_find_target_for_molecule(self, engine: DrugRepurposingEngine) -> None:
        candidates = engine.find_target_for_molecule("DB00951", top_k=3)
        assert isinstance(candidates, list)

    def test_candidates_to_dataframe(self, engine: DrugRepurposingEngine) -> None:
        candidates = engine.find_treats_candidates("MESH:D014397", top_k=3)
        df = engine.candidates_to_dataframe(candidates)
        if len(candidates) > 0:
            assert "molecule_name" in df.columns
            assert "score" in df.columns
