"""src.knowledge_graph — Biomedical knowledge graph for TB drug repurposing.

Entities: Molecule, Protein, Disease, Pathway, SideEffect
Relations: INHIBITS, ACTIVATES, TREATS, CAUSES, PART_OF, SIMILAR_TO

Link prediction via TransE / RotatE embeddings enables:
  - Drug repurposing (molecule → treats → disease)
  - Target identification (molecule → inhibits → protein)
  - Polypharmacology prediction
"""

from .graph import (
    Entity,
    EntityType,
    KnowledgeGraph,
    Relation,
    RelationType,
    load_graph,
    save_graph,
)
from .embeddings import (
    RotatEEmbedding,
    TransEEmbedding,
    train_embeddings,
)
from .repurposing import DrugRepurposingEngine

__all__ = [
    "Entity",
    "EntityType",
    "Relation",
    "RelationType",
    "KnowledgeGraph",
    "load_graph",
    "save_graph",
    "TransEEmbedding",
    "RotatEEmbedding",
    "train_embeddings",
    "DrugRepurposingEngine",
]
