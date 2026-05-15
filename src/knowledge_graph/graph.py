"""Knowledge graph data structures and serialisation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    MOLECULE = "molecule"
    PROTEIN = "protein"
    DISEASE = "disease"
    PATHWAY = "pathway"
    SIDE_EFFECT = "side_effect"
    GENE = "gene"


class RelationType(str, Enum):
    INHIBITS = "inhibits"
    ACTIVATES = "activates"
    TREATS = "treats"
    CAUSES = "causes"
    PART_OF = "part_of"
    SIMILAR_TO = "similar_to"
    BINDS = "binds"
    EXPRESSED_IN = "expressed_in"
    ASSOCIATED_WITH = "associated_with"
    DRUG_TARGET = "drug_target"


@dataclass
class Entity:
    entity_id: str
    entity_type: EntityType
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.entity_id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Entity) and self.entity_id == other.entity_id


@dataclass
class Relation:
    head_id: str
    relation_type: RelationType
    tail_id: str
    weight: float = 1.0
    source: str = ""

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.head_id, self.relation_type.value, self.tail_id)


class KnowledgeGraph:
    """Biomedical knowledge graph with transductive link prediction support.

    Stores entities and relations; provides neighbourhood lookup,
    subgraph extraction, and import/export utilities.
    """

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._relations: list[Relation] = []
        self._adj: dict[str, list[Relation]] = {}  # head_id → outgoing relations
        self._rev_adj: dict[str, list[Relation]] = {}  # tail_id → incoming relations

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_entity(self, entity: Entity) -> None:
        self._entities[entity.entity_id] = entity
        self._adj.setdefault(entity.entity_id, [])
        self._rev_adj.setdefault(entity.entity_id, [])

    def add_relation(self, relation: Relation) -> None:
        for eid in [relation.head_id, relation.tail_id]:
            if eid not in self._entities:
                logger.warning("Entity '%s' not in graph — auto-adding as unknown", eid)
                self._entities[eid] = Entity(
                    entity_id=eid,
                    entity_type=EntityType.MOLECULE,
                    name=eid,
                )
        self._relations.append(relation)
        self._adj.setdefault(relation.head_id, []).append(relation)
        self._rev_adj.setdefault(relation.tail_id, []).append(relation)

    def add_tb_default_entities(self) -> None:
        """Populate with known TB drug targets and diseases."""
        tb_proteins = [
            ("CHEMBL1849", "InhA", {"function": "enoyl-ACP reductase", "organism": "M.tb"}),
            ("CHEMBL1790", "KatG", {"function": "catalase-peroxidase", "organism": "M.tb"}),
            ("CHEMBL1916", "rpoB", {"function": "RNA polymerase beta", "organism": "M.tb"}),
            ("CHEMBL3622", "DprE1", {"function": "DprE1 oxidoreductase", "organism": "M.tb"}),
            ("CHEMBL4296", "MmpL3", {"function": "mycolic acid transport", "organism": "M.tb"}),
        ]
        for eid, name, attrs in tb_proteins:
            self.add_entity(Entity(eid, EntityType.PROTEIN, name, attrs))

        tb_diseases = [
            ("MESH:D014397", "Tuberculosis", {"icd10": "A15-A19"}),
            ("MESH:D014399", "Pulmonary TB", {"icd10": "A15"}),
            ("MESH:D018088", "MDR-TB", {"icd10": "A15.0"}),
        ]
        for eid, name, attrs in tb_diseases:
            self.add_entity(Entity(eid, EntityType.DISEASE, name, attrs))

        # Add known first-line TB drugs
        drugs = [
            ("DB00951", "Isoniazid", {"mw": 137.14, "logp": -0.72}),
            ("DB01045", "Rifampicin", {"mw": 822.95, "logp": 4.24}),
            ("DB00608", "Chloroquine", {"mw": 319.87, "logp": 4.63}),
            ("DB00204", "Dofetilide", {"mw": 441.57, "logp": 1.12}),
        ]
        for eid, name, attrs in drugs:
            self.add_entity(Entity(eid, EntityType.MOLECULE, name, attrs))

        # Add known relations
        known_relations = [
            ("DB00951", RelationType.INHIBITS, "CHEMBL1849"),
            ("DB00951", RelationType.DRUG_TARGET, "CHEMBL1790"),
            ("DB01045", RelationType.INHIBITS, "CHEMBL1916"),
            ("DB00951", RelationType.TREATS, "MESH:D014397"),
            ("DB01045", RelationType.TREATS, "MESH:D014397"),
        ]
        for h, r, t in known_relations:
            self.add_relation(Relation(h, r, t, source="known_db"))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def neighbours(
        self,
        entity_id: str,
        relation_type: RelationType | None = None,
        direction: str = "out",
    ) -> list[Entity]:
        """Return neighbouring entities."""
        if direction == "out":
            rels = self._adj.get(entity_id, [])
            ids = [r.tail_id for r in rels if relation_type is None or r.relation_type == relation_type]
        else:
            rels = self._rev_adj.get(entity_id, [])
            ids = [r.head_id for r in rels if relation_type is None or r.relation_type == relation_type]
        return [self._entities[i] for i in ids if i in self._entities]

    def triples(self) -> list[tuple[str, str, str]]:
        return [r.triple for r in self._relations]

    def entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        return [e for e in self._entities.values() if e.entity_type == entity_type]

    def entity_ids(self) -> list[str]:
        return list(self._entities.keys())

    def relation_types(self) -> list[str]:
        return list({r.relation_type.value for r in self._relations})

    def stats(self) -> dict[str, int]:
        from collections import Counter

        type_counts = Counter(e.entity_type.value for e in self._entities.values())
        rel_counts = Counter(r.relation_type.value for r in self._relations)
        return {
            "n_entities": len(self._entities),
            "n_relations": len(self._relations),
            **{f"entity_{k}": v for k, v in type_counts.items()},
            **{f"relation_{k}": v for k, v in rel_counts.items()},
        }

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self._relations:
            h = self._entities.get(r.head_id)
            t = self._entities.get(r.tail_id)
            rows.append({
                "head_id": r.head_id,
                "head_name": h.name if h else r.head_id,
                "head_type": h.entity_type.value if h else "unknown",
                "relation": r.relation_type.value,
                "tail_id": r.tail_id,
                "tail_name": t.name if t else r.tail_id,
                "tail_type": t.entity_type.value if t else "unknown",
                "weight": r.weight,
                "source": r.source,
            })
        return pd.DataFrame(rows)

    def entity_index(self) -> dict[str, int]:
        """Map entity_id → integer index (for embedding lookup)."""
        return {eid: i for i, eid in enumerate(sorted(self._entities))}

    def relation_index(self) -> dict[str, int]:
        """Map relation_type → integer index."""
        all_types = sorted({r.relation_type.value for r in self._relations})
        return {rt: i for i, rt in enumerate(all_types)}

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "entities": [
                {
                    "entity_id": e.entity_id,
                    "entity_type": e.entity_type.value,
                    "name": e.name,
                    "attributes": e.attributes,
                }
                for e in self._entities.values()
            ],
            "relations": [
                {
                    "head_id": r.head_id,
                    "relation_type": r.relation_type.value,
                    "tail_id": r.tail_id,
                    "weight": r.weight,
                    "source": r.source,
                }
                for r in self._relations
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        kg = cls()
        for e in data["entities"]:
            kg.add_entity(
                Entity(
                    entity_id=e["entity_id"],
                    entity_type=EntityType(e["entity_type"]),
                    name=e["name"],
                    attributes=e.get("attributes", {}),
                )
            )
        for r in data["relations"]:
            kg.add_relation(
                Relation(
                    head_id=r["head_id"],
                    relation_type=RelationType(r["relation_type"]),
                    tail_id=r["tail_id"],
                    weight=r.get("weight", 1.0),
                    source=r.get("source", ""),
                )
            )
        return kg


def save_graph(kg: KnowledgeGraph, path: str) -> None:
    import joblib

    joblib.dump(kg, path)
    logger.info("KnowledgeGraph saved → %s (%d entities, %d relations)", path, len(kg._entities), len(kg._relations))


def load_graph(path: str) -> KnowledgeGraph:
    import joblib

    return joblib.load(path)
