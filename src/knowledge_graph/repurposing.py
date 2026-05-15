"""Drug repurposing engine using KG embeddings.

Identifies candidate molecules for TB treatment by link prediction:
  molecule --[treats]--> disease

Also supports target identification:
  molecule --[inhibits]--> protein
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .graph import EntityType, KnowledgeGraph, RelationType
from .embeddings import TransEEmbedding, RotatEEmbedding

logger = logging.getLogger(__name__)


@dataclass
class RepurposingCandidate:
    molecule_id: str
    molecule_name: str
    target_id: str
    target_name: str
    relation: str
    score: float
    evidence: str = ""


class DrugRepurposingEngine:
    """Query KG embeddings for drug repurposing candidates.

    Usage
    -----
    engine = DrugRepurposingEngine(kg, embedding)
    candidates = engine.find_treats_candidates("MESH:D014397", top_k=20)
    candidates = engine.find_target_for_molecule("DB00951", top_k=5)
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        embedding: TransEEmbedding | RotatEEmbedding,
    ) -> None:
        self.kg = kg
        self.embedding = embedding

    def find_treats_candidates(
        self,
        disease_id: str,
        top_k: int = 20,
    ) -> list[RepurposingCandidate]:
        """Find molecules predicted to treat the given disease."""
        molecules = self.kg.entities_by_type(EntityType.MOLECULE)
        if not molecules:
            return []

        relation = RelationType.TREATS.value
        scored = []
        for mol in molecules:
            score = self.embedding.score(mol.entity_id, relation, disease_id)
            disease_ent = self.kg.get_entity(disease_id)
            scored.append(
                RepurposingCandidate(
                    molecule_id=mol.entity_id,
                    molecule_name=mol.name,
                    target_id=disease_id,
                    target_name=disease_ent.name if disease_ent else disease_id,
                    relation=relation,
                    score=score,
                )
            )

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]

    def find_target_for_molecule(
        self,
        molecule_id: str,
        top_k: int = 10,
    ) -> list[RepurposingCandidate]:
        """Predict protein targets for a given molecule."""
        proteins = self.kg.entities_by_type(EntityType.PROTEIN)
        if not proteins:
            return []

        relation = RelationType.INHIBITS.value
        mol_ent = self.kg.get_entity(molecule_id)
        scored = []
        for prot in proteins:
            score = self.embedding.score(molecule_id, relation, prot.entity_id)
            scored.append(
                RepurposingCandidate(
                    molecule_id=molecule_id,
                    molecule_name=mol_ent.name if mol_ent else molecule_id,
                    target_id=prot.entity_id,
                    target_name=prot.name,
                    relation=relation,
                    score=score,
                )
            )

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]

    def similar_drugs(
        self,
        molecule_id: str,
        top_k: int = 10,
    ) -> list[tuple[str, str, float]]:
        """Find structurally/functionally similar drugs in embedding space."""
        if not hasattr(self.embedding, "most_similar"):
            return []
        similar = self.embedding.most_similar(molecule_id, top_k=top_k)
        results = []
        for eid, sim in similar:
            ent = self.kg.get_entity(eid)
            if ent and ent.entity_type == EntityType.MOLECULE:
                results.append((eid, ent.name, sim))
        return results[:top_k]

    def candidates_to_dataframe(
        self, candidates: list[RepurposingCandidate]
    ) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "molecule_id": c.molecule_id,
                "molecule_name": c.molecule_name,
                "target_id": c.target_id,
                "target_name": c.target_name,
                "relation": c.relation,
                "score": c.score,
            }
            for c in candidates
        ])
