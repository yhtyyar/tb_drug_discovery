"""Multi-objective molecular optimization.

Optimizes molecules for multiple properties simultaneously:
- Activity (pIC50 prediction)
- Drug-likeness (QED)
- Synthetic accessibility (SA score)
- Novelty vs training set

Provides scoring functions and Pareto front utilities.
"""

from dataclasses import dataclass
from typing import Callable, List, Set, Optional, Tuple
import numpy as np


@dataclass
class MolecularScore:
    """Multi-objective score for a generated molecule.

    Attributes:
        smiles: SMILES string of the molecule.
        activity: Predicted pIC50 (higher = better).
        qed: Drug-likeness score 0-1 (higher = better).
        sa_score: Synthetic accessibility 1-10 (lower = better).
        novelty: 1.0 if not in training set, else 0.0.
    """
    smiles: str
    activity: float
    qed: float
    sa_score: float
    novelty: float

    def weighted_sum(
        self,
        w_activity: float = 0.5,
        w_qed: float = 0.3,
        w_sa: float = 0.1,
        w_novelty: float = 0.1,
    ) -> float:
        """Calculate weighted sum score (higher = better).

        SA score is normalized to 0-1 scale (inverted so higher = better).
        """
        # Normalize SA: 1 = best (easy to synthesize), 0 = worst
        sa_normalized = 1.0 - (self.sa_score - 1) / 9.0
        sa_normalized = np.clip(sa_normalized, 0, 1)

        return (
            w_activity * self.activity +
            w_qed * self.qed +
            w_sa * sa_normalized +
            w_novelty * self.novelty
        )

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "smiles": self.smiles,
            "activity": self.activity,
            "qed": self.qed,
            "sa_score": self.sa_score,
            "novelty": self.novelty,
            "weighted_score": self.weighted_sum(),
        }


def calculate_sa_score(mol) -> float:
    """Calculate synthetic accessibility score.

    Uses RDKit's SAscore if available, otherwise returns neutral value.

    Args:
        mol: RDKit molecule object.

    Returns:
        SA score (1-10, lower = easier to synthesize).
    """
    try:
        from rdkit.Chem import RDConfig
        import os
        import sys

        # Add Contrib directory to path
        contrib_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if contrib_path not in sys.path:
            sys.path.append(contrib_path)

        import sascorer
        return sascorer.calculateScore(mol)
    except Exception:
        # Fallback: return neutral score if SA scorer not available
        return 5.0


def score_molecules(
    smiles_list: List[str],
    activity_predictor: Callable[[List[str]], np.ndarray],
    training_smiles: Set[str],
    skip_invalid: bool = True,
) -> List[MolecularScore]:
    """Score generated molecules on all objectives.

    Args:
        smiles_list: List of SMILES strings to score.
        activity_predictor: Function that takes SMILES list and returns
            predicted activity values (e.g., pIC50).
        training_smiles: Set of training SMILES for novelty calculation.
        skip_invalid: If True, skip invalid SMILES; otherwise score with defaults.

    Returns:
        List of MolecularScore objects (only valid molecules if skip_invalid=True).
    """
    from rdkit import Chem
    from rdkit.Chem import QED

    scores = []
    valid_smiles = []
    valid_mols = []

    # Filter valid molecules
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(smi)
            valid_mols.append(mol)
        elif not skip_invalid:
            # Include with zero scores
            scores.append(MolecularScore(
                smiles=smi,
                activity=0.0,
                qed=0.0,
                sa_score=10.0,
                novelty=0.0,
            ))

    if not valid_mols:
        return scores

    # Predict activity for all valid molecules
    activities = activity_predictor(valid_smiles)

    # Calculate scores for each molecule
    for smi, mol, activity in zip(valid_smiles, valid_mols, activities):
        # QED (drug-likeness)
        try:
            qed = QED.qed(mol)
        except Exception:
            qed = 0.0

        # SA score (synthetic accessibility)
        sa = calculate_sa_score(mol)

        # Novelty
        novelty = 1.0 if smi not in training_smiles else 0.0

        scores.append(MolecularScore(
            smiles=smi,
            activity=float(activity),
            qed=float(qed),
            sa_score=float(sa),
            novelty=float(novelty),
        ))

    return scores


def select_pareto_optimal(scores: List[MolecularScore]) -> List[MolecularScore]:
    """Select Pareto-optimal molecules.

    A molecule is Pareto-optimal if no other molecule is better
    in all objectives.

    Args:
        scores: List of MolecularScore objects.

    Returns:
        Subset of Pareto-optimal molecules.
    """
    if not scores:
        return []

    pareto = []
    for i, s1 in enumerate(scores):
        dominated = False
        for j, s2 in enumerate(scores):
            if i == j:
                continue
            # s2 dominates s1 if s2 is >= in all objectives and > in at least one
            # (remember: higher activity, qed, novelty = better; lower sa = better)
            if (
                s2.activity >= s1.activity and
                s2.qed >= s1.qed and
                (10 - s2.sa_score) >= (10 - s1.sa_score) and  # inverted SA
                s2.novelty >= s1.novelty and
                (
                    s2.activity > s1.activity or
                    s2.qed > s1.qed or
                    s2.sa_score < s1.sa_score or
                    s2.novelty > s1.novelty
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(s1)

    return pareto


def rank_molecules(
    scores: List[MolecularScore],
    method: str = "weighted_sum",
    w_activity: float = 0.5,
    w_qed: float = 0.3,
    w_sa: float = 0.1,
    w_novelty: float = 0.1,
) -> List[Tuple[MolecularScore, float]]:
    """Rank molecules by specified method.

    Args:
        scores: List of MolecularScore objects.
        method: 'weighted_sum', 'pareto', or 'activity'.
        w_activity: Weight for activity (weighted_sum method).
        w_qed: Weight for QED.
        w_sa: Weight for SA score.
        w_novelty: Weight for novelty.

    Returns:
        List of (score, rank_value) tuples, sorted by rank_value descending.
    """
    if method == "weighted_sum":
        ranked = [(s, s.weighted_sum(w_activity, w_qed, w_sa, w_novelty)) for s in scores]
    elif method == "activity":
        ranked = [(s, s.activity) for s in scores]
    elif method == "pareto":
        pareto_set = set(select_pareto_optimal(scores))
        # Pareto-optimal gets 1.0, others ranked by weighted sum
        ranked = [(s, 1.0 if s in pareto_set else s.weighted_sum()) for s in scores]
    else:
        raise ValueError(f"Unknown ranking method: {method}")

    # Sort by rank value descending
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def filter_molecules(
    scores: List[MolecularScore],
    min_activity: Optional[float] = None,
    min_qed: Optional[float] = None,
    max_sa: Optional[float] = None,
    require_novel: bool = False,
) -> List[MolecularScore]:
    """Filter molecules by thresholds.

    Args:
        scores: List of MolecularScore objects.
        min_activity: Minimum activity threshold.
        min_qed: Minimum QED threshold.
        max_sa: Maximum SA score threshold.
        require_novel: If True, only return novel molecules.

    Returns:
        Filtered list of molecules meeting all criteria.
    """
    filtered = scores

    if min_activity is not None:
        filtered = [s for s in filtered if s.activity >= min_activity]

    if min_qed is not None:
        filtered = [s for s in filtered if s.qed >= min_qed]

    if max_sa is not None:
        filtered = [s for s in filtered if s.sa_score <= max_sa]

    if require_novel:
        filtered = [s for s in filtered if s.novelty == 1.0]

    return filtered
