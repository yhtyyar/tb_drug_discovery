"""Generation quality metrics for molecular generative models.

Standard metrics used in drug discovery to evaluate generative models:
- Validity: fraction of generated SMILES that parse as valid molecules
- Uniqueness: fraction of valid molecules that are unique
- Novelty: fraction of valid unique molecules not in the training set
- QED: drug-likeness score (0-1, higher is better)
- SA Score: synthetic accessibility (1=easy, 10=hard; lower is better)
- Diversity: mean pairwise Tanimoto distance between fingerprints
- Drug-likeness filters: Lipinski Ro5, Veber rules

References:
    Bickerton et al. (2012) Nature Chemistry — QED
    Ertl & Schuffenhauer (2009) J. Cheminformatics — SA Score
    MoleculeNet benchmark suite
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, QED, AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from rdkit.Chem import RDConfig
    import sys
    import os
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
    SA_AVAILABLE = True
except Exception:
    SA_AVAILABLE = False


def compute_validity(smiles_list: List[str]) -> Tuple[float, List[str]]:
    """Compute validity fraction of generated SMILES.

    Args:
        smiles_list: List of generated SMILES strings.

    Returns:
        Tuple of (validity_fraction, list_of_valid_smiles).
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit required for generation metrics")

    valid = []
    for smi in smiles_list:
        if not smi or not isinstance(smi, str):
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                Chem.SanitizeMol(mol)
                canonical = Chem.MolToSmiles(mol, canonical=True)
                if canonical:
                    valid.append(canonical)
        except Exception:
            pass

    validity = len(valid) / len(smiles_list) if smiles_list else 0.0
    return validity, valid


def compute_uniqueness(valid_smiles: List[str]) -> Tuple[float, List[str]]:
    """Compute uniqueness fraction among valid SMILES.

    Args:
        valid_smiles: List of valid canonical SMILES.

    Returns:
        Tuple of (uniqueness_fraction, deduplicated_list).
    """
    if not valid_smiles:
        return 0.0, []
    unique = list(set(valid_smiles))
    return len(unique) / len(valid_smiles), unique


def compute_novelty(
    unique_smiles: List[str],
    training_smiles: Set[str],
) -> Tuple[float, List[str]]:
    """Compute novelty fraction vs. training set.

    Args:
        unique_smiles: Deduplicated valid generated SMILES.
        training_smiles: Set of canonical SMILES from training data.

    Returns:
        Tuple of (novelty_fraction, novel_smiles_list).
    """
    if not unique_smiles:
        return 0.0, []
    novel = [s for s in unique_smiles if s not in training_smiles]
    return len(novel) / len(unique_smiles), novel


def compute_qed_distribution(smiles_list: List[str]) -> Dict[str, float]:
    """Compute QED distribution statistics for a list of molecules.

    QED (Quantitative Estimate of Drug-likeness) ranges from 0 to 1,
    with higher values indicating more drug-like properties.

    Args:
        smiles_list: List of valid SMILES strings.

    Returns:
        Dictionary with mean, std, median, and fraction > 0.5.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit required for QED")

    scores = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                scores.append(QED.qed(mol))
        except Exception:
            pass

    if not scores:
        return {"qed_mean": 0.0, "qed_std": 0.0, "qed_median": 0.0, "qed_gt05": 0.0}

    arr = np.array(scores)
    return {
        "qed_mean": float(np.mean(arr)),
        "qed_std": float(np.std(arr)),
        "qed_median": float(np.median(arr)),
        "qed_gt05": float((arr > 0.5).mean()),
        "n_scored": len(scores),
    }


def compute_sa_score_distribution(smiles_list: List[str]) -> Dict[str, float]:
    """Compute SA score distribution (requires RDKit SA_Score contrib).

    SA score ranges 1-10; values < 6 are considered synthetically accessible.

    Args:
        smiles_list: List of valid SMILES strings.

    Returns:
        Dictionary with mean, std, and fraction with SA < 6.
    """
    if not SA_AVAILABLE:
        logger.warning("SA Score not available — install RDKit contrib SA_Score")
        return {"sa_available": False}

    scores = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                scores.append(sascorer.calculateScore(mol))
        except Exception:
            pass

    if not scores:
        return {"sa_mean": 0.0, "sa_std": 0.0, "sa_lt6": 0.0}

    arr = np.array(scores)
    return {
        "sa_mean": float(np.mean(arr)),
        "sa_std": float(np.std(arr)),
        "sa_median": float(np.median(arr)),
        "sa_lt6": float((arr < 6.0).mean()),
        "n_scored": len(scores),
    }


def compute_diversity(smiles_list: List[str], n_sample: int = 1000) -> float:
    """Compute mean pairwise Tanimoto distance (diversity measure).

    Samples up to n_sample molecules to keep runtime manageable.

    Args:
        smiles_list: List of valid SMILES strings.
        n_sample: Maximum number of molecules to sample.

    Returns:
        Mean pairwise Tanimoto distance (0=identical, 1=maximally diverse).
    """
    if not RDKIT_AVAILABLE or len(smiles_list) < 2:
        return 0.0

    sample = smiles_list[:n_sample] if len(smiles_list) > n_sample else smiles_list
    fps = []
    for smi in sample:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))

    if len(fps) < 2:
        return 0.0

    distances = []
    for i in range(min(len(fps), 200)):
        for j in range(i + 1, min(len(fps), 200)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            distances.append(1.0 - sim)

    return float(np.mean(distances)) if distances else 0.0


def compute_lipinski_compliance(smiles_list: List[str]) -> Dict[str, float]:
    """Compute fraction of molecules passing Lipinski Rule of 5.

    Ro5 criteria: MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10.

    Args:
        smiles_list: List of valid SMILES strings.

    Returns:
        Dictionary with pass rate and per-rule statistics.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit required for Lipinski check")

    passed = 0
    total = 0
    mw_ok = logp_ok = hbd_ok = hba_ok = 0

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        total += 1
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        r_mw = mw <= 500
        r_logp = logp <= 5
        r_hbd = hbd <= 5
        r_hba = hba <= 10

        mw_ok += r_mw
        logp_ok += r_logp
        hbd_ok += r_hbd
        hba_ok += r_hba
        passed += all([r_mw, r_logp, r_hbd, r_hba])

    if total == 0:
        return {"ro5_pass_rate": 0.0}

    return {
        "ro5_pass_rate": passed / total,
        "mw_ok_rate": mw_ok / total,
        "logp_ok_rate": logp_ok / total,
        "hbd_ok_rate": hbd_ok / total,
        "hba_ok_rate": hba_ok / total,
        "n_evaluated": total,
    }


def evaluate_generation(
    generated_smiles: List[str],
    training_smiles: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Full evaluation suite for a generative model run.

    Computes all standard metrics in one call. This is the primary
    function to use in training loops and dashboard logging.

    Args:
        generated_smiles: Raw SMILES from the generative model.
        training_smiles: Training set SMILES for novelty calculation.
        verbose: Log summary to console.

    Returns:
        Flat dictionary of all metrics, ready for MLflow/W&B logging.

    Example:
        >>> smiles = model.generate(100)
        >>> metrics = evaluate_generation(smiles, train_smiles)
        >>> print(f"Validity={metrics['validity']:.2%}, QED={metrics['qed_mean']:.3f}")
    """
    results: Dict[str, float] = {"n_generated": len(generated_smiles)}

    # Validity
    validity, valid_smiles = compute_validity(generated_smiles)
    results["validity"] = validity
    results["n_valid"] = len(valid_smiles)

    if not valid_smiles:
        logger.warning("No valid molecules generated")
        return results

    # Uniqueness
    uniqueness, unique_smiles = compute_uniqueness(valid_smiles)
    results["uniqueness"] = uniqueness
    results["n_unique"] = len(unique_smiles)

    # Novelty
    if training_smiles is not None:
        train_set = set(training_smiles)
        novelty, novel_smiles = compute_novelty(unique_smiles, train_set)
        results["novelty"] = novelty
        results["n_novel"] = len(novel_smiles)
    else:
        results["novelty"] = float("nan")

    # QED
    qed_stats = compute_qed_distribution(unique_smiles)
    results.update(qed_stats)

    # SA Score
    sa_stats = compute_sa_score_distribution(unique_smiles)
    results.update(sa_stats)

    # Diversity
    results["diversity"] = compute_diversity(unique_smiles)

    # Lipinski
    lipinski_stats = compute_lipinski_compliance(unique_smiles)
    results.update(lipinski_stats)

    if verbose:
        logger.info(
            f"Generation metrics: "
            f"validity={validity:.1%}, "
            f"uniqueness={uniqueness:.1%}, "
            f"novelty={results.get('novelty', float('nan')):.1%}, "
            f"QED={results.get('qed_mean', 0):.3f}, "
            f"Ro5={results.get('ro5_pass_rate', 0):.1%}, "
            f"diversity={results.get('diversity', 0):.3f}"
        )

    return results
