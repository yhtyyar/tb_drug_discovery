"""Shared pytest fixtures for the TB Drug Discovery test suite."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Molecule fixtures
# ---------------------------------------------------------------------------

MINI_SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "CCN",
    "CCCC",
    "c1ccc(O)cc1",
    "CC(=O)Nc1ccc(O)cc1",  # Paracetamol
    "CN1CCC[C@H]1c1cccnc1",  # Nicotine
    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # Ibuprofen
    "O=C(O)c1ccccc1",  # Benzoic acid
    "Cc1ccc(S(N)(=O)=O)cc1",  # Sulfonamide
    "CCOC(=O)c1ccc(N)cc1",  # Ethyl 4-aminobenzoate
    "CC1=CC2=CC=CC=C2N1",
    "c1ccc2ncccc2c1",  # Quinoline
    "O=C1CCCCC1",  # Cyclohexanone
]


@pytest.fixture(scope="session")
def mini_smiles():
    """Small set of valid SMILES for fast tests."""
    return MINI_SMILES.copy()


@pytest.fixture(scope="session")
def mini_dataframe():
    """Mini DataFrame mimicking preprocessed ChEMBL output."""
    np.random.seed(42)
    n = len(MINI_SMILES)
    return pd.DataFrame({
        "smiles": MINI_SMILES,
        "pIC50": np.random.uniform(4.0, 9.0, n),
        "active": np.random.randint(0, 2, n),
    })


@pytest.fixture(scope="session")
def feature_matrix():
    """Synthetic feature matrix (200 samples × 50 features)."""
    np.random.seed(42)
    X = np.random.randn(200, 50)
    y_reg = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(200) * 0.2
    y_cls = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y_reg, y_cls
