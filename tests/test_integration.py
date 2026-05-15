"""End-to-end integration tests on a mini dataset.

These tests run the full pipeline from raw SMILES → descriptors → QSAR
model → metrics, using a tiny synthetic dataset so they run in seconds
on any machine without GPU or large data files.

Run: pytest tests/test_integration.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


MINI_SMILES = [
    "CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCCC",
    "c1ccc(O)cc1", "CC(=O)Nc1ccc(O)cc1",
    "CN1CCC[C@H]1c1cccnc1", "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
    "O=C(O)c1ccccc1", "Cc1ccc(S(N)(=O)=O)cc1",
    "CCOC(=O)c1ccc(N)cc1", "c1ccc2ncccc2c1", "O=C1CCCCC1",
    "Cc1ccccc1", "c1ccc(Cl)cc1", "CC(N)=O",
    "CC(=O)c1ccccc1", "c1ccc(F)cc1", "CCCCO",
]


@pytest.fixture(scope="module")
def mini_df():
    np.random.seed(42)
    n = len(MINI_SMILES)
    df = pd.DataFrame({
        "smiles": MINI_SMILES,
        "pIC50": np.random.uniform(5.0, 9.0, n),
    })
    df["active"] = (df["pIC50"] >= 7.0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Pipeline stage 1: Data preprocessing
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
class TestDataPipeline:
    def test_smiles_validation_pipeline(self, mini_df):
        from data.chembl_loader import ChEMBLLoader
        loader = ChEMBLLoader.__new__(ChEMBLLoader)
        loader.target_id = "CHEMBL1849"
        loader.min_compounds = 5
        loader.random_seed = 42

        valid_mask = mini_df["smiles"].apply(loader.validate_smiles)
        assert valid_mask.all(), "All mini SMILES should be valid"

    def test_descriptor_calculation_pipeline(self, mini_df):
        from data.descriptor_calculator import DescriptorCalculator
        calc = DescriptorCalculator(lipinski=True, topological=True, extended=False)
        desc_df = calc.calculate_batch(mini_df["smiles"].tolist(), show_progress=False)

        assert len(desc_df) == len(mini_df)
        assert "MolWt" in desc_df.columns
        assert "LogP" in desc_df.columns
        assert not desc_df[["MolWt", "LogP"]].isna().any().any()

    def test_scaffold_split_pipeline(self, mini_df):
        from data.scaffold_split import scaffold_split_df
        train_df, val_df, test_df = scaffold_split_df(mini_df, frac_train=0.6, frac_val=0.2, frac_test=0.2)

        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(mini_df), "Scaffold split lost molecules"
        assert len(train_df) >= len(test_df), "Train set smaller than test set"


# ---------------------------------------------------------------------------
# Pipeline stage 2: QSAR training & evaluation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not installed")
class TestQSARPipeline:
    @pytest.fixture(scope="class")
    def descriptor_features(self, mini_df):
        from data.descriptor_calculator import DescriptorCalculator
        calc = DescriptorCalculator(lipinski=True, topological=True, extended=False)
        desc_df = calc.calculate_batch(mini_df["smiles"].tolist(), show_progress=False)
        feature_cols = [c for c in desc_df.columns if c != "smiles"]
        X = desc_df[feature_cols].fillna(0).values
        y_reg = mini_df["pIC50"].values
        y_cls = mini_df["active"].values
        return X, y_reg, y_cls, feature_cols

    def test_regression_pipeline(self, descriptor_features):
        from models.qsar_model import QSARModel
        from evaluation.metrics import calculate_metrics

        X, y_reg, _, _ = descriptor_features
        split = int(0.7 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y_reg[:split], y_reg[split:]

        model = QSARModel(task="regression", n_estimators=20, random_seed=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = calculate_metrics(y_test, y_pred)
        assert "r2" in metrics
        assert "rmse" in metrics
        assert metrics["rmse"] > 0

    def test_classification_pipeline(self, descriptor_features):
        from models.qsar_model import QSARModel
        from evaluation.metrics import calculate_classification_metrics

        X, _, y_cls, _ = descriptor_features

        # Ensure both classes present in train
        split = int(0.6 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y_cls[:split], y_cls[split:]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            pytest.skip("Mini dataset too small for balanced classification test")

        model = QSARModel(task="classification", n_estimators=20, random_seed=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = calculate_classification_metrics(y_test, y_pred, y_proba)
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Pipeline stage 3: Cross-validation
# ---------------------------------------------------------------------------

class TestCrossValidationPipeline:
    def test_cv_returns_expected_keys(self, feature_matrix):
        from evaluation.cross_validation import cross_validate_model
        from sklearn.ensemble import RandomForestRegressor

        X, y_reg, _ = feature_matrix
        model = RandomForestRegressor(n_estimators=10, random_seed=42) \
            if hasattr(RandomForestRegressor, 'random_seed') \
            else RandomForestRegressor(n_estimators=10, random_state=42)

        results = cross_validate_model(model, X, y_reg, n_folds=3, task="regression")

        assert "r2_mean" in results
        assert "r2_std" in results
        assert "overall" in results
        assert results["n_folds"] == 3

    def test_cv_no_data_leakage(self, feature_matrix):
        """Predictions array must cover every sample exactly once."""
        from evaluation.cross_validation import cross_validate_model
        from sklearn.ensemble import RandomForestRegressor

        X, y_reg, _ = feature_matrix
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        results = cross_validate_model(
            model, X, y_reg, n_folds=3, task="regression", return_predictions=True
        )
        assert len(results["predictions"]) == len(y_reg)
        assert np.isfinite(results["predictions"]).all()


# ---------------------------------------------------------------------------
# Pipeline stage 4: Metrics utilities
# ---------------------------------------------------------------------------

class TestMetricsPipeline:
    def test_enrichment_factor_better_than_random(self):
        from evaluation.metrics import calculate_enrichment_factor
        np.random.seed(42)
        y_true = np.array([1] * 20 + [0] * 80)
        y_proba = np.where(y_true == 1,
                           np.random.uniform(0.7, 1.0, 100),
                           np.random.uniform(0.0, 0.4, 100))
        ef = calculate_enrichment_factor(y_true, y_proba, top_percent=0.1)
        assert ef > 1.0, f"EF should be > 1 for a good model, got {ef:.2f}"

    def test_bedroc_random_model(self):
        from evaluation.metrics import calculate_bedroc
        np.random.seed(42)
        y_true = np.array([1] * 50 + [0] * 50)
        y_proba = np.random.uniform(0, 1, 100)
        bedroc = calculate_bedroc(y_true, y_proba)
        assert 0.0 <= bedroc <= 1.0

    def test_pr_auc_perfect_model(self):
        from evaluation.metrics import get_precision_recall_curve
        y_true = np.array([1, 1, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.1, 0.05])
        _, _, _, pr_auc = get_precision_recall_curve(y_true, y_proba)
        assert pr_auc > 0.9, f"PR-AUC for perfect model should be ~1, got {pr_auc:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
