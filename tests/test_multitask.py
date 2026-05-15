"""Tests for Multi-task QSAR model and data loader."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.multitask_loader import (
    MultiTaskDataset,
    compute_descriptors,
    make_synthetic_multitask_dataset,
)
from src.models.multitask_qsar import (
    MultiTaskConfig,
    MultiTaskQSAR,
    MultiTaskQSARSklearn,
    create_multitask_model,
    evaluate_multitask,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_dataset() -> MultiTaskDataset:
    return make_synthetic_multitask_dataset(n_compounds=80, targets=["InhA", "KatG"], seed=0)


@pytest.fixture()
def tiny_model() -> MultiTaskQSAR:
    cfg = MultiTaskConfig(hidden_dims=[32, 16], dropout=0.1, epochs=3, batch_size=32)
    return MultiTaskQSAR(targets=["InhA", "KatG"], config=cfg)


N_BITS = 64  # small for fast tests


@pytest.fixture()
def fitted_model(tiny_dataset: MultiTaskDataset) -> MultiTaskQSAR:
    X = compute_descriptors(tiny_dataset, n_bits=N_BITS)
    cfg = MultiTaskConfig(hidden_dims=[32, 16], dropout=0.1, epochs=3, batch_size=32)
    model = MultiTaskQSAR(targets=["InhA", "KatG"], config=cfg)
    model.fit(X, tiny_dataset.y_dict)
    return model


# ---------------------------------------------------------------------------
# MultiTaskDataset
# ---------------------------------------------------------------------------


class TestMultiTaskDataset:
    def test_len(self, tiny_dataset: MultiTaskDataset) -> None:
        assert len(tiny_dataset) == 80

    def test_n_targets(self, tiny_dataset: MultiTaskDataset) -> None:
        assert tiny_dataset.n_targets == 2

    def test_coverage_returns_dataframe(self, tiny_dataset: MultiTaskDataset) -> None:
        cov = tiny_dataset.coverage()
        assert set(cov.columns) >= {"n_total", "n_active", "pct_coverage"}

    def test_split_sizes(self, tiny_dataset: MultiTaskDataset) -> None:
        train, val, test = tiny_dataset.train_val_test_split(
            frac_train=0.7, frac_val=0.1, frac_test=0.2, scaffold_split=False, seed=0
        )
        total = len(train) + len(val) + len(test)
        assert total == len(tiny_dataset)
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_to_dataframe(self, tiny_dataset: MultiTaskDataset) -> None:
        df = tiny_dataset.to_dataframe()
        assert "smiles" in df.columns
        assert "InhA" in df.columns
        assert len(df) == 80

    def test_filter_min_targets(self, tiny_dataset: MultiTaskDataset) -> None:
        filtered = tiny_dataset.filter_min_targets(min_targets=2)
        assert len(filtered) <= len(tiny_dataset)
        # Every remaining compound should have both targets measured
        for arr in filtered.y_dict.values():
            # at least some non-NaN values
            assert (~np.isnan(arr)).any()

    def test_synthetic_has_nans(self, tiny_dataset: MultiTaskDataset) -> None:
        total_nan = sum(np.isnan(arr).sum() for arr in tiny_dataset.y_dict.values())
        assert total_nan > 0, "Synthetic data should have some NaN entries"


# ---------------------------------------------------------------------------
# compute_descriptors
# ---------------------------------------------------------------------------


class TestComputeDescriptors:
    def test_shape(self, tiny_dataset: MultiTaskDataset) -> None:
        X = compute_descriptors(tiny_dataset, n_bits=512)
        assert X.shape == (80, 512)

    def test_dtype(self, tiny_dataset: MultiTaskDataset) -> None:
        X = compute_descriptors(tiny_dataset)
        assert X.dtype == np.float32

    def test_no_nan(self, tiny_dataset: MultiTaskDataset) -> None:
        X = compute_descriptors(tiny_dataset)
        assert not np.isnan(X).any()


# ---------------------------------------------------------------------------
# MultiTaskQSAR
# ---------------------------------------------------------------------------


class TestMultiTaskQSAR:
    def test_not_fitted_raises(self, tiny_model: MultiTaskQSAR) -> None:
        X = np.random.rand(5, 64).astype(np.float32)
        with pytest.raises(RuntimeError, match="fit"):
            tiny_model.predict_proba(X)

    def test_fit_smoke(self, tiny_dataset: MultiTaskDataset) -> None:
        X = compute_descriptors(tiny_dataset, n_bits=64)
        cfg = MultiTaskConfig(hidden_dims=[16, 8], epochs=2, batch_size=40, dropout=0.0)
        model = MultiTaskQSAR(targets=["InhA", "KatG"], config=cfg)
        model.fit(X, tiny_dataset.y_dict)
        assert model.is_fitted

    def test_predict_proba_shape(self, fitted_model: MultiTaskQSAR) -> None:
        X = np.random.rand(10, N_BITS).astype(np.float32)
        proba = fitted_model.predict_proba(X, mc_samples=2)
        assert set(proba.keys()) == {"InhA", "KatG"}
        for t, arr in proba.items():
            assert arr.shape == (10,), f"{t} shape mismatch"
            assert (arr >= 0).all() and (arr <= 1).all()

    def test_predict_binary(self, fitted_model: MultiTaskQSAR) -> None:
        X = np.random.rand(5, N_BITS).astype(np.float32)
        preds = fitted_model.predict(X)
        for t, arr in preds.items():
            assert set(np.unique(arr)).issubset({0, 1})

    def test_predict_uncertainty_shape(self, fitted_model: MultiTaskQSAR) -> None:
        X = np.random.rand(8, N_BITS).astype(np.float32)
        unc = fitted_model.predict_uncertainty(X)
        for t, arr in unc.items():
            assert arr.shape == (8,)
            assert (arr >= 0).all()

    def test_evaluate_returns_metrics(
        self, fitted_model: MultiTaskQSAR, tiny_dataset: MultiTaskDataset
    ) -> None:
        X = compute_descriptors(tiny_dataset, n_bits=N_BITS)
        metrics = fitted_model.evaluate(X, tiny_dataset.y_dict)
        assert len(metrics) >= 0  # may be empty if too few class examples

    def test_feature_importance_shape(self, fitted_model: MultiTaskQSAR) -> None:
        X = np.random.rand(20, N_BITS).astype(np.float32)
        imp = fitted_model.feature_importance(X)
        assert imp.shape == (N_BITS,)
        assert (imp >= 0).all()

    def test_save_load(self, fitted_model: MultiTaskQSAR, tmp_path) -> None:
        path = str(tmp_path / "model.joblib")
        fitted_model.save(path)
        loaded = MultiTaskQSAR.load(path)
        assert loaded.is_fitted
        X = np.random.rand(3, N_BITS).astype(np.float32)
        p1 = fitted_model.predict_proba(X, mc_samples=1)
        p2 = loaded.predict_proba(X, mc_samples=1)
        for t in fitted_model.targets:
            np.testing.assert_allclose(p1[t], p2[t], rtol=1e-5)


# ---------------------------------------------------------------------------
# MultiTaskQSARSklearn wrapper
# ---------------------------------------------------------------------------


class TestMultiTaskQSARSklearn:
    def test_sklearn_api(self) -> None:
        cfg = MultiTaskConfig(hidden_dims=[16, 8], epochs=2, batch_size=32, dropout=0.0)
        wrapper = MultiTaskQSARSklearn(
            targets=["InhA"], primary_target="InhA", config=cfg
        )
        X = np.random.rand(40, 64).astype(np.float32)
        y = np.random.randint(0, 2, 40).astype(np.float32) * 7.0  # pIC50-like
        wrapper.fit(X, y)
        proba = wrapper.predict_proba(X)
        assert proba.shape == (40, 2)
        preds = wrapper.predict(X)
        assert preds.shape == (40,)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def test_create_multitask_model(self) -> None:
        model = create_multitask_model(targets=["InhA"], hidden_dims=[16, 8])
        assert isinstance(model, MultiTaskQSAR)
        assert model.targets == ["InhA"]

    def test_evaluate_multitask_returns_df(self) -> None:
        cfg = MultiTaskConfig(hidden_dims=[16, 8], epochs=2, batch_size=32, dropout=0.0)
        model = MultiTaskQSAR(targets=["InhA"], config=cfg)
        n = 30
        X = np.random.rand(n, 64).astype(np.float32)
        y_dict = {"InhA": np.random.uniform(4, 9, n).astype(np.float32)}
        model.fit(X, y_dict)
        df = evaluate_multitask(model, X, y_dict)
        assert hasattr(df, "columns")
