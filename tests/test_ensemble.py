"""Tests for ensemble models module.

Tests cover:
- VotingEnsemble
- StackingEnsemble
- BlendingEnsemble
- Factory functions
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from src.models.ensemble import (
    VotingEnsemble,
    StackingEnsemble,
    BlendingEnsemble,
    create_ensemble,
    train_best_ensemble,
)


@pytest.fixture
def classification_data():
    """Generate classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


@pytest.fixture
def regression_data():
    """Generate regression dataset."""
    X, y = make_regression(
        n_samples=200,
        n_features=20,
        n_informative=10,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


class TestVotingEnsemble:
    """Tests for VotingEnsemble."""
    
    def test_init_default_models(self):
        """Test initialization with default models."""
        ensemble = VotingEnsemble(task="classification")
        
        assert len(ensemble.base_models) > 0
        assert len(ensemble.model_names) == len(ensemble.base_models)
        assert not ensemble.is_fitted
    
    def test_fit_classification(self, classification_data):
        """Test fitting on classification data."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = VotingEnsemble(task="classification")
        ensemble.fit(X_train, y_train)
        
        assert ensemble.is_fitted
    
    def test_predict_classification(self, classification_data):
        """Test prediction on classification data."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = VotingEnsemble(task="classification")
        ensemble.fit(X_train, y_train)
        
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba(self, classification_data):
        """Test probability prediction."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = VotingEnsemble(task="classification", voting="soft")
        ensemble.fit(X_train, y_train)
        
        proba = ensemble.predict_proba(X_test)
        
        assert proba.shape == (len(X_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_evaluate(self, classification_data):
        """Test evaluation metrics."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = VotingEnsemble(task="classification")
        ensemble.fit(X_train, y_train)
        
        metrics = ensemble.evaluate(X_test, y_test)
        
        assert "accuracy" in metrics
        assert "roc_auc" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1
    
    def test_regression(self, regression_data):
        """Test regression task."""
        X_train, X_test, y_train, y_test = regression_data
        
        ensemble = VotingEnsemble(task="regression")
        ensemble.fit(X_train, y_train)
        
        predictions = ensemble.predict(X_test)
        metrics = ensemble.evaluate(X_test, y_test)
        
        assert len(predictions) == len(X_test)
        assert "r2" in metrics
        assert "rmse" in metrics
    
    def test_get_model_scores(self, classification_data):
        """Test individual model scores."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = VotingEnsemble(task="classification")
        ensemble.fit(X_train, y_train)
        
        scores = ensemble.get_model_scores(X_test, y_test)
        
        assert len(scores) == len(ensemble.model_names)
        for name in ensemble.model_names:
            assert name in scores
    
    def test_weighted_voting(self, classification_data):
        """Test weighted voting."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = VotingEnsemble(
            task="classification",
            weights=[1.0, 2.0, 1.5, 0.5],  # Custom weights
        )
        ensemble.fit(X_train, y_train)
        
        predictions = ensemble.predict(X_test)
        assert len(predictions) == len(X_test)


class TestStackingEnsemble:
    """Tests for StackingEnsemble."""
    
    def test_init(self):
        """Test initialization."""
        ensemble = StackingEnsemble(task="classification")
        
        assert len(ensemble.base_models) > 0
        assert ensemble.meta_model is not None
        assert not ensemble.is_fitted
    
    def test_fit_classification(self, classification_data):
        """Test fitting on classification data."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = StackingEnsemble(task="classification", cv_folds=3)
        ensemble.fit(X_train, y_train)
        
        assert ensemble.is_fitted
        assert len(ensemble.fitted_base_models) == len(ensemble.base_models)
    
    def test_predict(self, classification_data):
        """Test prediction."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = StackingEnsemble(task="classification", cv_folds=3)
        ensemble.fit(X_train, y_train)
        
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == len(X_test)
    
    def test_predict_proba(self, classification_data):
        """Test probability prediction."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = StackingEnsemble(task="classification", cv_folds=3)
        ensemble.fit(X_train, y_train)
        
        proba = ensemble.predict_proba(X_test)
        
        assert proba.shape[0] == len(X_test)
    
    def test_use_features(self, classification_data):
        """Test stacking with original features."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = StackingEnsemble(
            task="classification",
            use_features=True,
            cv_folds=3,
        )
        ensemble.fit(X_train, y_train)
        
        predictions = ensemble.predict(X_test)
        assert len(predictions) == len(X_test)
    
    def test_get_feature_importance(self, classification_data):
        """Test feature importance extraction."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = StackingEnsemble(task="classification", cv_folds=3)
        ensemble.fit(X_train, y_train)
        
        importance = ensemble.get_feature_importance()
        
        # Should have importance for each base model
        assert len(importance) > 0
    
    def test_regression(self, regression_data):
        """Test regression task."""
        X_train, X_test, y_train, y_test = regression_data
        
        ensemble = StackingEnsemble(task="regression", cv_folds=3)
        ensemble.fit(X_train, y_train)
        
        metrics = ensemble.evaluate(X_test, y_test)
        
        assert "r2" in metrics


class TestBlendingEnsemble:
    """Tests for BlendingEnsemble."""
    
    def test_init(self):
        """Test initialization."""
        ensemble = BlendingEnsemble(task="classification")
        
        assert len(ensemble.base_models) > 0
        assert ensemble.holdout_ratio == 0.2
    
    def test_fit(self, classification_data):
        """Test fitting."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = BlendingEnsemble(task="classification", holdout_ratio=0.3)
        ensemble.fit(X_train, y_train)
        
        assert ensemble.is_fitted
    
    def test_predict(self, classification_data):
        """Test prediction."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = BlendingEnsemble(task="classification")
        ensemble.fit(X_train, y_train)
        
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == len(X_test)
    
    def test_regression(self, regression_data):
        """Test regression."""
        X_train, X_test, y_train, y_test = regression_data
        
        ensemble = BlendingEnsemble(task="regression")
        ensemble.fit(X_train, y_train)
        
        metrics = ensemble.evaluate(X_test, y_test)
        
        assert "r2" in metrics


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_ensemble_voting(self):
        """Test creating voting ensemble."""
        ensemble = create_ensemble("voting", task="classification")
        assert isinstance(ensemble, VotingEnsemble)
    
    def test_create_ensemble_stacking(self):
        """Test creating stacking ensemble."""
        ensemble = create_ensemble("stacking", task="classification")
        assert isinstance(ensemble, StackingEnsemble)
    
    def test_create_ensemble_blending(self):
        """Test creating blending ensemble."""
        ensemble = create_ensemble("blending", task="classification")
        assert isinstance(ensemble, BlendingEnsemble)
    
    def test_create_ensemble_invalid(self):
        """Test invalid ensemble type."""
        with pytest.raises(ValueError):
            create_ensemble("invalid", task="classification")
    
    def test_train_best_ensemble(self, classification_data):
        """Test automatic best ensemble selection."""
        X_train, X_test, y_train, y_test = classification_data
        
        # Split train into train/val
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42
        )
        
        best_ensemble, metrics = train_best_ensemble(
            X_tr, y_tr, X_val, y_val, task="classification"
        )
        
        assert best_ensemble.is_fitted
        assert "roc_auc" in metrics


class TestSaveLoad:
    """Tests for model persistence."""
    
    def test_save_load_voting(self, classification_data, tmp_path):
        """Test saving and loading voting ensemble."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = VotingEnsemble(task="classification")
        ensemble.fit(X_train, y_train)
        
        save_path = str(tmp_path / "voting.pkl")
        ensemble.save(save_path)
        
        loaded = VotingEnsemble.load(save_path)
        
        # Should produce same predictions
        orig_pred = ensemble.predict(X_test)
        loaded_pred = loaded.predict(X_test)
        
        np.testing.assert_array_equal(orig_pred, loaded_pred)
    
    def test_save_load_stacking(self, classification_data, tmp_path):
        """Test saving and loading stacking ensemble."""
        X_train, X_test, y_train, y_test = classification_data
        
        ensemble = StackingEnsemble(task="classification", cv_folds=3)
        ensemble.fit(X_train, y_train)
        
        save_path = str(tmp_path / "stacking.pkl")
        ensemble.save(save_path)
        
        loaded = StackingEnsemble.load(save_path)
        
        assert loaded.is_fitted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
