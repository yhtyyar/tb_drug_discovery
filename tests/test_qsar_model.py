"""Unit tests for QSAR model."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.qsar_model import QSARModel


class TestQSARModelRegression:
    """Tests for QSARModel regression task."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample regression data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1
        
        return X, y
    
    @pytest.fixture
    def model(self):
        """Create regression model instance."""
        return QSARModel(
            task="regression",
            n_estimators=50,  # Fewer trees for faster tests
            random_seed=42,
        )
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.task == "regression"
        assert model.is_fitted is False
        assert model.params["n_estimators"] == 50
    
    def test_fit(self, model, sample_data):
        """Test model training."""
        X, y = sample_data
        
        model.fit(X, y)
        
        assert model.is_fitted is True
        assert "r2_train" in model.training_metrics
        assert model.training_metrics["r2_train"] > 0.8  # Should fit training data well
    
    def test_predict(self, model, sample_data):
        """Test prediction."""
        X, y = sample_data
        model.fit(X, y)
        
        predictions = model.predict(X)
        
        assert predictions.shape == (len(X),)
        assert np.isfinite(predictions).all()
    
    def test_predict_before_fit(self, model, sample_data):
        """Test error when predicting before fitting."""
        X, _ = sample_data
        
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)
    
    def test_evaluate(self, model, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]
        
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        
        assert "r2" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert metrics["r2"] > 0  # Should have positive R²
    
    def test_cross_validate(self, model, sample_data):
        """Test cross-validation."""
        X, y = sample_data
        
        cv_results = model.cross_validate(X, y, n_folds=3)
        
        assert "r2_mean" in cv_results
        assert "r2_std" in cv_results
        assert cv_results["r2_std"] < 0.2  # Reasonable stability
    
    def test_feature_importance(self, model, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        feature_names = [f"desc_{i}" for i in range(X.shape[1])]
        
        model.fit(X, y, feature_names=feature_names)
        importance = model.get_feature_importance(top_n=5)
        
        assert isinstance(importance, pd.DataFrame)
        assert len(importance) == 5
        assert "feature" in importance.columns
        assert "importance" in importance.columns
    
    def test_save_load(self, model, sample_data):
        """Test model serialization."""
        X, y = sample_data
        model.fit(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pkl")
            model.save(path)
            
            loaded_model = QSARModel.load(path)
            
            assert loaded_model.is_fitted is True
            assert loaded_model.task == "regression"
            
            # Predictions should match
            original_pred = model.predict(X[:10])
            loaded_pred = loaded_model.predict(X[:10])
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)
    
    def test_save_metrics(self, model):
        """Test metrics saving."""
        metrics = {"r2": 0.85, "rmse": 0.42}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.json")
            model.save_metrics(path, metrics)
            
            with open(path) as f:
                loaded = json.load(f)
            
            assert loaded["r2"] == 0.85


class TestQSARModelClassification:
    """Tests for QSARModel classification task."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample classification data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        return X, y
    
    @pytest.fixture
    def model(self):
        """Create classification model instance."""
        return QSARModel(
            task="classification",
            n_estimators=50,
            random_seed=42,
        )
    
    def test_initialization(self, model):
        """Test model initialization."""
        assert model.task == "classification"
        assert model.is_fitted is False
    
    def test_fit(self, model, sample_data):
        """Test model training."""
        X, y = sample_data
        
        model.fit(X, y)
        
        assert model.is_fitted is True
        assert "accuracy_train" in model.training_metrics
        assert "roc_auc_train" in model.training_metrics
    
    def test_predict_proba(self, model, sample_data):
        """Test probability prediction."""
        X, y = sample_data
        model.fit(X, y)
        
        proba = model.predict_proba(X)
        
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_evaluate(self, model, sample_data):
        """Test classification evaluation."""
        X, y = sample_data
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]
        
        model.fit(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        
        assert "accuracy" in metrics
        assert "roc_auc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert metrics["roc_auc"] > 0.5  # Better than random
    
    def test_cross_validate(self, model, sample_data):
        """Test cross-validation for classification."""
        X, y = sample_data
        
        cv_results = model.cross_validate(X, y, n_folds=3)
        
        assert "roc_auc_mean" in cv_results
        assert "accuracy_mean" in cv_results


class TestQSARModelEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_task(self):
        """Test error on invalid task."""
        with pytest.raises(ValueError, match="Unknown task"):
            QSARModel(task="invalid")
    
    def test_predict_proba_regression(self):
        """Test error when calling predict_proba on regression model."""
        model = QSARModel(task="regression")
        
        with pytest.raises(ValueError, match="classification"):
            model.predict_proba(np.array([[1, 2, 3]]))
    
    def test_repr(self):
        """Test string representation."""
        model = QSARModel(task="regression", n_estimators=100)
        repr_str = repr(model)
        
        assert "regression" in repr_str
        assert "100" in repr_str
        assert "not fitted" in repr_str
    
    def test_set_params(self):
        """Test parameter setting."""
        model = QSARModel(n_estimators=50)
        model.set_params(n_estimators=100)
        
        assert model.params["n_estimators"] == 100
        assert model.is_fitted is False  # Should reset fitted state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
