"""Tests for hyperparameter optimization module.

Tests cover:
- OptimizationConfig
- QSAROptimizer
- GNNOptimizer (mocked)
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression

# Check if optuna is available
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

if HAS_OPTUNA:
    from src.models.hyperopt import (
        OptimizationConfig,
        QSAROptimizer,
        run_full_optimization,
    )


@pytest.fixture
def classification_data():
    """Generate classification dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=10,
        random_state=42,
    )
    return X, y


@pytest.fixture
def regression_data():
    """Generate regression dataset."""
    X, y = make_regression(
        n_samples=100,
        n_features=20,
        n_informative=10,
        random_state=42,
    )
    return X, y


@pytest.mark.skipif(not HAS_OPTUNA, reason="Optuna not installed")
class TestOptimizationConfig:
    """Tests for OptimizationConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = OptimizationConfig()
        
        assert config.n_trials == 100
        assert config.cv_folds == 5
        assert config.metric == "roc_auc"
        assert config.direction == "maximize"
        assert config.pruning is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = OptimizationConfig(
            n_trials=50,
            cv_folds=3,
            metric="r2",
            direction="maximize",
        )
        
        assert config.n_trials == 50
        assert config.cv_folds == 3
        assert config.metric == "r2"


@pytest.mark.skipif(not HAS_OPTUNA, reason="Optuna not installed")
class TestQSAROptimizer:
    """Tests for QSAROptimizer."""
    
    def test_init(self):
        """Test optimizer initialization."""
        config = OptimizationConfig(n_trials=5)
        optimizer = QSAROptimizer(config)
        
        assert optimizer.config.n_trials == 5
        assert optimizer.sampler is not None
    
    def test_optimize_random_forest_classification(self, classification_data):
        """Test RF optimization for classification."""
        X, y = classification_data
        
        config = OptimizationConfig(n_trials=3, cv_folds=2)
        optimizer = QSAROptimizer(config)
        
        best_params = optimizer.optimize_random_forest(X, y, task="classification")
        
        assert "n_estimators" in best_params
        assert "max_depth" in best_params
        assert best_params["n_estimators"] >= 50
    
    def test_optimize_random_forest_regression(self, regression_data):
        """Test RF optimization for regression."""
        X, y = regression_data
        
        config = OptimizationConfig(n_trials=3, cv_folds=2)
        optimizer = QSAROptimizer(config)
        
        best_params = optimizer.optimize_random_forest(X, y, task="regression")
        
        assert "n_estimators" in best_params
        assert "max_depth" in best_params
    
    def test_optimize_xgboost(self, classification_data):
        """Test XGBoost optimization."""
        try:
            import xgboost
        except ImportError:
            pytest.skip("XGBoost not installed")
        
        X, y = classification_data
        
        config = OptimizationConfig(n_trials=3, cv_folds=2)
        optimizer = QSAROptimizer(config)
        
        best_params = optimizer.optimize_xgboost(X, y, task="classification")
        
        assert "n_estimators" in best_params
        assert "learning_rate" in best_params
    
    def test_optimize_lightgbm(self, classification_data):
        """Test LightGBM optimization."""
        try:
            import lightgbm
        except ImportError:
            pytest.skip("LightGBM not installed")
        
        X, y = classification_data
        
        config = OptimizationConfig(n_trials=3, cv_folds=2)
        optimizer = QSAROptimizer(config)
        
        best_params = optimizer.optimize_lightgbm(X, y, task="classification")
        
        assert "n_estimators" in best_params
        assert "num_leaves" in best_params
    
    def test_optimize_all(self, classification_data):
        """Test optimizing all available models."""
        X, y = classification_data
        
        config = OptimizationConfig(n_trials=2, cv_folds=2)
        optimizer = QSAROptimizer(config)
        
        # Only optimize RF to make test fast
        results = optimizer.optimize_all(X, y, task="classification", models=["rf"])
        
        assert "rf" in results
        assert "n_estimators" in results["rf"]


@pytest.mark.skipif(not HAS_OPTUNA, reason="Optuna not installed")
class TestRunFullOptimization:
    """Tests for full optimization pipeline."""
    
    def test_run_full_optimization(self, classification_data, tmp_path):
        """Test complete optimization run."""
        X, y = classification_data
        
        results = run_full_optimization(
            X, y,
            task="classification",
            n_trials=2,
            output_dir=str(tmp_path / "hyperopt"),
        )
        
        assert len(results) > 0
        assert (tmp_path / "hyperopt" / "best_params.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
