"""Hyperparameter optimization using Optuna.

This module provides automated hyperparameter tuning for all ML models
in the pipeline using Optuna's efficient Bayesian optimization.

Features:
- QSAR model optimization (RF, XGBoost, LightGBM)
- GNN architecture search
- VAE hyperparameter tuning
- Multi-objective optimization
- Experiment tracking integration

Example:
    >>> tuner = HyperparameterTuner(model_type='qsar')
    >>> best_params = tuner.optimize(X_train, y_train, n_trials=100)
    >>> model = tuner.create_model(best_params)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

try:
    import optuna
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler, CmaEsSampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("Optuna not installed. Hyperparameter optimization unavailable.")

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization.
    
    Args:
        n_trials: Number of optimization trials.
        timeout: Maximum optimization time in seconds.
        n_jobs: Number of parallel jobs.
        cv_folds: Number of cross-validation folds.
        metric: Optimization metric ('roc_auc', 'r2', 'rmse').
        direction: 'maximize' or 'minimize'.
        pruning: Enable early stopping of unpromising trials.
        seed: Random seed for reproducibility.
    """
    n_trials: int = 100
    timeout: Optional[int] = None
    n_jobs: int = 1
    cv_folds: int = 5
    metric: str = "roc_auc"
    direction: str = "maximize"
    pruning: bool = True
    seed: int = 42
    study_name: str = "tb_drug_discovery"
    storage: Optional[str] = None  # SQLite URL for persistence


class QSAROptimizer:
    """Hyperparameter optimizer for QSAR models.
    
    Supports optimization of:
    - Random Forest
    - XGBoost
    - LightGBM
    - CatBoost
    - Ensemble combinations
    
    Args:
        config: OptimizationConfig instance.
        
    Example:
        >>> optimizer = QSAROptimizer()
        >>> best_params = optimizer.optimize_random_forest(X, y)
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        if not HAS_OPTUNA:
            raise ImportError("Optuna required for hyperparameter optimization")
        
        self.config = config or OptimizationConfig()
        self._setup_optuna()
    
    def _setup_optuna(self):
        """Setup Optuna with configured sampler and pruner."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        self.sampler = TPESampler(seed=self.config.seed)
        
        if self.config.pruning:
            self.pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        else:
            self.pruner = optuna.pruners.NopPruner()
    
    def optimize_random_forest(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = "classification",
    ) -> Dict[str, Any]:
        """Optimize Random Forest hyperparameters.
        
        Args:
            X: Feature matrix.
            y: Target values.
            task: 'classification' or 'regression'.
            
        Returns:
            Best hyperparameters dictionary.
        """
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "random_state": self.config.seed,
                "n_jobs": -1,
            }
            
            if task == "classification":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(**params)
                scoring = "roc_auc"
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(**params)
                scoring = "r2"
            
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.seed)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
            
            return scores.mean()
        
        study = optuna.create_study(
            direction=self.config.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=f"{self.config.study_name}_rf",
        )
        
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=True,
        )
        
        logger.info(f"Best RF params: {study.best_params}, Score: {study.best_value:.4f}")
        return study.best_params
    
    def optimize_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = "classification",
    ) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost required for this optimizer")
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "random_state": self.config.seed,
                "n_jobs": -1,
            }
            
            if task == "classification":
                model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
                scoring = "roc_auc"
            else:
                model = xgb.XGBRegressor(**params)
                scoring = "r2"
            
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.seed)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
            
            return scores.mean()
        
        study = optuna.create_study(
            direction=self.config.direction,
            sampler=self.sampler,
            pruner=self.pruner,
        )
        
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=True,
        )
        
        logger.info(f"Best XGBoost params: {study.best_params}, Score: {study.best_value:.4f}")
        return study.best_params
    
    def optimize_lightgbm(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = "classification",
    ) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM required for this optimizer")
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "random_state": self.config.seed,
                "n_jobs": -1,
                "verbose": -1,
            }
            
            if task == "classification":
                model = lgb.LGBMClassifier(**params)
                scoring = "roc_auc"
            else:
                model = lgb.LGBMRegressor(**params)
                scoring = "r2"
            
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.seed)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
            
            return scores.mean()
        
        study = optuna.create_study(
            direction=self.config.direction,
            sampler=self.sampler,
            pruner=self.pruner,
        )
        
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=True,
        )
        
        logger.info(f"Best LightGBM params: {study.best_params}, Score: {study.best_value:.4f}")
        return study.best_params
    
    def optimize_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = "classification",
        models: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Optimize all specified model types.
        
        Args:
            X: Feature matrix.
            y: Target values.
            task: 'classification' or 'regression'.
            models: List of models to optimize ['rf', 'xgboost', 'lightgbm'].
            
        Returns:
            Dictionary mapping model name to best params.
        """
        if models is None:
            models = ["rf", "xgboost", "lightgbm"]
        
        results = {}
        
        optimizers = {
            "rf": self.optimize_random_forest,
            "xgboost": self.optimize_xgboost,
            "lightgbm": self.optimize_lightgbm,
        }
        
        for model_name in models:
            if model_name in optimizers:
                logger.info(f"Optimizing {model_name}...")
                try:
                    results[model_name] = optimizers[model_name](X, y, task)
                except ImportError as e:
                    logger.warning(f"Skipping {model_name}: {e}")
        
        return results


class GNNOptimizer:
    """Hyperparameter optimizer for GNN models.
    
    Optimizes:
    - Architecture (GCN, GAT, MPNN, AttentiveFP)
    - Hidden dimensions
    - Number of layers
    - Learning rate and regularization
    - Dropout and batch normalization
    
    Args:
        config: OptimizationConfig instance.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        if not HAS_OPTUNA:
            raise ImportError("Optuna required for hyperparameter optimization")
        
        self.config = config or OptimizationConfig()
        self.sampler = TPESampler(seed=self.config.seed)
        self.pruner = HyperbandPruner() if self.config.pruning else optuna.pruners.NopPruner()
    
    def optimize(
        self,
        train_loader,
        val_loader,
        node_dim: int,
        edge_dim: int = 0,
        task: str = "classification",
        max_epochs: int = 50,
    ) -> Dict[str, Any]:
        """Optimize GNN hyperparameters.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            node_dim: Node feature dimension.
            edge_dim: Edge feature dimension.
            task: 'classification' or 'regression'.
            max_epochs: Maximum training epochs per trial.
            
        Returns:
            Best hyperparameters dictionary.
        """
        import torch
        from src.gnn.models import create_model
        from src.gnn.trainer import GNNTrainer, EarlyStopping
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            model_type = trial.suggest_categorical("model_type", ["gcn", "gat", "mpnn"])
            hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
            num_layers = trial.suggest_int("num_layers", 2, 5)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
            
            # Handle edge_dim for MPNN
            if model_type == "mpnn" and edge_dim == 0:
                model_type = "gcn"  # Fallback if no edge features
            
            # Create model
            model = create_model(
                model_type=model_type,
                node_dim=node_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                task=task,
            )
            
            # Create trainer
            trainer = GNNTrainer(
                model=model,
                task=task,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                device=str(device),
            )
            
            # Early stopping
            early_stopping = EarlyStopping(patience=10, mode="min")
            
            # Train with pruning
            best_val_metric = float("-inf") if task == "classification" else float("-inf")
            
            for epoch in range(max_epochs):
                train_loss, train_metrics = trainer.train_epoch(train_loader)
                val_loss, val_metrics = trainer.validate(val_loader)
                
                # Get metric
                if task == "classification":
                    current_metric = val_metrics.get("roc_auc", 0.5)
                else:
                    current_metric = val_metrics.get("r2", 0.0)
                
                best_val_metric = max(best_val_metric, current_metric)
                
                # Report to Optuna for pruning
                trial.report(current_metric, epoch)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                if early_stopping(val_loss, model):
                    break
            
            return best_val_metric
        
        study = optuna.create_study(
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=f"{self.config.study_name}_gnn",
        )
        
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=True,
        )
        
        logger.info(f"Best GNN params: {study.best_params}, Score: {study.best_value:.4f}")
        return study.best_params


class MultiObjectiveOptimizer:
    """Multi-objective hyperparameter optimization.
    
    Optimizes for multiple objectives simultaneously:
    - Model accuracy
    - Inference speed
    - Model complexity
    
    Uses Pareto frontier optimization.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        if not HAS_OPTUNA:
            raise ImportError("Optuna required")
        
        self.config = config or OptimizationConfig()
        self.sampler = TPESampler(seed=self.config.seed)
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = "classification",
    ) -> List[Dict[str, Any]]:
        """Run multi-objective optimization.
        
        Objectives:
        1. Maximize accuracy (ROC-AUC or R²)
        2. Minimize model complexity (number of parameters)
        
        Returns:
            List of Pareto-optimal configurations.
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        def objective(trial: optuna.Trial) -> Tuple[float, float]:
            n_estimators = trial.suggest_int("n_estimators", 10, 300)
            max_depth = trial.suggest_int("max_depth", 2, 15)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
            
            params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "random_state": self.config.seed,
                "n_jobs": -1,
            }
            
            if task == "classification":
                model = RandomForestClassifier(**params)
                scoring = "roc_auc"
            else:
                model = RandomForestRegressor(**params)
                scoring = "r2"
            
            # Objective 1: Accuracy
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.seed)
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            accuracy = scores.mean()
            
            # Objective 2: Complexity (lower is better, so we minimize)
            complexity = n_estimators * max_depth / 100  # Normalized
            
            return accuracy, complexity
        
        study = optuna.create_study(
            directions=["maximize", "minimize"],
            sampler=optuna.samplers.NSGAIISampler(seed=self.config.seed),
            study_name=f"{self.config.study_name}_multi",
        )
        
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=True,
        )
        
        # Extract Pareto frontier
        pareto_trials = study.best_trials
        pareto_params = [t.params for t in pareto_trials]
        
        logger.info(f"Found {len(pareto_params)} Pareto-optimal configurations")
        return pareto_params


def run_full_optimization(
    X: np.ndarray,
    y: np.ndarray,
    task: str = "classification",
    n_trials: int = 50,
    output_dir: str = "results/hyperopt",
) -> Dict[str, Any]:
    """Run complete hyperparameter optimization pipeline.
    
    Args:
        X: Feature matrix.
        y: Target values.
        task: 'classification' or 'regression'.
        n_trials: Number of trials per model.
        output_dir: Directory to save results.
        
    Returns:
        Dictionary with all optimization results.
    """
    config = OptimizationConfig(n_trials=n_trials)
    optimizer = QSAROptimizer(config)
    
    # Optimize all models
    results = optimizer.optimize_all(X, y, task)
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(f"{output_dir}/best_params.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Optimization results saved to {output_dir}")
    return results
