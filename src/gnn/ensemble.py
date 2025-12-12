"""Ensemble models combining QSAR and GNN predictions.

This module provides ensemble methods that combine traditional
QSAR models with graph neural networks for improved predictions.

Ensemble strategies:
- Simple averaging
- Weighted averaging (learned or fixed)
- Stacking with meta-learner
- Voting for classification

Example:
    >>> ensemble = EnsembleModel(qsar_model, gnn_model, strategy='weighted')
    >>> predictions = ensemble.predict(smiles_list, X_descriptors)
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score

try:
    from torch_geometric.loader import DataLoader
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


class EnsembleModel:
    """Ensemble model combining QSAR and GNN predictions.
    
    Supports multiple ensemble strategies for combining predictions
    from traditional descriptor-based QSAR models with graph neural networks.
    
    Args:
        qsar_model: Trained QSAR model (sklearn-compatible).
        gnn_model: Trained GNN model (PyTorch).
        featurizer: MolecularGraphFeaturizer for GNN input.
        preprocessor: DataPreprocessor for QSAR input.
        strategy: Ensemble strategy ('average', 'weighted', 'stacking', 'voting').
        weights: Weights for 'weighted' strategy [qsar_weight, gnn_weight].
        task: 'classification' or 'regression'.
        device: Device for GNN inference.
        
    Example:
        >>> ensemble = EnsembleModel(qsar_model, gnn_model, featurizer, preprocessor)
        >>> ensemble.set_weights([0.4, 0.6])  # 40% QSAR, 60% GNN
        >>> predictions = ensemble.predict(smiles_list, X_descriptors)
    """
    
    def __init__(
        self,
        qsar_model=None,
        gnn_model: Optional[nn.Module] = None,
        featurizer=None,
        preprocessor=None,
        strategy: str = 'weighted',
        weights: Optional[List[float]] = None,
        task: str = 'classification',
        device: Optional[str] = None,
    ):
        self.qsar_model = qsar_model
        self.gnn_model = gnn_model
        self.featurizer = featurizer
        self.preprocessor = preprocessor
        self.strategy = strategy
        self.task = task
        
        # Device for GNN
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        if gnn_model is not None:
            self.gnn_model = gnn_model.to(self.device)
            self.gnn_model.eval()
        
        # Weights for weighted averaging
        if weights is None:
            self.weights = [0.5, 0.5]
        else:
            self.weights = weights
        
        # Meta-learner for stacking
        self.meta_learner = None
        
        logger.info(f"EnsembleModel initialized: strategy={strategy}, weights={self.weights}")
    
    def set_weights(self, weights: List[float]) -> None:
        """Set ensemble weights.
        
        Args:
            weights: [qsar_weight, gnn_weight], should sum to 1.
        """
        if len(weights) != 2:
            raise ValueError("Weights must be a list of 2 values [qsar, gnn]")
        
        total = sum(weights)
        self.weights = [w / total for w in weights]  # Normalize
        logger.info(f"Weights set: QSAR={self.weights[0]:.2f}, GNN={self.weights[1]:.2f}")
    
    def _get_qsar_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from QSAR model.
        
        Args:
            X: Feature matrix (scaled descriptors).
            
        Returns:
            QSAR predictions.
        """
        if self.qsar_model is None:
            raise ValueError("QSAR model not set")
        
        if self.task == 'classification':
            if hasattr(self.qsar_model, 'predict_proba'):
                return self.qsar_model.predict_proba(X)[:, 1]
            else:
                return self.qsar_model.predict(X)
        else:
            return self.qsar_model.predict(X)
    
    @torch.no_grad()
    def _get_gnn_predictions(self, smiles_list: List[str]) -> np.ndarray:
        """Get predictions from GNN model.
        
        Args:
            smiles_list: List of SMILES strings.
            
        Returns:
            GNN predictions.
        """
        if self.gnn_model is None or self.featurizer is None:
            raise ValueError("GNN model or featurizer not set")
        
        self.gnn_model.eval()
        
        # Convert SMILES to graphs
        graphs = self.featurizer.batch_smiles_to_graphs(smiles_list, progress=False)
        
        if not graphs:
            return np.array([])
        
        # Create DataLoader
        loader = DataLoader(graphs, batch_size=32, shuffle=False)
        
        predictions = []
        for batch in loader:
            batch = batch.to(self.device)
            out = self.gnn_model(batch)
            predictions.extend(out.cpu().numpy())
        
        return np.array(predictions)
    
    def predict(
        self,
        smiles_list: List[str],
        X: Optional[np.ndarray] = None,
        descriptor_calculator=None,
    ) -> np.ndarray:
        """Generate ensemble predictions.
        
        Args:
            smiles_list: List of SMILES strings.
            X: Pre-computed and scaled descriptor matrix.
            descriptor_calculator: Calculator to compute descriptors if X not provided.
            
        Returns:
            Ensemble predictions.
        """
        # Get QSAR predictions
        if X is None and descriptor_calculator is not None and self.preprocessor is not None:
            # Compute descriptors
            import pandas as pd
            df = pd.DataFrame({'smiles': smiles_list})
            df_desc = descriptor_calculator.calculate_from_dataframe(df, smiles_col='smiles')
            X = df_desc[descriptor_calculator.descriptor_names].values
            X = self.preprocessor.transform(X)
        
        qsar_preds = self._get_qsar_predictions(X) if X is not None else None
        
        # Get GNN predictions
        gnn_preds = self._get_gnn_predictions(smiles_list) if self.gnn_model is not None else None
        
        # Combine predictions based on strategy
        if self.strategy == 'average':
            if qsar_preds is not None and gnn_preds is not None:
                return (qsar_preds + gnn_preds) / 2
            return qsar_preds if qsar_preds is not None else gnn_preds
        
        elif self.strategy == 'weighted':
            if qsar_preds is not None and gnn_preds is not None:
                return self.weights[0] * qsar_preds + self.weights[1] * gnn_preds
            return qsar_preds if qsar_preds is not None else gnn_preds
        
        elif self.strategy == 'stacking':
            if self.meta_learner is None:
                raise ValueError("Meta-learner not trained. Call fit_stacking() first.")
            
            X_meta = np.column_stack([qsar_preds, gnn_preds])
            
            if self.task == 'classification':
                return self.meta_learner.predict_proba(X_meta)[:, 1]
            else:
                return self.meta_learner.predict(X_meta)
        
        elif self.strategy == 'voting':
            if self.task != 'classification':
                raise ValueError("Voting only supported for classification")
            
            qsar_votes = (qsar_preds > 0.5).astype(int)
            gnn_votes = (gnn_preds > 0.5).astype(int)
            
            # Majority voting
            return ((qsar_votes + gnn_votes) >= 1).astype(float)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def fit_stacking(
        self,
        smiles_list: List[str],
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Fit the stacking meta-learner.
        
        Args:
            smiles_list: Training SMILES.
            X: Training descriptor matrix (scaled).
            y: Training targets.
        """
        # Get base predictions
        qsar_preds = self._get_qsar_predictions(X)
        gnn_preds = self._get_gnn_predictions(smiles_list)
        
        # Stack predictions
        X_meta = np.column_stack([qsar_preds, gnn_preds])
        
        # Train meta-learner
        if self.task == 'classification':
            self.meta_learner = LogisticRegression(random_state=42)
        else:
            self.meta_learner = Ridge(alpha=1.0)
        
        self.meta_learner.fit(X_meta, y)
        
        logger.info("Stacking meta-learner trained")
    
    def optimize_weights(
        self,
        smiles_list: List[str],
        X: np.ndarray,
        y: np.ndarray,
        n_steps: int = 21,
    ) -> Tuple[List[float], float]:
        """Find optimal ensemble weights via grid search.
        
        Args:
            smiles_list: Validation SMILES.
            X: Validation descriptor matrix.
            y: Validation targets.
            n_steps: Number of weight steps to try.
            
        Returns:
            Tuple of (best_weights, best_score).
        """
        qsar_preds = self._get_qsar_predictions(X)
        gnn_preds = self._get_gnn_predictions(smiles_list)
        
        best_score = -float('inf')
        best_weights = [0.5, 0.5]
        
        for w in np.linspace(0, 1, n_steps):
            ensemble_preds = w * qsar_preds + (1 - w) * gnn_preds
            
            if self.task == 'classification':
                try:
                    score = roc_auc_score(y, ensemble_preds)
                except ValueError:
                    score = 0.5
            else:
                score = r2_score(y, ensemble_preds)
            
            if score > best_score:
                best_score = score
                best_weights = [w, 1 - w]
        
        self.weights = best_weights
        logger.info(f"Optimized weights: QSAR={best_weights[0]:.2f}, GNN={best_weights[1]:.2f}, score={best_score:.4f}")
        
        return best_weights, best_score
    
    def evaluate(
        self,
        smiles_list: List[str],
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict:
        """Evaluate ensemble performance.
        
        Args:
            smiles_list: Test SMILES.
            X: Test descriptor matrix.
            y: Test targets.
            
        Returns:
            Dictionary of metrics.
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_squared_error, mean_absolute_error
        )
        
        # Get individual predictions
        qsar_preds = self._get_qsar_predictions(X)
        gnn_preds = self._get_gnn_predictions(smiles_list)
        ensemble_preds = self.predict(smiles_list, X)
        
        metrics = {}
        
        if self.task == 'classification':
            # QSAR metrics
            try:
                metrics['qsar_roc_auc'] = roc_auc_score(y, qsar_preds)
            except:
                metrics['qsar_roc_auc'] = 0.5
            
            # GNN metrics
            try:
                metrics['gnn_roc_auc'] = roc_auc_score(y, gnn_preds)
            except:
                metrics['gnn_roc_auc'] = 0.5
            
            # Ensemble metrics
            try:
                metrics['ensemble_roc_auc'] = roc_auc_score(y, ensemble_preds)
            except:
                metrics['ensemble_roc_auc'] = 0.5
            
            # Classification metrics
            ensemble_labels = (ensemble_preds > 0.5).astype(int)
            metrics['ensemble_accuracy'] = accuracy_score(y, ensemble_labels)
            metrics['ensemble_precision'] = precision_score(y, ensemble_labels, zero_division=0)
            metrics['ensemble_recall'] = recall_score(y, ensemble_labels, zero_division=0)
            metrics['ensemble_f1'] = f1_score(y, ensemble_labels, zero_division=0)
            
            # Improvement
            metrics['improvement_over_qsar'] = metrics['ensemble_roc_auc'] - metrics['qsar_roc_auc']
            metrics['improvement_over_gnn'] = metrics['ensemble_roc_auc'] - metrics['gnn_roc_auc']
        
        else:
            metrics['qsar_r2'] = r2_score(y, qsar_preds)
            metrics['gnn_r2'] = r2_score(y, gnn_preds)
            metrics['ensemble_r2'] = r2_score(y, ensemble_preds)
            metrics['ensemble_rmse'] = np.sqrt(mean_squared_error(y, ensemble_preds))
            metrics['ensemble_mae'] = mean_absolute_error(y, ensemble_preds)
        
        metrics['weights'] = self.weights
        
        logger.info(f"Ensemble evaluation: {metrics}")
        return metrics
    
    def save(self, path: str) -> None:
        """Save ensemble model.
        
        Args:
            path: Directory to save model components.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config = {
            'strategy': self.strategy,
            'weights': self.weights,
            'task': self.task,
        }
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save QSAR model
        if self.qsar_model is not None:
            with open(path / 'qsar_model.pkl', 'wb') as f:
                pickle.dump(self.qsar_model, f)
        
        # Save GNN model
        if self.gnn_model is not None:
            torch.save(self.gnn_model.state_dict(), path / 'gnn_model.pt')
        
        # Save meta-learner
        if self.meta_learner is not None:
            with open(path / 'meta_learner.pkl', 'wb') as f:
                pickle.dump(self.meta_learner, f)
        
        logger.info(f"Ensemble saved to: {path}")
    
    @classmethod
    def load(
        cls,
        path: str,
        gnn_model_class=None,
        gnn_model_kwargs: Dict = None,
        featurizer=None,
        preprocessor=None,
    ) -> 'EnsembleModel':
        """Load ensemble model.
        
        Args:
            path: Directory with saved model.
            gnn_model_class: GNN model class to instantiate.
            gnn_model_kwargs: Arguments for GNN model.
            featurizer: MolecularGraphFeaturizer instance.
            preprocessor: DataPreprocessor instance.
            
        Returns:
            Loaded EnsembleModel.
        """
        path = Path(path)
        
        # Load config
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)
        
        # Load QSAR model
        qsar_model = None
        if (path / 'qsar_model.pkl').exists():
            with open(path / 'qsar_model.pkl', 'rb') as f:
                qsar_model = pickle.load(f)
        
        # Load GNN model
        gnn_model = None
        if (path / 'gnn_model.pt').exists() and gnn_model_class is not None:
            gnn_model = gnn_model_class(**(gnn_model_kwargs or {}))
            gnn_model.load_state_dict(torch.load(path / 'gnn_model.pt'))
        
        # Create ensemble
        ensemble = cls(
            qsar_model=qsar_model,
            gnn_model=gnn_model,
            featurizer=featurizer,
            preprocessor=preprocessor,
            strategy=config['strategy'],
            weights=config['weights'],
            task=config['task'],
        )
        
        # Load meta-learner
        if (path / 'meta_learner.pkl').exists():
            with open(path / 'meta_learner.pkl', 'rb') as f:
                ensemble.meta_learner = pickle.load(f)
        
        logger.info(f"Ensemble loaded from: {path}")
        return ensemble
