"""Training pipeline for GNN models.

This module provides a comprehensive training framework including:
- Training loop with validation
- Early stopping
- Learning rate scheduling
- Checkpointing
- Metrics logging

Example:
    >>> trainer = GNNTrainer(model, task='classification')
    >>> history = trainer.fit(train_loader, val_loader, epochs=100)
    >>> metrics = trainer.evaluate(test_loader)
"""

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

try:
    from torch_geometric.loader import DataLoader
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


class EarlyStopping:
    """Early stopping callback to stop training when validation loss stops improving.
    
    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as improvement.
        mode: 'min' for loss, 'max' for metrics like accuracy.
        restore_best: Whether to restore best weights on stop.
        
    Example:
        >>> early_stop = EarlyStopping(patience=10)
        >>> for epoch in range(100):
        ...     if early_stop(val_loss):
        ...         break
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
    
    def __call__(self, score: float, model: nn.Module = None) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score.
            model: Model to save weights from.
            
        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            if model is not None and self.restore_best:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            if model is not None and self.restore_best:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def restore(self, model: nn.Module) -> None:
        """Restore best weights to model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info(f"Restored best weights (score: {self.best_score:.4f})")


class GNNTrainer:
    """Training pipeline for GNN models.
    
    Handles the complete training workflow including:
    - Forward/backward pass
    - Optimizer and scheduler management
    - Validation and metrics computation
    - Checkpointing and early stopping
    - Training history logging
    
    Args:
        model: GNN model to train.
        task: 'classification' or 'regression'.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization weight.
        scheduler: Learning rate scheduler type ('plateau', 'cosine', None).
        device: Device to train on ('cuda', 'cpu', or None for auto).
        
    Example:
        >>> model = GCNModel(node_dim=78, hidden_dim=128)
        >>> trainer = GNNTrainer(model, task='classification')
        >>> history = trainer.fit(train_loader, val_loader, epochs=100)
    """
    
    def __init__(
        self,
        model: nn.Module,
        task: str = 'classification',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler: Optional[str] = 'plateau',
        device: Optional[str] = None,
    ):
        self.task = task
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Device setup
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Model
        self.model = model.to(self.device)
        
        # Loss function
        if task == 'classification':
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = None
        if scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        elif scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=100
            )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': [],
        }
        
        logger.info(f"GNNTrainer initialized on {self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            
        Returns:
            Tuple of (average loss, metrics dict).
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            out = self.model(batch)
            loss = self.criterion(out, batch.y.float())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            all_preds.extend(out.detach().cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader.dataset)
        metrics = self._compute_metrics(all_targets, all_preds)
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader.
            
        Returns:
            Tuple of (average loss, metrics dict).
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in val_loader:
            batch = batch.to(self.device)
            
            out = self.model(batch)
            loss = self.criterion(out, batch.y.float())
            
            total_loss += loss.item() * batch.num_graphs
            all_preds.extend(out.cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader.dataset)
        metrics = self._compute_metrics(all_targets, all_preds)
        
        return avg_loss, metrics
    
    def _compute_metrics(self, targets: List, predictions: List) -> Dict:
        """Compute evaluation metrics.
        
        Args:
            targets: Ground truth values.
            predictions: Model predictions.
            
        Returns:
            Dictionary of metrics.
        """
        targets = np.array(targets)
        predictions = np.array(predictions)
        
        if self.task == 'classification':
            pred_labels = (predictions > 0.5).astype(int)
            
            metrics = {
                'accuracy': accuracy_score(targets, pred_labels),
                'precision': precision_score(targets, pred_labels, zero_division=0),
                'recall': recall_score(targets, pred_labels, zero_division=0),
                'f1': f1_score(targets, pred_labels, zero_division=0),
            }
            
            try:
                metrics['roc_auc'] = roc_auc_score(targets, predictions)
            except ValueError:
                metrics['roc_auc'] = 0.5
        else:
            metrics = {
                'mse': mean_squared_error(targets, predictions),
                'rmse': np.sqrt(mean_squared_error(targets, predictions)),
                'mae': mean_absolute_error(targets, predictions),
                'r2': r2_score(targets, predictions),
            }
        
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint_dir: Optional[str] = None,
        verbose: int = 1,
    ) -> Dict:
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Maximum number of epochs.
            early_stopping: Early stopping callback.
            checkpoint_dir: Directory to save checkpoints.
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
            
        Returns:
            Training history dictionary.
        """
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Validation
            if val_loader is not None:
                val_loss, val_metrics = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_metrics'].append(val_metrics)
                
                # Scheduler step
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Checkpointing
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if checkpoint_dir:
                        self.save_checkpoint(
                            os.path.join(checkpoint_dir, 'best_model.pt'),
                            epoch, val_loss, val_metrics
                        )
                
                # Early stopping
                if early_stopping is not None:
                    if early_stopping(val_loss, self.model):
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
                
                # Logging
                if verbose >= 1:
                    metric_key = 'roc_auc' if self.task == 'classification' else 'r2'
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs} - "
                        f"Loss: {train_loss:.4f}/{val_loss:.4f} - "
                        f"{metric_key}: {train_metrics.get(metric_key, 0):.4f}/{val_metrics.get(metric_key, 0):.4f}"
                    )
            else:
                if verbose >= 1:
                    logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}")
        
        # Restore best weights
        if early_stopping is not None and early_stopping.restore_best:
            early_stopping.restore(self.model)
        
        return self.history
    
    @torch.no_grad()
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """Generate predictions.
        
        Args:
            data_loader: Data loader for prediction.
            
        Returns:
            Array of predictions.
        """
        self.model.eval()
        predictions = []
        
        for batch in data_loader:
            batch = batch.to(self.device)
            out = self.model(batch)
            predictions.extend(out.cpu().numpy())
        
        return np.array(predictions)
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate the model on test data.
        
        Args:
            test_loader: Test data loader.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        test_loss, test_metrics = self.validate(test_loader)
        test_metrics['loss'] = test_loss
        
        logger.info(f"Test evaluation: {test_metrics}")
        return test_metrics
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_loss: float,
        val_metrics: Dict,
    ) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint.
            epoch: Current epoch.
            val_loss: Validation loss.
            val_metrics: Validation metrics.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'history': self.history,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint.
            
        Returns:
            Checkpoint dictionary.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
    
    def save_model(self, path: str) -> None:
        """Save model weights only.
        
        Args:
            path: Path to save model.
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved: {path}")
    
    def load_model(self, path: str) -> None:
        """Load model weights.
        
        Args:
            path: Path to model weights.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded: {path}")


def train_gnn_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    model_type: str = 'gcn',
    node_dim: int = 78,
    edge_dim: int = 0,
    hidden_dim: int = 128,
    num_layers: int = 3,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    patience: int = 15,
    task: str = 'classification',
    save_dir: str = 'models/gnn',
) -> Tuple[nn.Module, Dict, Dict]:
    """Complete GNN training pipeline.
    
    Args:
        train_loader: Training data loader.
        val_loader: Validation data loader.
        test_loader: Test data loader.
        model_type: GNN architecture type.
        node_dim: Node feature dimension.
        edge_dim: Edge feature dimension.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of GNN layers.
        epochs: Maximum training epochs.
        learning_rate: Initial learning rate.
        patience: Early stopping patience.
        task: 'classification' or 'regression'.
        save_dir: Directory to save model.
        
    Returns:
        Tuple of (trained model, training history, test metrics).
    """
    from src.gnn.models import create_model
    
    # Create model
    model = create_model(
        model_type=model_type,
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        task=task,
    )
    
    # Create trainer
    trainer = GNNTrainer(
        model=model,
        task=task,
        learning_rate=learning_rate,
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, mode='min')
    
    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping=early_stopping,
        checkpoint_dir=save_dir,
    )
    
    # Evaluate
    test_metrics = trainer.evaluate(test_loader)
    
    # Save final model
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(os.path.join(save_dir, f'{model_type}_final.pt'))
    
    # Save metrics
    results = {
        'model_type': model_type,
        'test_metrics': test_metrics,
        'best_val_loss': min(history['val_loss']),
        'epochs_trained': len(history['train_loss']),
    }
    
    with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, history, test_metrics
