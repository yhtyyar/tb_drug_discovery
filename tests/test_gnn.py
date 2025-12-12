"""Unit tests for GNN modules.

Tests cover:
- Molecular graph featurization
- GNN model architectures
- Training pipeline
- Ensemble models
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Skip all tests if PyTorch Geometric not installed
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get('CI') == 'true',
    reason="Skipping GNN tests in CI without GPU"
)

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


@pytest.fixture
def sample_smiles():
    """Sample SMILES for testing."""
    return [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CC(C)O",  # Isopropanol
        "CCN",  # Ethylamine
    ]


@pytest.fixture
def sample_targets():
    """Sample targets for testing."""
    return [0, 1, 1, 0, 1]


@pytest.mark.skipif(not HAS_TORCH_GEOMETRIC or not HAS_RDKIT, reason="Requires PyTorch Geometric and RDKit")
class TestMolecularGraphFeaturizer:
    """Tests for MolecularGraphFeaturizer."""
    
    def test_featurizer_initialization(self):
        """Test featurizer initialization."""
        from src.gnn.featurizer import MolecularGraphFeaturizer
        
        featurizer = MolecularGraphFeaturizer()
        
        assert featurizer.atom_dim > 0
        assert featurizer.bond_dim > 0
    
    def test_smiles_to_graph(self):
        """Test converting SMILES to graph."""
        from src.gnn.featurizer import MolecularGraphFeaturizer
        
        featurizer = MolecularGraphFeaturizer()
        graph = featurizer.smiles_to_graph("CCO", y=1.0)
        
        assert graph is not None
        assert isinstance(graph, Data)
        assert graph.x.shape[0] == 3  # 3 atoms (C, C, O)
        assert graph.x.shape[1] == featurizer.atom_dim
        assert graph.edge_index.shape[0] == 2
        assert hasattr(graph, 'y')
    
    def test_invalid_smiles(self):
        """Test handling of invalid SMILES."""
        from src.gnn.featurizer import MolecularGraphFeaturizer
        
        featurizer = MolecularGraphFeaturizer()
        graph = featurizer.smiles_to_graph("invalid_smiles")
        
        assert graph is None
    
    def test_batch_processing(self, sample_smiles, sample_targets):
        """Test batch conversion."""
        from src.gnn.featurizer import MolecularGraphFeaturizer
        
        featurizer = MolecularGraphFeaturizer()
        graphs = featurizer.batch_smiles_to_graphs(
            sample_smiles, sample_targets, progress=False
        )
        
        assert len(graphs) == len(sample_smiles)
        for g in graphs:
            assert isinstance(g, Data)
    
    def test_benzene_graph(self):
        """Test benzene ring graph structure."""
        from src.gnn.featurizer import MolecularGraphFeaturizer
        
        featurizer = MolecularGraphFeaturizer()
        graph = featurizer.smiles_to_graph("c1ccccc1")
        
        assert graph.num_nodes == 6  # 6 carbons
        assert graph.edge_index.shape[1] == 12  # 6 bonds * 2 directions


@pytest.mark.skipif(not HAS_TORCH_GEOMETRIC or not HAS_RDKIT, reason="Requires PyTorch Geometric and RDKit")
class TestMoleculeDataset:
    """Tests for MoleculeDataset."""
    
    def test_dataset_creation(self, sample_smiles, sample_targets):
        """Test dataset creation."""
        from src.gnn.featurizer import MoleculeDataset
        
        dataset = MoleculeDataset(sample_smiles, sample_targets)
        
        assert len(dataset) == len(sample_smiles)
        assert dataset.num_node_features > 0
    
    def test_dataset_getitem(self, sample_smiles, sample_targets):
        """Test dataset indexing."""
        from src.gnn.featurizer import MoleculeDataset
        
        dataset = MoleculeDataset(sample_smiles, sample_targets)
        
        data = dataset[0]
        assert isinstance(data, Data)
        assert hasattr(data, 'x')
        assert hasattr(data, 'edge_index')
    
    def test_dataset_statistics(self, sample_smiles, sample_targets):
        """Test dataset statistics."""
        from src.gnn.featurizer import MoleculeDataset
        
        dataset = MoleculeDataset(sample_smiles, sample_targets)
        stats = dataset.get_statistics()
        
        assert 'num_molecules' in stats
        assert 'avg_nodes' in stats
        assert stats['num_molecules'] == len(sample_smiles)


@pytest.mark.skipif(not HAS_TORCH_GEOMETRIC or not HAS_RDKIT, reason="Requires PyTorch Geometric and RDKit")
class TestGNNModels:
    """Tests for GNN model architectures."""
    
    def test_gcn_model(self):
        """Test GCN model."""
        from src.gnn.models import GCNModel
        
        model = GCNModel(
            node_dim=78,
            hidden_dim=64,
            output_dim=1,
            num_layers=2,
        )
        
        # Check model has required components
        assert hasattr(model, 'convs')
        assert hasattr(model, 'output_mlp')
        assert len(model.convs) == 2
    
    def test_gat_model(self):
        """Test GAT model."""
        from src.gnn.models import GATModel
        
        model = GATModel(
            node_dim=78,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
        )
        
        assert hasattr(model, 'convs')
        assert model.num_heads == 4
    
    def test_mpnn_model(self):
        """Test MPNN model."""
        from src.gnn.models import MPNNModel
        
        model = MPNNModel(
            node_dim=78,
            edge_dim=10,
            hidden_dim=64,
            num_layers=2,
        )
        
        assert hasattr(model, 'convs')
        assert hasattr(model, 'gru')
    
    def test_attentivefp_model(self):
        """Test AttentiveFP model."""
        from src.gnn.models import AttentiveFPModel
        
        model = AttentiveFPModel(
            node_dim=78,
            edge_dim=10,
            hidden_dim=64,
            num_layers=2,
        )
        
        assert hasattr(model, 'gat_layers')
        assert hasattr(model, 'graph_attention')
    
    def test_model_forward_pass(self, sample_smiles, sample_targets):
        """Test model forward pass."""
        from src.gnn.featurizer import MoleculeDataset
        from src.gnn.models import GCNModel
        
        dataset = MoleculeDataset(sample_smiles, sample_targets)
        loader = DataLoader(dataset, batch_size=2)
        
        model = GCNModel(
            node_dim=dataset.num_node_features,
            hidden_dim=32,
        )
        model.eval()
        
        batch = next(iter(loader))
        with torch.no_grad():
            out = model(batch)
        
        assert out.shape[0] == 2  # Batch size
        assert (out >= 0).all() and (out <= 1).all()  # Sigmoid output
    
    def test_create_model_factory(self):
        """Test model factory function."""
        from src.gnn.models import create_model
        
        for model_type in ['gcn', 'gat']:
            model = create_model(
                model_type=model_type,
                node_dim=78,
                hidden_dim=32,
            )
            assert model is not None
    
    def test_create_model_invalid(self):
        """Test factory with invalid model type."""
        from src.gnn.models import create_model
        
        with pytest.raises(ValueError):
            create_model(model_type='invalid', node_dim=78)


@pytest.mark.skipif(not HAS_TORCH_GEOMETRIC or not HAS_RDKIT, reason="Requires PyTorch Geometric and RDKit")
class TestGNNTrainer:
    """Tests for GNN training pipeline."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        from src.gnn.models import GCNModel
        from src.gnn.trainer import GNNTrainer
        
        model = GCNModel(node_dim=78, hidden_dim=32)
        trainer = GNNTrainer(model, task='classification')
        
        assert trainer.model is not None
        assert trainer.criterion is not None
        assert trainer.optimizer is not None
    
    def test_early_stopping(self):
        """Test early stopping callback."""
        from src.gnn.trainer import EarlyStopping
        
        early_stop = EarlyStopping(patience=3, mode='min')
        
        # Improving
        assert not early_stop(1.0)
        assert not early_stop(0.9)
        assert not early_stop(0.8)
        
        # Not improving
        assert not early_stop(0.9)
        assert not early_stop(0.95)
        assert early_stop(1.0)  # Should trigger after 3 non-improvements
    
    def test_training_step(self, sample_smiles, sample_targets):
        """Test single training epoch."""
        from src.gnn.featurizer import MoleculeDataset
        from src.gnn.models import GCNModel
        from src.gnn.trainer import GNNTrainer
        
        dataset = MoleculeDataset(sample_smiles, sample_targets)
        loader = DataLoader(dataset, batch_size=2)
        
        model = GCNModel(node_dim=dataset.num_node_features, hidden_dim=32)
        trainer = GNNTrainer(model, task='classification')
        
        loss, metrics = trainer.train_epoch(loader)
        
        assert isinstance(loss, float)
        assert 'accuracy' in metrics
        assert 'roc_auc' in metrics
    
    def test_validation(self, sample_smiles, sample_targets):
        """Test validation step."""
        from src.gnn.featurizer import MoleculeDataset
        from src.gnn.models import GCNModel
        from src.gnn.trainer import GNNTrainer
        
        dataset = MoleculeDataset(sample_smiles, sample_targets)
        loader = DataLoader(dataset, batch_size=2)
        
        model = GCNModel(node_dim=dataset.num_node_features, hidden_dim=32)
        trainer = GNNTrainer(model, task='classification')
        
        loss, metrics = trainer.validate(loader)
        
        assert isinstance(loss, float)
        assert 'accuracy' in metrics
    
    def test_save_load_model(self, sample_smiles, sample_targets):
        """Test model save and load."""
        from src.gnn.featurizer import MoleculeDataset
        from src.gnn.models import GCNModel
        from src.gnn.trainer import GNNTrainer
        
        dataset = MoleculeDataset(sample_smiles, sample_targets)
        model = GCNModel(node_dim=dataset.num_node_features, hidden_dim=32)
        trainer = GNNTrainer(model)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        
        try:
            trainer.save_model(path)
            assert os.path.exists(path)
            
            # Create new trainer and load
            model2 = GCNModel(node_dim=dataset.num_node_features, hidden_dim=32)
            trainer2 = GNNTrainer(model2)
            trainer2.load_model(path)
            
        finally:
            os.unlink(path)


@pytest.mark.skipif(not HAS_TORCH_GEOMETRIC or not HAS_RDKIT, reason="Requires PyTorch Geometric and RDKit")
class TestEnsembleModel:
    """Tests for ensemble model."""
    
    def test_ensemble_initialization(self):
        """Test ensemble initialization."""
        from src.gnn.ensemble import EnsembleModel
        
        ensemble = EnsembleModel(strategy='weighted')
        
        assert ensemble.weights == [0.5, 0.5]
        assert ensemble.strategy == 'weighted'
    
    def test_set_weights(self):
        """Test weight setting."""
        from src.gnn.ensemble import EnsembleModel
        
        ensemble = EnsembleModel()
        ensemble.set_weights([0.3, 0.7])
        
        assert abs(ensemble.weights[0] - 0.3) < 0.01
        assert abs(ensemble.weights[1] - 0.7) < 0.01
    
    def test_weights_normalization(self):
        """Test weight normalization."""
        from src.gnn.ensemble import EnsembleModel
        
        ensemble = EnsembleModel()
        ensemble.set_weights([1, 3])  # Should normalize to [0.25, 0.75]
        
        assert abs(sum(ensemble.weights) - 1.0) < 0.01


@pytest.mark.skipif(not HAS_TORCH_GEOMETRIC or not HAS_RDKIT, reason="Requires PyTorch Geometric and RDKit")
class TestDataLoaders:
    """Tests for data loader creation."""
    
    def test_create_data_loaders(self, sample_smiles, sample_targets):
        """Test DataLoader creation."""
        from src.gnn.featurizer import create_data_loaders
        
        train_loader, val_loader, test_loader, featurizer = create_data_loaders(
            sample_smiles, sample_targets,
            train_ratio=0.6,
            val_ratio=0.2,
            batch_size=2,
        )
        
        assert len(train_loader.dataset) > 0
        assert featurizer is not None
    
    def test_data_loader_iteration(self, sample_smiles, sample_targets):
        """Test iterating through DataLoader."""
        from src.gnn.featurizer import create_data_loaders
        
        train_loader, _, _, _ = create_data_loaders(
            sample_smiles, sample_targets,
            batch_size=2,
        )
        
        batch = next(iter(train_loader))
        
        assert hasattr(batch, 'x')
        assert hasattr(batch, 'edge_index')
        assert hasattr(batch, 'batch')


class TestWithoutDependencies:
    """Tests that work without PyTorch Geometric."""
    
    def test_import_guard(self):
        """Test that modules handle missing dependencies gracefully."""
        # This should not raise even without PyTorch Geometric
        try:
            from src.gnn import featurizer
        except ImportError:
            pass  # Expected if PyTorch Geometric not installed
