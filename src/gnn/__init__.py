"""Graph Neural Network modules for molecular property prediction.

This package provides:
- Molecular graph featurization (SMILES → PyTorch Geometric Data)
- GNN architectures (GCN, GAT, MPNN, AttentiveFP)
- Training and evaluation pipelines
- Ensemble models combining QSAR + GNN

Classes:
    MolecularGraphFeaturizer: Convert molecules to graphs
    GCNModel: Graph Convolutional Network
    GATModel: Graph Attention Network  
    MPNNModel: Message Passing Neural Network
    AttentiveFPModel: Attentive Fingerprint model
    GNNTrainer: Training pipeline
    EnsembleModel: QSAR + GNN ensemble
"""

from src.gnn.featurizer import MolecularGraphFeaturizer, MoleculeDataset
from src.gnn.models import GCNModel, GATModel, MPNNModel, AttentiveFPModel, create_model
from src.gnn.trainer import GNNTrainer, EarlyStopping
from src.gnn.ensemble import EnsembleModel

__all__ = [
    "MolecularGraphFeaturizer",
    "MoleculeDataset",
    "GCNModel",
    "GATModel", 
    "MPNNModel",
    "AttentiveFPModel",
    "create_model",
    "GNNTrainer",
    "EarlyStopping",
    "EnsembleModel",
]
