"""Graph Neural Network architectures for molecular property prediction.

This module implements various GNN architectures:
- GCN: Graph Convolutional Network
- GAT: Graph Attention Network
- MPNN: Message Passing Neural Network
- AttentiveFP: Attentive Fingerprint

All models support both classification and regression tasks.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

try:
    from torch_geometric.nn import (
        GCNConv, GATConv, NNConv, 
        global_mean_pool, global_add_pool, global_max_pool,
        BatchNorm, LayerNorm
    )
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    logger.warning("PyTorch Geometric not installed")


class BaseGNN(nn.Module):
    """Base class for GNN models.
    
    Provides common functionality for all GNN architectures including
    output layers, pooling, and forward pass structure.
    
    Args:
        hidden_dim: Hidden layer dimensionality.
        output_dim: Output dimensionality (1 for regression/binary classification).
        num_layers: Number of GNN layers.
        dropout: Dropout probability.
        task: 'classification' or 'regression'.
        pooling: Global pooling method ('mean', 'add', 'max').
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 3,
        dropout: float = 0.2,
        task: str = 'classification',
        pooling: str = 'mean',
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.task = task
        self.pooling = pooling
        
        # Pooling function
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'add':
            self.pool = global_add_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        # Output layers (to be set by subclass after GNN layers)
        self.output_mlp = None
    
    def _build_output_layers(self, input_dim: int):
        """Build output MLP layers."""
        self.output_mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.output_dim),
        )
    
    def get_embeddings(self, data: Data) -> torch.Tensor:
        """Get node embeddings before pooling (to be implemented by subclass)."""
        raise NotImplementedError
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass.
        
        Args:
            data: PyTorch Geometric Data object.
            
        Returns:
            Model predictions.
        """
        # Get node embeddings
        x = self.get_embeddings(data)
        
        # Global pooling
        x = self.pool(x, data.batch)
        
        # Output MLP
        out = self.output_mlp(x)
        
        # Apply sigmoid for binary classification
        if self.task == 'classification' and self.output_dim == 1:
            out = torch.sigmoid(out)
        
        return out.squeeze(-1)


class GCNModel(BaseGNN):
    """Graph Convolutional Network.
    
    Implements the GCN architecture from Kipf & Welling (2017):
    "Semi-Supervised Classification with Graph Convolutional Networks"
    
    Args:
        node_dim: Input node feature dimensionality.
        hidden_dim: Hidden layer dimensionality.
        output_dim: Output dimensionality.
        num_layers: Number of GCN layers.
        dropout: Dropout probability.
        task: 'classification' or 'regression'.
        
    Example:
        >>> model = GCNModel(node_dim=78, hidden_dim=128)
        >>> out = model(batch)
    """
    
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 3,
        dropout: float = 0.2,
        task: str = 'classification',
        **kwargs
    ):
        super().__init__(hidden_dim, output_dim, num_layers, dropout, task, **kwargs)
        
        self.node_dim = node_dim
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(node_dim, hidden_dim))
        self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Output layers
        self._build_output_layers(hidden_dim)
        
        logger.info(f"GCN initialized: layers={num_layers}, hidden={hidden_dim}")
    
    def get_embeddings(self, data: Data) -> torch.Tensor:
        """Get node embeddings from GCN layers."""
        x, edge_index = data.x, data.edge_index
        
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class GATModel(BaseGNN):
    """Graph Attention Network.
    
    Implements the GAT architecture from Veličković et al. (2018):
    "Graph Attention Networks"
    
    Args:
        node_dim: Input node feature dimensionality.
        hidden_dim: Hidden layer dimensionality.
        output_dim: Output dimensionality.
        num_layers: Number of GAT layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        task: 'classification' or 'regression'.
    """
    
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        task: str = 'classification',
        **kwargs
    ):
        super().__init__(hidden_dim, output_dim, num_layers, dropout, task, **kwargs)
        
        self.node_dim = node_dim
        self.num_heads = num_heads
        
        # GAT layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(node_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
        self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Final layer (single head)
        if num_layers > 1:
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Output layers
        self._build_output_layers(hidden_dim)
        
        logger.info(f"GAT initialized: layers={num_layers}, heads={num_heads}, hidden={hidden_dim}")
    
    def get_embeddings(self, data: Data) -> torch.Tensor:
        """Get node embeddings from GAT layers."""
        x, edge_index = data.x, data.edge_index
        
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < len(self.convs) - 1:  # No activation on last layer
                x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def get_attention_weights(self, data: Data) -> List[torch.Tensor]:
        """Extract attention weights from each layer.
        
        Useful for model interpretability.
        """
        x, edge_index = data.x, data.edge_index
        attention_weights = []
        
        for conv in self.convs:
            x, (edge_index_out, alpha) = conv(x, edge_index, return_attention_weights=True)
            attention_weights.append(alpha)
            x = F.elu(x)
        
        return attention_weights


class MPNNModel(BaseGNN):
    """Message Passing Neural Network.
    
    Implements the MPNN framework from Gilmer et al. (2017):
    "Neural Message Passing for Quantum Chemistry"
    
    Uses edge features in message passing.
    
    Args:
        node_dim: Input node feature dimensionality.
        edge_dim: Input edge feature dimensionality.
        hidden_dim: Hidden layer dimensionality.
        output_dim: Output dimensionality.
        num_layers: Number of MPNN layers.
        dropout: Dropout probability.
        task: 'classification' or 'regression'.
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 3,
        dropout: float = 0.2,
        task: str = 'classification',
        **kwargs
    ):
        super().__init__(hidden_dim, output_dim, num_layers, dropout, task, **kwargs)
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # Initial node embedding
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        # MPNN layers with edge network
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # Edge network for NNConv
            edge_nn = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim * hidden_dim)
            )
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_nn, aggr='mean'))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # GRU for updating node states
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Output layers
        self._build_output_layers(hidden_dim)
        
        logger.info(f"MPNN initialized: layers={num_layers}, hidden={hidden_dim}")
    
    def get_embeddings(self, data: Data) -> torch.Tensor:
        """Get node embeddings from MPNN layers."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Initial embedding
        x = self.node_encoder(x)
        h = x.unsqueeze(0)  # For GRU
        
        for conv, bn in zip(self.convs, self.batch_norms):
            # Message passing
            m = conv(x, edge_index, edge_attr)
            m = bn(m)
            m = F.relu(m)
            
            # GRU update
            m = m.unsqueeze(0)
            _, h = self.gru(m, h)
            x = h.squeeze(0)
            
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class AttentiveFPModel(BaseGNN):
    """Attentive Fingerprint Model.
    
    Implements the AttentiveFP architecture from Xiong et al. (2019):
    "Pushing the Boundaries of Molecular Representation for Drug Discovery"
    
    Uses a two-step attention mechanism for graph-level readout.
    
    Args:
        node_dim: Input node feature dimensionality.
        edge_dim: Input edge feature dimensionality.
        hidden_dim: Hidden layer dimensionality.
        output_dim: Output dimensionality.
        num_layers: Number of attention layers.
        num_timesteps: Number of readout timesteps.
        dropout: Dropout probability.
        task: 'classification' or 'regression'.
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 3,
        num_timesteps: int = 2,
        dropout: float = 0.2,
        task: str = 'classification',
        **kwargs
    ):
        super().__init__(hidden_dim, output_dim, num_layers, dropout, task, **kwargs)
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_timesteps = num_timesteps
        
        # Node and edge encoders
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        
        # Attention layers
        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
            )
        
        # Graph-level attention
        self.graph_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # GRU for readout
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Output layers
        self._build_output_layers(hidden_dim)
        
        logger.info(f"AttentiveFP initialized: layers={num_layers}, timesteps={num_timesteps}")
    
    def get_embeddings(self, data: Data) -> torch.Tensor:
        """Get node embeddings from AttentiveFP layers."""
        x, edge_index = data.x, data.edge_index
        
        # Initial encoding
        x = F.relu(self.node_encoder(x))
        
        # GAT layers
        for gat in self.gat_layers:
            x = gat(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass with attentive readout."""
        x = self.get_embeddings(data)
        batch = data.batch
        
        # Attentive readout over multiple timesteps
        # Initialize super-node for each graph
        num_graphs = batch.max().item() + 1
        h = torch.zeros(num_graphs, self.hidden_dim, device=x.device)
        
        for _ in range(self.num_timesteps):
            # Compute attention weights
            attention_scores = self.graph_attention(x)
            
            # Scatter softmax per graph
            attention_weights = torch.zeros_like(attention_scores)
            for g in range(num_graphs):
                mask = (batch == g)
                attention_weights[mask] = F.softmax(attention_scores[mask], dim=0)
            
            # Weighted sum of node features per graph
            context = torch.zeros(num_graphs, self.hidden_dim, device=x.device)
            for g in range(num_graphs):
                mask = (batch == g)
                context[g] = (attention_weights[mask] * x[mask]).sum(dim=0)
            
            # Update with GRU
            h = self.gru(context, h)
        
        # Output MLP
        out = self.output_mlp(h)
        
        if self.task == 'classification' and self.output_dim == 1:
            out = torch.sigmoid(out)
        
        return out.squeeze(-1)


def create_model(
    model_type: str,
    node_dim: int,
    edge_dim: int = 0,
    hidden_dim: int = 128,
    output_dim: int = 1,
    num_layers: int = 3,
    dropout: float = 0.2,
    task: str = 'classification',
    **kwargs
) -> BaseGNN:
    """Factory function to create GNN models.
    
    Args:
        model_type: One of 'gcn', 'gat', 'mpnn', 'attentivefp'.
        node_dim: Input node feature dimensionality.
        edge_dim: Input edge feature dimensionality (for MPNN/AttentiveFP).
        hidden_dim: Hidden layer dimensionality.
        output_dim: Output dimensionality.
        num_layers: Number of GNN layers.
        dropout: Dropout probability.
        task: 'classification' or 'regression'.
        **kwargs: Additional model-specific arguments.
        
    Returns:
        GNN model instance.
    """
    model_type = model_type.lower()
    
    if model_type == 'gcn':
        return GCNModel(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            task=task,
            **kwargs
        )
    elif model_type == 'gat':
        return GATModel(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            task=task,
            **kwargs
        )
    elif model_type == 'mpnn':
        if edge_dim == 0:
            raise ValueError("MPNN requires edge_dim > 0")
        return MPNNModel(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            task=task,
            **kwargs
        )
    elif model_type == 'attentivefp':
        return AttentiveFPModel(
            node_dim=node_dim,
            edge_dim=edge_dim if edge_dim > 0 else 1,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            task=task,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from: gcn, gat, mpnn, attentivefp")
