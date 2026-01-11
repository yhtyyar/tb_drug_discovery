"""Graph Transformer architectures for molecular property prediction.

This module implements modern Graph Transformer architectures including:
- GraphGPS: General, Powerful, Scalable Graph Transformers
- Graph Attention with Positional Encodings
- Hybrid Message Passing + Attention models

These architectures combine the strengths of message passing GNNs
with global attention mechanisms for improved molecular modeling.

Reference:
- Rampášek et al. (2022) "Recipe for a General, Powerful, Scalable Graph Transformer"

Example:
    >>> model = GraphGPS(node_dim=78, hidden_dim=256)
    >>> output = model(batch)
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

try:
    from torch_geometric.nn import (
        GINEConv, GATv2Conv, TransformerConv,
        global_mean_pool, global_add_pool, global_max_pool,
    )
    from torch_geometric.data import Batch
    from torch_geometric.utils import to_dense_batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    logger.warning("PyTorch Geometric not installed. Graph Transformer unavailable.")


class PositionalEncoding(nn.Module):
    """Positional encoding for graph nodes.
    
    Supports multiple encoding types:
    - Random Walk PE (RWPE)
    - Laplacian PE (LapPE)
    - Learnable PE
    
    Args:
        dim: Encoding dimension.
        pe_type: Type of positional encoding.
        max_nodes: Maximum number of nodes (for learnable PE).
    """
    
    def __init__(
        self,
        dim: int,
        pe_type: str = "learnable",
        max_nodes: int = 500,
    ):
        super().__init__()
        self.dim = dim
        self.pe_type = pe_type
        
        if pe_type == "learnable":
            self.pe = nn.Embedding(max_nodes, dim)
        elif pe_type == "sinusoidal":
            self.register_buffer("pe", self._create_sinusoidal_pe(max_nodes, dim))
        else:
            self.pe = None
    
    def _create_sinusoidal_pe(self, max_len: int, dim: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, batch_idx: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Get positional encodings for nodes.
        
        Args:
            batch_idx: Batch assignment for each node.
            num_nodes: Total number of nodes.
            
        Returns:
            Positional encodings (num_nodes, dim).
        """
        if self.pe_type == "learnable":
            # Use node position within each graph
            positions = torch.zeros(num_nodes, dtype=torch.long, device=batch_idx.device)
            
            # Count position within each graph
            for i in range(num_nodes):
                batch_i = batch_idx[i].item()
                positions[i] = (batch_idx[:i] == batch_i).sum()
            
            return self.pe(positions.clamp(max=self.pe.num_embeddings - 1))
        
        elif self.pe_type == "sinusoidal":
            positions = torch.arange(num_nodes, device=batch_idx.device)
            return self.pe[positions.clamp(max=self.pe.size(0) - 1)]
        
        return torch.zeros(num_nodes, self.dim, device=batch_idx.device)


class GraphMultiHeadAttention(nn.Module):
    """Multi-head self-attention for graphs.
    
    Applies attention across all nodes within each graph in the batch.
    
    Args:
        hidden_dim: Hidden dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        bias: Use bias in projections.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply multi-head attention.
        
        Args:
            x: Node features (num_nodes, hidden_dim).
            batch: Batch assignment (num_nodes,).
            edge_index: Edge indices (optional, for edge-biased attention).
            attention_mask: Attention mask.
            
        Returns:
            Updated node features.
        """
        # Convert to dense batch for attention
        x_dense, mask = to_dense_batch(x, batch)
        batch_size, max_nodes, _ = x_dense.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x_dense)
        K = self.k_proj(x_dense)
        V = self.v_proj(x_dense)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, max_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, max_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, max_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        
        # Apply mask (prevent attention to padding)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~mask_expanded, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ V
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, max_nodes, self.hidden_dim)
        out = self.out_proj(out)
        
        # Convert back to sparse representation
        out = out[mask]
        
        return out


class GPSLayer(nn.Module):
    """GPS (General, Powerful, Scalable) layer.
    
    Combines local message passing with global attention
    in a single layer.
    
    Args:
        hidden_dim: Hidden dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        attn_dropout: Attention dropout.
        ffn_ratio: FFN hidden dim ratio.
        local_gnn: Type of local GNN ('gine', 'gat', None).
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        ffn_ratio: float = 4.0,
        local_gnn: str = "gine",
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.local_gnn_type = local_gnn
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Local message passing (optional)
        if local_gnn == "gine" and HAS_TORCH_GEOMETRIC:
            self.local_gnn = GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                edge_dim=hidden_dim,
            )
        elif local_gnn == "gat" and HAS_TORCH_GEOMETRIC:
            self.local_gnn = GATv2Conv(
                hidden_dim, hidden_dim // num_heads,
                heads=num_heads, edge_dim=hidden_dim,
            )
        else:
            self.local_gnn = None
        
        # Global attention
        self.global_attn = GraphMultiHeadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
        )
        
        # Feed-forward network
        ffn_dim = int(hidden_dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of GPS layer.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            edge_attr: Edge features.
            batch: Batch assignment.
            
        Returns:
            Updated node features.
        """
        # Local message passing
        if self.local_gnn is not None:
            h_local = self.norm1(x)
            if edge_attr is not None:
                h_local = self.local_gnn(h_local, edge_index, edge_attr)
            else:
                h_local = self.local_gnn(h_local, edge_index)
            x = x + self.dropout(h_local)
        
        # Global attention
        h_global = self.norm2(x)
        h_global = self.global_attn(h_global, batch, edge_index)
        x = x + self.dropout(h_global)
        
        # Feed-forward
        h_ffn = self.norm3(x)
        h_ffn = self.ffn(h_ffn)
        x = x + h_ffn
        
        return x


class GraphGPS(nn.Module):
    """GraphGPS: General, Powerful, Scalable Graph Transformer.
    
    A hybrid architecture combining:
    - Local message passing (GINE, GAT)
    - Global self-attention
    - Positional encodings
    
    Args:
        node_dim: Input node feature dimension.
        edge_dim: Edge feature dimension (0 if no edge features).
        hidden_dim: Hidden dimension.
        num_layers: Number of GPS layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        task: 'classification' or 'regression'.
        local_gnn: Type of local GNN ('gine', 'gat', None).
        pooling: Graph pooling method ('mean', 'add', 'max').
        
    Example:
        >>> model = GraphGPS(node_dim=78, hidden_dim=256, num_layers=4)
        >>> output = model(batch)  # batch from PyG DataLoader
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        task: str = "classification",
        local_gnn: str = "gine",
        pooling: str = "mean",
    ):
        super().__init__()
        
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("PyTorch Geometric required for GraphGPS")
        
        self.task = task
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Edge embedding (if edge features present)
        self.edge_encoder = None
        if edge_dim > 0:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
            )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, pe_type="learnable")
        
        # GPS layers
        self.layers = nn.ModuleList([
            GPSLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                local_gnn=local_gnn,
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Pooling
        self.pooling = pooling
        if pooling == "mean":
            self.pool_fn = global_mean_pool
        elif pooling == "add":
            self.pool_fn = global_add_pool
        elif pooling == "max":
            self.pool_fn = global_max_pool
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        if task == "classification":
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()
        
        logger.info(f"GraphGPS initialized: {num_layers} layers, {hidden_dim} dim, {num_heads} heads")
    
    def forward(self, data) -> torch.Tensor:
        """Forward pass.
        
        Args:
            data: PyG Batch object with x, edge_index, edge_attr, batch.
            
        Returns:
            Predictions (batch_size, 1).
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = data.batch
        
        # Node embedding
        x = self.node_encoder(x)
        
        # Add positional encoding
        x = x + self.pos_encoder(batch, x.size(0))
        
        # Edge embedding
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)
        elif self.edge_encoder is not None:
            # Create dummy edge features
            edge_attr = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)
        
        # GPS layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Pooling
        x = self.pool_fn(x, batch)
        
        # Output
        out = self.output_head(x)
        out = self.output_activation(out)
        
        return out.squeeze(-1)
    
    def get_embeddings(self, data) -> torch.Tensor:
        """Get graph-level embeddings before output head.
        
        Args:
            data: PyG Batch object.
            
        Returns:
            Graph embeddings (batch_size, hidden_dim).
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = data.batch
        
        x = self.node_encoder(x)
        x = x + self.pos_encoder(batch, x.size(0))
        
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)
        elif self.edge_encoder is not None:
            edge_attr = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch)
        
        x = self.final_norm(x)
        x = self.pool_fn(x, batch)
        
        return x


class LightGraphTransformer(nn.Module):
    """Lightweight Graph Transformer for faster training.
    
    Uses TransformerConv from PyG instead of custom attention.
    Good for quick experiments and baseline comparisons.
    
    Args:
        node_dim: Input node feature dimension.
        edge_dim: Edge feature dimension.
        hidden_dim: Hidden dimension.
        num_layers: Number of layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        task: 'classification' or 'regression'.
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        task: str = "classification",
    ):
        super().__init__()
        
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("PyTorch Geometric required")
        
        self.task = task
        
        # Input projection
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(
                TransformerConv(
                    hidden_dim, hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=edge_dim if edge_dim > 0 else None,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        if task == "classification":
            self.output_activation = nn.Sigmoid()
        else:
            self.output_activation = nn.Identity()
    
    def forward(self, data) -> torch.Tensor:
        """Forward pass."""
        x = self.node_encoder(data.x)
        edge_attr = getattr(data, 'edge_attr', None)
        
        for layer, norm in zip(self.layers, self.norms):
            if edge_attr is not None:
                x = x + layer(x, data.edge_index, edge_attr)
            else:
                x = x + layer(x, data.edge_index)
            x = norm(x)
        
        x = global_mean_pool(x, data.batch)
        out = self.output_head(x)
        out = self.output_activation(out)
        
        return out.squeeze(-1)


def create_graph_transformer(
    model_type: str = "gps",
    node_dim: int = 78,
    edge_dim: int = 0,
    hidden_dim: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    task: str = "classification",
    **kwargs,
) -> nn.Module:
    """Factory function to create Graph Transformer models.
    
    Args:
        model_type: 'gps' or 'light'.
        node_dim: Input node feature dimension.
        edge_dim: Edge feature dimension.
        hidden_dim: Hidden dimension.
        num_layers: Number of layers.
        num_heads: Number of attention heads.
        task: 'classification' or 'regression'.
        
    Returns:
        Graph Transformer model.
    """
    if model_type == "gps":
        return GraphGPS(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            task=task,
            **kwargs,
        )
    elif model_type == "light":
        return LightGraphTransformer(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            task=task,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
