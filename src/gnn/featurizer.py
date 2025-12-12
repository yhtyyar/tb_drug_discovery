"""Molecular graph featurization for GNN models.

This module converts SMILES strings to PyTorch Geometric graph representations
with atom and bond features suitable for graph neural networks.

Features:
- Atom features: element, degree, hybridization, aromaticity, formal charge, etc.
- Bond features: bond type, conjugation, ring membership, stereo
- Support for batch processing and DataLoader integration

Example:
    >>> featurizer = MolecularGraphFeaturizer()
    >>> data = featurizer.smiles_to_graph("CCO")
    >>> print(data.x.shape)  # Node features
    >>> print(data.edge_index.shape)  # Edge connectivity
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

try:
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    logger.warning("PyTorch Geometric not installed. GNN features will be limited.")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# Atom feature specifications
ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),  # H to Og
    'degree': [0, 1, 2, 3, 4, 5, 6],
    'formal_charge': [-3, -2, -1, 0, 1, 2, 3],
    'num_hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ] if HAS_RDKIT else [],
}

# Bond feature specifications  
BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ] if HAS_RDKIT else [],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ] if HAS_RDKIT else [],
}


def one_hot_encoding(value, choices: List) -> List[int]:
    """One-hot encode a value from a list of choices.
    
    Args:
        value: Value to encode.
        choices: List of possible values.
        
    Returns:
        One-hot encoded list.
    """
    encoding = [0] * (len(choices) + 1)  # +1 for unknown
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        encoding[-1] = 1  # Unknown
    return encoding


class MolecularGraphFeaturizer:
    """Convert molecules to graph representations for GNN.
    
    This class creates node (atom) and edge (bond) features suitable
    for graph neural networks using PyTorch Geometric.
    
    Args:
        atom_features: List of atom feature names to include.
        bond_features: List of bond feature names to include.
        add_self_loops: Whether to add self-loops to the graph.
        add_hydrogens: Whether to add explicit hydrogens.
        
    Attributes:
        atom_dim: Dimensionality of atom feature vectors.
        bond_dim: Dimensionality of bond feature vectors.
        
    Example:
        >>> featurizer = MolecularGraphFeaturizer()
        >>> graph = featurizer.smiles_to_graph("c1ccccc1")
        >>> print(f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
    """
    
    def __init__(
        self,
        atom_features: Optional[List[str]] = None,
        bond_features: Optional[List[str]] = None,
        add_self_loops: bool = False,
        add_hydrogens: bool = False,
    ):
        if not HAS_RDKIT:
            raise ImportError("RDKit is required for molecular featurization")
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("PyTorch Geometric is required for graph features")
        
        self.atom_feature_names = atom_features or [
            'atomic_num', 'degree', 'formal_charge', 'num_hs',
            'hybridization', 'is_aromatic', 'is_in_ring'
        ]
        self.bond_feature_names = bond_features or [
            'bond_type', 'is_conjugated', 'is_in_ring', 'stereo'
        ]
        self.add_self_loops = add_self_loops
        self.add_hydrogens = add_hydrogens
        
        # Calculate feature dimensions
        self.atom_dim = self._calculate_atom_dim()
        self.bond_dim = self._calculate_bond_dim()
        
        logger.info(f"Featurizer initialized: atom_dim={self.atom_dim}, bond_dim={self.bond_dim}")
    
    def _calculate_atom_dim(self) -> int:
        """Calculate total atom feature dimensionality."""
        dim = 0
        for feat in self.atom_feature_names:
            if feat in ATOM_FEATURES:
                dim += len(ATOM_FEATURES[feat]) + 1  # +1 for unknown
            elif feat in ('is_aromatic', 'is_in_ring'):
                dim += 1  # Binary
        return dim
    
    def _calculate_bond_dim(self) -> int:
        """Calculate total bond feature dimensionality."""
        dim = 0
        for feat in self.bond_feature_names:
            if feat in BOND_FEATURES:
                dim += len(BOND_FEATURES[feat]) + 1
            elif feat in ('is_conjugated', 'is_in_ring'):
                dim += 1
        return dim
    
    def get_atom_features(self, atom) -> List[float]:
        """Extract features for a single atom.
        
        Args:
            atom: RDKit atom object.
            
        Returns:
            List of atom features.
        """
        features = []
        
        for feat_name in self.atom_feature_names:
            if feat_name == 'atomic_num':
                features.extend(one_hot_encoding(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']))
            elif feat_name == 'degree':
                features.extend(one_hot_encoding(atom.GetTotalDegree(), ATOM_FEATURES['degree']))
            elif feat_name == 'formal_charge':
                features.extend(one_hot_encoding(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']))
            elif feat_name == 'num_hs':
                features.extend(one_hot_encoding(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs']))
            elif feat_name == 'hybridization':
                features.extend(one_hot_encoding(atom.GetHybridization(), ATOM_FEATURES['hybridization']))
            elif feat_name == 'is_aromatic':
                features.append(float(atom.GetIsAromatic()))
            elif feat_name == 'is_in_ring':
                features.append(float(atom.IsInRing()))
        
        return features
    
    def get_bond_features(self, bond) -> List[float]:
        """Extract features for a single bond.
        
        Args:
            bond: RDKit bond object.
            
        Returns:
            List of bond features.
        """
        features = []
        
        for feat_name in self.bond_feature_names:
            if feat_name == 'bond_type':
                features.extend(one_hot_encoding(bond.GetBondType(), BOND_FEATURES['bond_type']))
            elif feat_name == 'is_conjugated':
                features.append(float(bond.GetIsConjugated()))
            elif feat_name == 'is_in_ring':
                features.append(float(bond.IsInRing()))
            elif feat_name == 'stereo':
                features.extend(one_hot_encoding(bond.GetStereo(), BOND_FEATURES['stereo']))
        
        return features
    
    def smiles_to_graph(
        self,
        smiles: str,
        y: Optional[Union[float, int, List]] = None,
    ) -> Optional[Data]:
        """Convert SMILES to PyTorch Geometric Data object.
        
        Args:
            smiles: SMILES string.
            y: Target value(s) for the molecule.
            
        Returns:
            PyTorch Geometric Data object or None if conversion fails.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Failed to parse SMILES: {smiles}")
            return None
        
        if self.add_hydrogens:
            mol = Chem.AddHs(mol)
        
        # Get atom features
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.get_atom_features(atom))
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # Get bond features and edge indices
        edge_index = []
        edge_attr = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            bond_features = self.get_bond_features(bond)
            
            # Add both directions (undirected graph)
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append(bond_features)
            edge_attr.append(bond_features)
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            # Handle molecules with no bonds (single atoms)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, self.bond_dim), dtype=torch.float)
        
        # Add self-loops if requested
        if self.add_self_loops:
            num_nodes = mol.GetNumAtoms()
            self_loops = torch.arange(num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, self_loops], dim=1)
            self_loop_attr = torch.zeros((num_nodes, self.bond_dim), dtype=torch.float)
            edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        
        # Create Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.smiles = smiles
        data.num_nodes = mol.GetNumAtoms()
        
        # Add target if provided
        if y is not None:
            if isinstance(y, (int, float)):
                data.y = torch.tensor([y], dtype=torch.float)
            else:
                data.y = torch.tensor(y, dtype=torch.float)
        
        return data
    
    def batch_smiles_to_graphs(
        self,
        smiles_list: List[str],
        y_list: Optional[List] = None,
        progress: bool = True,
    ) -> List[Data]:
        """Convert multiple SMILES to graphs.
        
        Args:
            smiles_list: List of SMILES strings.
            y_list: Optional list of target values.
            progress: Show progress bar.
            
        Returns:
            List of Data objects (excluding failed conversions).
        """
        graphs = []
        
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Featurizing")
            except ImportError:
                iterator = enumerate(smiles_list)
        else:
            iterator = enumerate(smiles_list)
        
        for i, smiles in iterator:
            y = y_list[i] if y_list is not None else None
            graph = self.smiles_to_graph(smiles, y)
            if graph is not None:
                graphs.append(graph)
        
        logger.info(f"Converted {len(graphs)}/{len(smiles_list)} molecules to graphs")
        return graphs


class MoleculeDataset(Dataset):
    """PyTorch Dataset for molecular graphs.
    
    This dataset wraps a list of PyTorch Geometric Data objects
    for use with DataLoader.
    
    Args:
        smiles_list: List of SMILES strings.
        y_list: List of target values.
        featurizer: MolecularGraphFeaturizer instance.
        transform: Optional transform to apply.
        
    Example:
        >>> dataset = MoleculeDataset(smiles, targets)
        >>> loader = DataLoader(dataset, batch_size=32)
    """
    
    def __init__(
        self,
        smiles_list: List[str],
        y_list: Optional[List] = None,
        featurizer: Optional[MolecularGraphFeaturizer] = None,
        transform=None,
    ):
        self.smiles_list = smiles_list
        self.y_list = y_list
        self.transform = transform
        
        if featurizer is None:
            featurizer = MolecularGraphFeaturizer()
        self.featurizer = featurizer
        
        # Pre-compute graphs
        self.graphs = self.featurizer.batch_smiles_to_graphs(
            smiles_list, y_list, progress=True
        )
        
        # Store valid indices
        self.valid_indices = list(range(len(self.graphs)))
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Data:
        data = self.graphs[idx]
        
        if self.transform is not None:
            data = self.transform(data)
        
        return data
    
    @property
    def num_node_features(self) -> int:
        """Number of node (atom) features."""
        return self.featurizer.atom_dim
    
    @property
    def num_edge_features(self) -> int:
        """Number of edge (bond) features."""
        return self.featurizer.bond_dim
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        num_nodes = [g.num_nodes for g in self.graphs]
        num_edges = [g.edge_index.size(1) for g in self.graphs]
        
        stats = {
            'num_molecules': len(self.graphs),
            'avg_nodes': np.mean(num_nodes),
            'max_nodes': max(num_nodes),
            'min_nodes': min(num_nodes),
            'avg_edges': np.mean(num_edges),
            'node_feature_dim': self.num_node_features,
            'edge_feature_dim': self.num_edge_features,
        }
        
        if self.graphs[0].y is not None:
            targets = [g.y.item() for g in self.graphs if g.y is not None]
            stats['target_mean'] = np.mean(targets)
            stats['target_std'] = np.std(targets)
        
        return stats


def create_data_loaders(
    smiles_list: List[str],
    y_list: List,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    batch_size: int = 32,
    random_seed: int = 42,
    featurizer: Optional[MolecularGraphFeaturizer] = None,
) -> Tuple:
    """Create train/val/test DataLoaders.
    
    Args:
        smiles_list: List of SMILES.
        y_list: List of targets.
        train_ratio: Training set ratio.
        val_ratio: Validation set ratio.
        batch_size: Batch size.
        random_seed: Random seed for splitting.
        featurizer: Optional featurizer instance.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, featurizer).
    """
    from torch_geometric.loader import DataLoader
    
    # Create dataset
    if featurizer is None:
        featurizer = MolecularGraphFeaturizer()
    
    dataset = MoleculeDataset(smiles_list, y_list, featurizer)
    
    # Split dataset
    np.random.seed(random_seed)
    indices = np.random.permutation(len(dataset))
    
    n_train = int(len(dataset) * train_ratio)
    n_val = int(len(dataset) * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    test_data = [dataset[i] for i in test_idx]
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    logger.info(f"DataLoaders created: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    return train_loader, val_loader, test_loader, featurizer
