"""AlphaFold 3 integration for structure prediction.

This module provides integration with AlphaFold for:
- Protein structure prediction
- Protein-ligand complex prediction
- Structure analysis and comparison

Example:
    >>> from src.alphafold import AlphaFoldClient, StructureAnalyzer
    >>> client = AlphaFoldClient()
    >>> structure = client.predict_complex(protein_seq, ligand_smiles)
    >>> analyzer = StructureAnalyzer(structure)
    >>> binding_site = analyzer.find_binding_site()
"""

from .client import AlphaFoldClient, AlphaFoldConfig
from .structure_analysis import StructureAnalyzer, BindingSite
from .complex_prediction import ComplexPredictor

__all__ = [
    "AlphaFoldClient",
    "AlphaFoldConfig",
    "StructureAnalyzer",
    "BindingSite",
    "ComplexPredictor",
]
