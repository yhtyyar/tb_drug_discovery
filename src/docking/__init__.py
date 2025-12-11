"""Molecular docking modules for TB drug discovery.

This package provides tools for:
- AutoDock Vina integration
- Protein structure preparation
- Batch docking workflows
- Result analysis

Classes:
    VinaDocker: AutoDock Vina interface
    ProteinPreparator: Protein preparation utilities
"""

from src.docking.protein_prep import ProteinPreparator, prepare_tb_target, TB_TARGETS
from src.docking.vina_docker import VinaDocker, DockingResult, run_docking_pipeline

__all__ = [
    "VinaDocker",
    "DockingResult", 
    "ProteinPreparator",
    "prepare_tb_target",
    "TB_TARGETS",
    "run_docking_pipeline",
]
