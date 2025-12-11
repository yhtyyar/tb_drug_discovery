"""AutoDock Vina integration for molecular docking.

This module provides a Python interface for running AutoDock Vina
docking simulations for TB drug discovery.

Usage:
    from src.docking import VinaDocker
    
    docker = VinaDocker(vina_path="vina")
    docker.prepare_receptor("protein.pdb", "protein.pdbqt")
    results = docker.dock("ligand.sdf", "protein.pdbqt", center=(0,0,0), size=(20,20,20))
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logger.warning("RDKit not available. Some features will be limited.")


@dataclass
class DockingResult:
    """Container for docking results."""
    
    ligand_name: str
    smiles: str
    affinity: float  # kcal/mol (negative = better)
    pose_file: Optional[str] = None
    rmsd_lb: float = 0.0
    rmsd_ub: float = 0.0
    num_poses: int = 1
    all_affinities: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "ligand_name": self.ligand_name,
            "smiles": self.smiles,
            "affinity": self.affinity,
            "pose_file": self.pose_file,
            "rmsd_lb": self.rmsd_lb,
            "rmsd_ub": self.rmsd_ub,
            "num_poses": self.num_poses,
            "all_affinities": self.all_affinities,
        }


class VinaDocker:
    """AutoDock Vina docking interface.
    
    This class provides methods for:
    - Preparing ligands (SMILES → PDBQT)
    - Preparing receptors (PDB → PDBQT)
    - Running docking simulations
    - Parsing and analyzing results
    
    Args:
        vina_path: Path to Vina executable (default: "vina" assumes it's in PATH).
        obabel_path: Path to Open Babel executable for format conversions.
        num_modes: Number of binding poses to generate.
        exhaustiveness: Exhaustiveness of the search (higher = more thorough).
        energy_range: Maximum energy difference between best and worst pose.
        cpu: Number of CPUs to use (0 = auto-detect).
        
    Example:
        >>> docker = VinaDocker()
        >>> docker.set_receptor("protein.pdbqt", center=(10, 20, 30), size=(25, 25, 25))
        >>> result = docker.dock_smiles("CCO", "ethanol")
        >>> print(f"Binding affinity: {result.affinity} kcal/mol")
    """
    
    def __init__(
        self,
        vina_path: str = "vina",
        obabel_path: str = "obabel",
        num_modes: int = 9,
        exhaustiveness: int = 8,
        energy_range: float = 3.0,
        cpu: int = 0,
    ):
        self.vina_path = vina_path
        self.obabel_path = obabel_path
        self.num_modes = num_modes
        self.exhaustiveness = exhaustiveness
        self.energy_range = energy_range
        self.cpu = cpu
        
        self.receptor_path: Optional[str] = None
        self.center: Optional[Tuple[float, float, float]] = None
        self.size: Optional[Tuple[float, float, float]] = None
        
        self._temp_dir = tempfile.mkdtemp(prefix="vina_")
        logger.info(f"VinaDocker initialized. Temp dir: {self._temp_dir}")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required external tools are available.
        
        Returns:
            Dictionary with tool availability status.
        """
        status = {"vina": False, "obabel": False, "rdkit": HAS_RDKIT}
        
        # Check Vina
        try:
            result = subprocess.run(
                [self.vina_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            status["vina"] = result.returncode == 0 or "AutoDock Vina" in result.stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        # Check Open Babel
        try:
            result = subprocess.run(
                [self.obabel_path, "-V"],
                capture_output=True,
                text=True,
                timeout=10
            )
            status["obabel"] = result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return status
    
    def set_receptor(
        self,
        receptor_path: str,
        center: Tuple[float, float, float],
        size: Tuple[float, float, float] = (25.0, 25.0, 25.0),
    ) -> None:
        """Set the receptor for docking.
        
        Args:
            receptor_path: Path to receptor PDBQT file.
            center: Center of the docking box (x, y, z) in Angstroms.
            size: Size of the docking box (x, y, z) in Angstroms.
        """
        if not Path(receptor_path).exists():
            raise FileNotFoundError(f"Receptor file not found: {receptor_path}")
        
        self.receptor_path = receptor_path
        self.center = center
        self.size = size
        
        logger.info(f"Receptor set: {receptor_path}")
        logger.info(f"Docking box: center={center}, size={size}")
    
    def prepare_ligand_from_smiles(
        self,
        smiles: str,
        name: str = "ligand",
        optimize: bool = True,
    ) -> str:
        """Convert SMILES to PDBQT format for docking.
        
        Args:
            smiles: SMILES string of the ligand.
            name: Name for the ligand file.
            optimize: Whether to perform 3D optimization.
            
        Returns:
            Path to the generated PDBQT file.
        """
        if not HAS_RDKIT:
            raise ImportError("RDKit is required for SMILES to PDBQT conversion")
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        # Add hydrogens
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates
        if optimize:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        else:
            AllChem.EmbedMolecule(mol, randomSeed=42)
        
        # Save as SDF
        sdf_path = os.path.join(self._temp_dir, f"{name}.sdf")
        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()
        
        # Convert to PDBQT using Open Babel
        pdbqt_path = os.path.join(self._temp_dir, f"{name}.pdbqt")
        
        try:
            subprocess.run(
                [self.obabel_path, sdf_path, "-O", pdbqt_path, "-p", "7.4"],
                capture_output=True,
                check=True,
                timeout=60
            )
        except subprocess.SubprocessError as e:
            logger.error(f"Open Babel conversion failed: {e}")
            raise
        
        return pdbqt_path
    
    def prepare_receptor(
        self,
        pdb_path: str,
        output_path: Optional[str] = None,
        remove_water: bool = True,
        remove_heteroatoms: bool = False,
    ) -> str:
        """Prepare receptor PDB file for docking.
        
        Args:
            pdb_path: Path to input PDB file.
            output_path: Path for output PDBQT file. If None, uses temp directory.
            remove_water: Remove water molecules.
            remove_heteroatoms: Remove all heteroatoms (ligands, ions, etc.).
            
        Returns:
            Path to the prepared PDBQT file.
        """
        if output_path is None:
            output_path = os.path.join(self._temp_dir, "receptor.pdbqt")
        
        # Build Open Babel command
        cmd = [self.obabel_path, pdb_path, "-O", output_path, "-xr"]
        
        if remove_water:
            cmd.extend(["--delete", "HOH"])
        
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
            logger.info(f"Receptor prepared: {output_path}")
        except subprocess.SubprocessError as e:
            logger.error(f"Receptor preparation failed: {e}")
            raise
        
        return output_path
    
    def dock_smiles(
        self,
        smiles: str,
        name: str = "ligand",
    ) -> DockingResult:
        """Dock a molecule specified by SMILES.
        
        Args:
            smiles: SMILES string of the ligand.
            name: Name for the ligand.
            
        Returns:
            DockingResult with binding affinity and poses.
        """
        if self.receptor_path is None:
            raise ValueError("Receptor not set. Call set_receptor() first.")
        
        # Prepare ligand
        ligand_pdbqt = self.prepare_ligand_from_smiles(smiles, name)
        
        # Run docking
        return self._run_vina(ligand_pdbqt, smiles, name)
    
    def dock_file(
        self,
        ligand_path: str,
        smiles: str = "",
        name: Optional[str] = None,
    ) -> DockingResult:
        """Dock a ligand from a file.
        
        Args:
            ligand_path: Path to ligand file (PDBQT, SDF, MOL2).
            smiles: SMILES string (for record keeping).
            name: Name for the ligand.
            
        Returns:
            DockingResult with binding affinity and poses.
        """
        if self.receptor_path is None:
            raise ValueError("Receptor not set. Call set_receptor() first.")
        
        ligand_path = Path(ligand_path)
        if name is None:
            name = ligand_path.stem
        
        # Convert to PDBQT if needed
        if ligand_path.suffix.lower() != ".pdbqt":
            pdbqt_path = os.path.join(self._temp_dir, f"{name}.pdbqt")
            subprocess.run(
                [self.obabel_path, str(ligand_path), "-O", pdbqt_path],
                capture_output=True,
                check=True
            )
            ligand_path = pdbqt_path
        
        return self._run_vina(str(ligand_path), smiles, name)
    
    def _run_vina(
        self,
        ligand_pdbqt: str,
        smiles: str,
        name: str,
    ) -> DockingResult:
        """Run Vina docking.
        
        Args:
            ligand_pdbqt: Path to ligand PDBQT file.
            smiles: SMILES string for record.
            name: Ligand name.
            
        Returns:
            DockingResult object.
        """
        output_path = os.path.join(self._temp_dir, f"{name}_out.pdbqt")
        log_path = os.path.join(self._temp_dir, f"{name}_log.txt")
        
        # Build Vina command
        cmd = [
            self.vina_path,
            "--receptor", self.receptor_path,
            "--ligand", ligand_pdbqt,
            "--center_x", str(self.center[0]),
            "--center_y", str(self.center[1]),
            "--center_z", str(self.center[2]),
            "--size_x", str(self.size[0]),
            "--size_y", str(self.size[1]),
            "--size_z", str(self.size[2]),
            "--out", output_path,
            "--log", log_path,
            "--num_modes", str(self.num_modes),
            "--exhaustiveness", str(self.exhaustiveness),
            "--energy_range", str(self.energy_range),
        ]
        
        if self.cpu > 0:
            cmd.extend(["--cpu", str(self.cpu)])
        
        logger.info(f"Running Vina for {name}...")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 min timeout per ligand
            )
            
            if result.returncode != 0:
                logger.error(f"Vina failed: {result.stderr}")
                raise subprocess.SubprocessError(result.stderr)
                
        except subprocess.TimeoutExpired:
            logger.error(f"Vina timeout for {name}")
            raise
        
        # Parse results
        affinities = self._parse_vina_output(log_path)
        
        return DockingResult(
            ligand_name=name,
            smiles=smiles,
            affinity=affinities[0] if affinities else 0.0,
            pose_file=output_path if Path(output_path).exists() else None,
            num_poses=len(affinities),
            all_affinities=affinities,
        )
    
    def _parse_vina_output(self, log_path: str) -> List[float]:
        """Parse Vina log file for binding affinities.
        
        Args:
            log_path: Path to Vina log file.
            
        Returns:
            List of binding affinities (kcal/mol).
        """
        affinities = []
        
        try:
            with open(log_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and line[0].isdigit():
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                affinities.append(float(parts[1]))
                            except ValueError:
                                continue
        except FileNotFoundError:
            logger.warning(f"Log file not found: {log_path}")
        
        return affinities
    
    def dock_batch(
        self,
        smiles_list: List[str],
        names: Optional[List[str]] = None,
        progress: bool = True,
    ) -> pd.DataFrame:
        """Dock multiple molecules.
        
        Args:
            smiles_list: List of SMILES strings.
            names: List of ligand names. If None, uses "ligand_0", "ligand_1", etc.
            progress: Show progress bar.
            
        Returns:
            DataFrame with docking results.
        """
        if names is None:
            names = [f"ligand_{i}" for i in range(len(smiles_list))]
        
        results = []
        
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(zip(smiles_list, names), total=len(smiles_list), desc="Docking")
            except ImportError:
                iterator = zip(smiles_list, names)
        else:
            iterator = zip(smiles_list, names)
        
        for smiles, name in iterator:
            try:
                result = self.dock_smiles(smiles, name)
                results.append(result.to_dict())
            except Exception as e:
                logger.warning(f"Docking failed for {name}: {e}")
                results.append({
                    "ligand_name": name,
                    "smiles": smiles,
                    "affinity": np.nan,
                    "error": str(e),
                })
        
        return pd.DataFrame(results)
    
    def get_binding_site_from_ligand(
        self,
        pdb_path: str,
        ligand_name: str = "LIG",
        padding: float = 5.0,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Calculate docking box from co-crystallized ligand.
        
        Args:
            pdb_path: Path to PDB file with ligand.
            ligand_name: Residue name of the ligand.
            padding: Extra space around ligand (Angstroms).
            
        Returns:
            Tuple of (center, size) for the docking box.
        """
        coords = []
        
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    resname = line[17:20].strip()
                    if resname == ligand_name:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
        
        if not coords:
            raise ValueError(f"Ligand '{ligand_name}' not found in {pdb_path}")
        
        coords = np.array(coords)
        center = tuple(coords.mean(axis=0))
        
        extent = coords.max(axis=0) - coords.min(axis=0)
        size = tuple(extent + 2 * padding)
        
        logger.info(f"Binding site from {ligand_name}: center={center}, size={size}")
        
        return center, size
    
    def cleanup(self) -> None:
        """Remove temporary files."""
        import shutil
        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)
            logger.info(f"Cleaned up temp directory: {self._temp_dir}")


def run_docking_pipeline(
    receptor_pdb: str,
    smiles_csv: str,
    output_csv: str,
    center: Tuple[float, float, float],
    size: Tuple[float, float, float] = (25, 25, 25),
    smiles_col: str = "smiles",
    name_col: Optional[str] = None,
    exhaustiveness: int = 8,
) -> pd.DataFrame:
    """Run complete docking pipeline.
    
    Args:
        receptor_pdb: Path to receptor PDB file.
        smiles_csv: Path to CSV with SMILES.
        output_csv: Path to save results.
        center: Docking box center.
        size: Docking box size.
        smiles_col: Column name for SMILES.
        name_col: Column name for molecule names.
        exhaustiveness: Search exhaustiveness.
        
    Returns:
        DataFrame with docking results.
    """
    logger.info("Starting docking pipeline...")
    
    # Load molecules
    df = pd.read_csv(smiles_csv)
    smiles_list = df[smiles_col].tolist()
    
    if name_col and name_col in df.columns:
        names = df[name_col].tolist()
    else:
        names = None
    
    # Initialize docker
    docker = VinaDocker(exhaustiveness=exhaustiveness)
    
    # Prepare receptor
    receptor_pdbqt = docker.prepare_receptor(receptor_pdb)
    docker.set_receptor(receptor_pdbqt, center, size)
    
    # Run docking
    results = docker.dock_batch(smiles_list, names)
    
    # Save results
    results.to_csv(output_csv, index=False)
    logger.info(f"Results saved to: {output_csv}")
    
    # Cleanup
    docker.cleanup()
    
    return results
