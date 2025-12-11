"""Protein preparation utilities for molecular docking.

This module provides tools for preparing protein structures
for docking simulations with AutoDock Vina.

Functions:
    - download_pdb: Download PDB structure from RCSB
    - clean_protein: Remove waters, ions, alternate conformations
    - add_hydrogens: Add hydrogen atoms
    - extract_binding_site: Get binding site coordinates
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from loguru import logger


class ProteinPreparator:
    """Protein structure preparation for docking.
    
    This class handles:
    - Downloading PDB structures from RCSB
    - Cleaning structures (removing waters, ions)
    - Extracting binding site information
    - Adding hydrogens and converting to PDBQT
    
    Example:
        >>> prep = ProteinPreparator()
        >>> pdb_path = prep.download_pdb("4TZK")
        >>> clean_path = prep.clean_protein(pdb_path)
        >>> center, size = prep.get_binding_site(clean_path, "TCL")
    """
    
    def __init__(self, work_dir: Optional[str] = None):
        """Initialize protein preparator.
        
        Args:
            work_dir: Working directory for files. If None, uses temp directory.
        """
        if work_dir:
            self.work_dir = Path(work_dir)
            self.work_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.work_dir = Path(tempfile.mkdtemp(prefix="protein_prep_"))
        
        logger.info(f"ProteinPreparator initialized. Work dir: {self.work_dir}")
    
    def download_pdb(
        self,
        pdb_id: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Download PDB structure from RCSB.
        
        Args:
            pdb_id: 4-character PDB ID (e.g., "4TZK" for InhA).
            output_path: Output file path. If None, saves to work_dir.
            
        Returns:
            Path to downloaded PDB file.
        """
        pdb_id = pdb_id.upper()
        
        if output_path is None:
            output_path = str(self.work_dir / f"{pdb_id}.pdb")
        
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        
        logger.info(f"Downloading PDB {pdb_id} from RCSB...")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'w') as f:
                f.write(response.text)
            
            logger.info(f"Downloaded: {output_path}")
            return output_path
            
        except requests.RequestException as e:
            logger.error(f"Failed to download PDB {pdb_id}: {e}")
            raise
    
    def clean_protein(
        self,
        pdb_path: str,
        output_path: Optional[str] = None,
        remove_water: bool = True,
        remove_ions: bool = True,
        remove_ligands: bool = True,
        keep_ligand: Optional[str] = None,
        select_chain: Optional[str] = None,
        remove_altloc: bool = True,
    ) -> str:
        """Clean protein structure for docking.
        
        Args:
            pdb_path: Input PDB file path.
            output_path: Output file path.
            remove_water: Remove water molecules (HOH, WAT).
            remove_ions: Remove common ions (NA, CL, MG, ZN, etc.).
            remove_ligands: Remove heteroatoms (except specified).
            keep_ligand: Ligand residue name to keep (e.g., "TCL").
            select_chain: Keep only specified chain.
            remove_altloc: Remove alternate locations (keep A only).
            
        Returns:
            Path to cleaned PDB file.
        """
        if output_path is None:
            stem = Path(pdb_path).stem
            output_path = str(self.work_dir / f"{stem}_clean.pdb")
        
        waters = {"HOH", "WAT", "H2O", "TIP", "TIP3"}
        ions = {"NA", "CL", "MG", "ZN", "CA", "K", "FE", "MN", "CO", "NI", "CU"}
        
        cleaned_lines = []
        
        with open(pdb_path, 'r') as f:
            for line in f:
                record_type = line[0:6].strip()
                
                # Skip non-coordinate records we don't need
                if record_type not in ("ATOM", "HETATM", "TER", "END"):
                    if record_type in ("HEADER", "TITLE", "COMPND", "REMARK"):
                        cleaned_lines.append(line)
                    continue
                
                # Process ATOM/HETATM records
                if record_type in ("ATOM", "HETATM"):
                    resname = line[17:20].strip()
                    chain = line[21].strip()
                    altloc = line[16].strip()
                    
                    # Chain selection
                    if select_chain and chain != select_chain:
                        continue
                    
                    # Alternate location - keep only A or empty
                    if remove_altloc and altloc and altloc != 'A':
                        continue
                    
                    # Water removal
                    if remove_water and resname in waters:
                        continue
                    
                    # Ion removal
                    if remove_ions and resname in ions:
                        continue
                    
                    # Ligand handling
                    if record_type == "HETATM" and remove_ligands:
                        if keep_ligand and resname == keep_ligand:
                            pass  # Keep this ligand
                        elif resname not in waters and resname not in ions:
                            continue  # Remove other ligands
                    
                    # Remove altloc indicator for cleaner output
                    if remove_altloc and altloc:
                        line = line[:16] + ' ' + line[17:]
                
                cleaned_lines.append(line)
        
        # Ensure END record
        if cleaned_lines and not cleaned_lines[-1].startswith("END"):
            cleaned_lines.append("END\n")
        
        with open(output_path, 'w') as f:
            f.writelines(cleaned_lines)
        
        logger.info(f"Cleaned protein saved: {output_path}")
        return output_path
    
    def get_binding_site(
        self,
        pdb_path: str,
        ligand_resname: str,
        padding: float = 5.0,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Extract binding site coordinates from co-crystallized ligand.
        
        Args:
            pdb_path: Path to PDB file.
            ligand_resname: Residue name of the ligand.
            padding: Extra space around ligand for docking box.
            
        Returns:
            Tuple of (center, size) for docking box.
        """
        coords = []
        
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    resname = line[17:20].strip()
                    if resname == ligand_resname:
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            coords.append([x, y, z])
                        except ValueError:
                            continue
        
        if not coords:
            raise ValueError(f"Ligand '{ligand_resname}' not found in {pdb_path}")
        
        coords = np.array(coords)
        
        # Calculate center as centroid
        center = tuple(coords.mean(axis=0).round(3))
        
        # Calculate size with padding
        extent = coords.max(axis=0) - coords.min(axis=0)
        size = tuple((extent + 2 * padding).round(3))
        
        logger.info(f"Binding site: center={center}, size={size}")
        
        return center, size
    
    def get_active_site_residues(
        self,
        pdb_path: str,
        ligand_resname: str,
        distance_cutoff: float = 5.0,
    ) -> List[Dict]:
        """Find protein residues within distance of ligand.
        
        Args:
            pdb_path: Path to PDB file with ligand.
            ligand_resname: Residue name of the ligand.
            distance_cutoff: Distance threshold in Angstroms.
            
        Returns:
            List of dictionaries with residue information.
        """
        ligand_atoms = []
        protein_atoms = []
        
        with open(pdb_path, 'r') as f:
            for line in f:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                
                try:
                    resname = line[17:20].strip()
                    resnum = int(line[22:26])
                    chain = line[21].strip()
                    atom_name = line[12:16].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    atom_info = {
                        "resname": resname,
                        "resnum": resnum,
                        "chain": chain,
                        "atom": atom_name,
                        "coords": np.array([x, y, z])
                    }
                    
                    if resname == ligand_resname:
                        ligand_atoms.append(atom_info)
                    elif line.startswith("ATOM"):
                        protein_atoms.append(atom_info)
                        
                except (ValueError, IndexError):
                    continue
        
        if not ligand_atoms:
            raise ValueError(f"Ligand '{ligand_resname}' not found")
        
        # Find nearby residues
        nearby_residues = set()
        
        for patom in protein_atoms:
            for latom in ligand_atoms:
                dist = np.linalg.norm(patom["coords"] - latom["coords"])
                if dist <= distance_cutoff:
                    nearby_residues.add((patom["chain"], patom["resnum"], patom["resname"]))
                    break
        
        result = [
            {"chain": chain, "resnum": resnum, "resname": resname}
            for chain, resnum, resname in sorted(nearby_residues)
        ]
        
        logger.info(f"Found {len(result)} residues within {distance_cutoff}Å of {ligand_resname}")
        
        return result
    
    def extract_chain(
        self,
        pdb_path: str,
        chain_id: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Extract a single chain from PDB file.
        
        Args:
            pdb_path: Input PDB file.
            chain_id: Chain identifier to extract.
            output_path: Output file path.
            
        Returns:
            Path to output PDB file.
        """
        if output_path is None:
            stem = Path(pdb_path).stem
            output_path = str(self.work_dir / f"{stem}_chain{chain_id}.pdb")
        
        extracted_lines = []
        
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    chain = line[21].strip()
                    if chain == chain_id:
                        extracted_lines.append(line)
                elif line.startswith(("HEADER", "TITLE", "END")):
                    extracted_lines.append(line)
        
        if not extracted_lines:
            raise ValueError(f"Chain '{chain_id}' not found in {pdb_path}")
        
        if not extracted_lines[-1].startswith("END"):
            extracted_lines.append("END\n")
        
        with open(output_path, 'w') as f:
            f.writelines(extracted_lines)
        
        logger.info(f"Extracted chain {chain_id}: {output_path}")
        
        return output_path
    
    def get_structure_info(self, pdb_path: str) -> Dict:
        """Get basic information about PDB structure.
        
        Args:
            pdb_path: Path to PDB file.
            
        Returns:
            Dictionary with structure information.
        """
        chains = set()
        residues = set()
        ligands = set()
        n_atoms = 0
        
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    chain = line[21].strip()
                    resname = line[17:20].strip()
                    resnum = line[22:26].strip()
                    chains.add(chain)
                    residues.add((chain, resnum, resname))
                    n_atoms += 1
                elif line.startswith("HETATM"):
                    resname = line[17:20].strip()
                    if resname not in {"HOH", "WAT", "H2O"}:
                        ligands.add(resname)
                    n_atoms += 1
        
        return {
            "n_atoms": n_atoms,
            "n_residues": len(residues),
            "chains": sorted(chains),
            "n_chains": len(chains),
            "ligands": sorted(ligands),
        }


# TB-specific protein preparation
TB_TARGETS = {
    "InhA": {
        "pdb_id": "4TZK",
        "ligand": "TCL",  # Triclosan
        "chain": "A",
        "description": "Enoyl-ACP reductase",
    },
    "KatG": {
        "pdb_id": "1SJ2", 
        "ligand": "INH",  # Isoniazid
        "chain": "A",
        "description": "Catalase-peroxidase",
    },
    "DprE1": {
        "pdb_id": "4FDO",
        "ligand": "BTZ",
        "chain": "A", 
        "description": "Decaprenylphosphoryl-beta-D-ribose oxidase",
    },
    "MmpL3": {
        "pdb_id": "6AJG",
        "ligand": "SQ109",
        "chain": "A",
        "description": "Mycolic acid transporter",
    },
}


def prepare_tb_target(
    target_name: str,
    output_dir: str = "data/structures",
) -> Dict:
    """Download and prepare a TB drug target for docking.
    
    Args:
        target_name: Name of the target (InhA, KatG, DprE1, MmpL3).
        output_dir: Directory to save prepared files.
        
    Returns:
        Dictionary with paths and binding site information.
    """
    if target_name not in TB_TARGETS:
        raise ValueError(f"Unknown target: {target_name}. Available: {list(TB_TARGETS.keys())}")
    
    target = TB_TARGETS[target_name]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prep = ProteinPreparator(str(output_dir))
    
    # Download
    pdb_path = prep.download_pdb(target["pdb_id"])
    
    # Get structure info
    info = prep.get_structure_info(pdb_path)
    logger.info(f"Structure info: {info}")
    
    # Clean (keep ligand for binding site)
    clean_path = prep.clean_protein(
        pdb_path,
        keep_ligand=target["ligand"],
        select_chain=target["chain"],
    )
    
    # Get binding site
    try:
        center, size = prep.get_binding_site(clean_path, target["ligand"])
    except ValueError:
        logger.warning(f"Ligand {target['ligand']} not found, using default box")
        center = (0, 0, 0)
        size = (30, 30, 30)
    
    # Clean version without ligand for docking
    receptor_path = prep.clean_protein(
        pdb_path,
        output_path=str(output_dir / f"{target_name}_receptor.pdb"),
        remove_ligands=True,
        select_chain=target["chain"],
    )
    
    return {
        "target": target_name,
        "pdb_id": target["pdb_id"],
        "description": target["description"],
        "receptor_pdb": receptor_path,
        "center": center,
        "size": size,
        "chain": target["chain"],
    }
