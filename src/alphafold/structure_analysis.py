"""Structure analysis utilities for AlphaFold predictions.

This module provides tools for analyzing predicted protein structures,
including binding site detection, structural comparison, and quality assessment.

Example:
    >>> analyzer = StructureAnalyzer(prediction_result)
    >>> binding_sites = analyzer.find_binding_sites()
    >>> rmsd = analyzer.compare_structures(reference_pdb)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

try:
    from Bio.PDB import PDBParser, Superimposer, NeighborSearch
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False


@dataclass
class BindingSite:
    """Represents a potential binding site in a protein structure.
    
    Args:
        center: 3D coordinates of site center.
        residues: List of residue indices in the site.
        residue_names: List of residue names (3-letter code).
        volume: Estimated volume in cubic angstroms.
        druggability_score: Predicted druggability (0-1).
        pocket_type: Type of pocket (catalytic, allosteric, etc.).
    """
    center: Tuple[float, float, float]
    residues: List[int]
    residue_names: List[str]
    volume: float = 0.0
    druggability_score: float = 0.0
    pocket_type: str = "unknown"
    
    def get_residue_string(self) -> str:
        """Get comma-separated residue list."""
        return ", ".join(f"{name}{idx}" for name, idx in zip(self.residue_names, self.residues))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "center": list(self.center),
            "residues": self.residues,
            "residue_names": self.residue_names,
            "volume": self.volume,
            "druggability_score": self.druggability_score,
            "pocket_type": self.pocket_type,
        }


@dataclass
class StructureQuality:
    """Quality metrics for a predicted structure.
    
    Args:
        mean_plddt: Mean pLDDT score across all residues.
        low_confidence_regions: Residue ranges with low confidence.
        clash_score: Steric clash score.
        ramachandran_outliers: Percentage of Ramachandran outliers.
    """
    mean_plddt: float
    low_confidence_regions: List[Tuple[int, int]]
    clash_score: float = 0.0
    ramachandran_outliers: float = 0.0
    
    def is_high_quality(self, plddt_threshold: float = 70.0) -> bool:
        """Check if structure meets quality threshold."""
        return self.mean_plddt >= plddt_threshold


class StructureAnalyzer:
    """Analyzer for protein structures from AlphaFold predictions.
    
    Provides methods for:
    - Binding site detection
    - Structure comparison (RMSD)
    - Quality assessment
    - Contact analysis
    
    Args:
        pdb_string: PDB format string of the structure.
        plddt_scores: Optional per-residue confidence scores.
        
    Example:
        >>> analyzer = StructureAnalyzer(pdb_string, plddt_scores)
        >>> sites = analyzer.find_binding_sites()
        >>> quality = analyzer.assess_quality()
    """
    
    def __init__(
        self,
        pdb_string: str,
        plddt_scores: Optional[List[float]] = None,
    ):
        self.pdb_string = pdb_string
        self.plddt_scores = plddt_scores or []
        
        self.structure = None
        if HAS_BIOPYTHON:
            self.structure = self._parse_pdb(pdb_string)
        
        self._atoms = None
        self._residues = None
    
    def _parse_pdb(self, pdb_string: str) -> Optional[Structure]:
        """Parse PDB string into structure object."""
        import io
        parser = PDBParser(QUIET=True)
        
        try:
            structure = parser.get_structure("protein", io.StringIO(pdb_string))
            return structure
        except Exception as e:
            logger.error(f"Failed to parse PDB: {e}")
            return None
    
    @property
    def atoms(self) -> List[Any]:
        """Get all atoms in structure."""
        if self._atoms is None and self.structure is not None:
            self._atoms = list(self.structure.get_atoms())
        return self._atoms or []
    
    @property
    def residues(self) -> List[Any]:
        """Get all residues in structure."""
        if self._residues is None and self.structure is not None:
            self._residues = list(self.structure.get_residues())
        return self._residues or []
    
    def find_binding_sites(
        self,
        min_size: int = 10,
        probe_radius: float = 1.4,
        grid_spacing: float = 1.0,
    ) -> List[BindingSite]:
        """Find potential binding sites in the structure.
        
        Uses a grid-based algorithm to identify cavities and pockets
        that could serve as ligand binding sites.
        
        Args:
            min_size: Minimum number of residues in a binding site.
            probe_radius: Radius of probe sphere for cavity detection.
            grid_spacing: Grid spacing for cavity search.
            
        Returns:
            List of identified BindingSite objects.
        """
        if not HAS_BIOPYTHON or self.structure is None:
            logger.warning("BioPython required for binding site detection")
            return []
        
        binding_sites = []
        
        # Get CA atoms for analysis
        ca_atoms = [a for a in self.atoms if a.get_name() == "CA"]
        if not ca_atoms:
            return []
        
        # Get coordinates
        coords = np.array([a.get_coord() for a in ca_atoms])
        
        # Find cavities using alpha-sphere approach (simplified)
        # In practice, use fpocket or similar tool
        sites = self._detect_cavities(coords, probe_radius, grid_spacing)
        
        for site_coords, site_residues in sites:
            if len(site_residues) >= min_size:
                # Get residue names
                residue_names = []
                for idx in site_residues:
                    if idx < len(self.residues):
                        res = self.residues[idx]
                        residue_names.append(res.get_resname())
                
                # Calculate center
                center = tuple(np.mean(site_coords, axis=0))
                
                # Estimate volume (simplified)
                volume = len(site_residues) * 150  # rough estimate
                
                # Druggability score (placeholder)
                druggability = self._estimate_druggability(site_residues)
                
                binding_site = BindingSite(
                    center=center,
                    residues=list(site_residues),
                    residue_names=residue_names,
                    volume=volume,
                    druggability_score=druggability,
                )
                binding_sites.append(binding_site)
        
        # Sort by druggability score
        binding_sites.sort(key=lambda x: x.druggability_score, reverse=True)
        
        logger.info(f"Found {len(binding_sites)} potential binding sites")
        return binding_sites
    
    def _detect_cavities(
        self,
        coords: np.ndarray,
        probe_radius: float,
        grid_spacing: float,
    ) -> List[Tuple[np.ndarray, List[int]]]:
        """Detect cavities using simplified grid-based approach."""
        # Find bounding box
        min_coords = coords.min(axis=0) - 5
        max_coords = coords.max(axis=0) + 5
        
        # Create grid
        grid_points = []
        x = min_coords[0]
        while x < max_coords[0]:
            y = min_coords[1]
            while y < max_coords[1]:
                z = min_coords[2]
                while z < max_coords[2]:
                    grid_points.append([x, y, z])
                    z += grid_spacing
                y += grid_spacing
            x += grid_spacing
        
        grid_points = np.array(grid_points)
        
        if len(grid_points) == 0:
            return []
        
        # Find grid points inside protein but not too close to atoms
        from scipy.spatial.distance import cdist
        distances = cdist(grid_points, coords)
        min_distances = distances.min(axis=1)
        
        # Points in cavity: between probe_radius and 2*probe_radius from any atom
        cavity_mask = (min_distances > probe_radius) & (min_distances < 6.0)
        cavity_points = grid_points[cavity_mask]
        
        if len(cavity_points) == 0:
            return []
        
        # Cluster cavity points
        from scipy.cluster.hierarchy import fcluster, linkage
        
        if len(cavity_points) > 1:
            Z = linkage(cavity_points, method='average')
            clusters = fcluster(Z, t=5.0, criterion='distance')
        else:
            clusters = np.array([1])
        
        # Group by cluster
        sites = []
        for cluster_id in np.unique(clusters):
            cluster_points = cavity_points[clusters == cluster_id]
            
            # Find nearby residues
            cluster_center = cluster_points.mean(axis=0)
            dist_to_center = np.linalg.norm(coords - cluster_center, axis=1)
            nearby_residues = np.where(dist_to_center < 8.0)[0].tolist()
            
            if len(nearby_residues) >= 5:
                sites.append((cluster_points, nearby_residues))
        
        return sites
    
    def _estimate_druggability(self, residue_indices: List[int]) -> float:
        """Estimate druggability score based on residue composition."""
        if not self.residues:
            return 0.5
        
        # Hydrophobic residues are favorable for druggability
        hydrophobic = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO"}
        aromatic = {"PHE", "TYR", "TRP", "HIS"}
        
        hydrophobic_count = 0
        aromatic_count = 0
        
        for idx in residue_indices:
            if idx < len(self.residues):
                res_name = self.residues[idx].get_resname()
                if res_name in hydrophobic:
                    hydrophobic_count += 1
                if res_name in aromatic:
                    aromatic_count += 1
        
        n_residues = len(residue_indices)
        if n_residues == 0:
            return 0.0
        
        # Simple druggability estimate
        hydrophobic_ratio = hydrophobic_count / n_residues
        aromatic_ratio = aromatic_count / n_residues
        
        score = 0.4 * hydrophobic_ratio + 0.3 * aromatic_ratio + 0.3 * min(1.0, n_residues / 20)
        
        return min(1.0, max(0.0, score))
    
    def compare_structures(
        self,
        reference_pdb: str,
        atom_selection: str = "CA",
    ) -> float:
        """Calculate RMSD between this structure and a reference.
        
        Args:
            reference_pdb: PDB string of reference structure.
            atom_selection: Atom type for comparison ('CA' for alpha carbons).
            
        Returns:
            RMSD value in angstroms.
        """
        if not HAS_BIOPYTHON:
            logger.warning("BioPython required for RMSD calculation")
            return float('inf')
        
        ref_structure = self._parse_pdb(reference_pdb)
        if ref_structure is None or self.structure is None:
            return float('inf')
        
        # Get atoms for superposition
        ref_atoms = [a for a in ref_structure.get_atoms() if a.get_name() == atom_selection]
        mobile_atoms = [a for a in self.structure.get_atoms() if a.get_name() == atom_selection]
        
        # Match atoms by residue number
        min_len = min(len(ref_atoms), len(mobile_atoms))
        ref_atoms = ref_atoms[:min_len]
        mobile_atoms = mobile_atoms[:min_len]
        
        if min_len == 0:
            return float('inf')
        
        # Superimpose
        super_imposer = Superimposer()
        super_imposer.set_atoms(ref_atoms, mobile_atoms)
        
        return super_imposer.rms
    
    def assess_quality(self) -> StructureQuality:
        """Assess overall quality of the predicted structure.
        
        Returns:
            StructureQuality object with quality metrics.
        """
        # Mean pLDDT
        mean_plddt = 0.0
        if self.plddt_scores:
            mean_plddt = sum(self.plddt_scores) / len(self.plddt_scores)
        
        # Find low confidence regions
        low_confidence_regions = []
        if self.plddt_scores:
            in_low_region = False
            region_start = 0
            
            for i, score in enumerate(self.plddt_scores):
                if score < 50.0:
                    if not in_low_region:
                        region_start = i
                        in_low_region = True
                else:
                    if in_low_region:
                        low_confidence_regions.append((region_start, i - 1))
                        in_low_region = False
            
            if in_low_region:
                low_confidence_regions.append((region_start, len(self.plddt_scores) - 1))
        
        return StructureQuality(
            mean_plddt=mean_plddt,
            low_confidence_regions=low_confidence_regions,
        )
    
    def get_contact_map(self, distance_cutoff: float = 8.0) -> np.ndarray:
        """Calculate residue contact map.
        
        Args:
            distance_cutoff: Maximum distance for contact.
            
        Returns:
            Binary contact matrix (N x N).
        """
        if not self.residues:
            return np.array([])
        
        n_residues = len(self.residues)
        contact_map = np.zeros((n_residues, n_residues))
        
        # Get CA coordinates
        ca_coords = []
        for res in self.residues:
            for atom in res.get_atoms():
                if atom.get_name() == "CA":
                    ca_coords.append(atom.get_coord())
                    break
            else:
                # Use first atom if no CA
                atoms = list(res.get_atoms())
                if atoms:
                    ca_coords.append(atoms[0].get_coord())
        
        if len(ca_coords) != n_residues:
            return np.array([])
        
        ca_coords = np.array(ca_coords)
        
        # Calculate pairwise distances
        for i in range(n_residues):
            for j in range(i + 1, n_residues):
                dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
                if dist < distance_cutoff:
                    contact_map[i, j] = 1
                    contact_map[j, i] = 1
        
        return contact_map
    
    def get_secondary_structure(self) -> Dict[str, List[Tuple[int, int]]]:
        """Identify secondary structure elements.
        
        Returns:
            Dictionary with 'helix', 'sheet', 'coil' regions.
        """
        # Simplified secondary structure assignment
        # In practice, use DSSP
        
        if not self.residues:
            return {"helix": [], "sheet": [], "coil": []}
        
        n_residues = len(self.residues)
        
        # Parse HELIX and SHEET records from PDB
        helices = []
        sheets = []
        
        for line in self.pdb_string.split("\n"):
            if line.startswith("HELIX"):
                try:
                    start = int(line[21:25].strip())
                    end = int(line[33:37].strip())
                    helices.append((start, end))
                except (ValueError, IndexError):
                    pass
            elif line.startswith("SHEET"):
                try:
                    start = int(line[22:26].strip())
                    end = int(line[33:37].strip())
                    sheets.append((start, end))
                except (ValueError, IndexError):
                    pass
        
        # Identify coil regions
        assigned = set()
        for start, end in helices + sheets:
            for i in range(start, end + 1):
                assigned.add(i)
        
        coils = []
        in_coil = False
        coil_start = 0
        
        for i in range(1, n_residues + 1):
            if i not in assigned:
                if not in_coil:
                    coil_start = i
                    in_coil = True
            else:
                if in_coil:
                    coils.append((coil_start, i - 1))
                    in_coil = False
        
        if in_coil:
            coils.append((coil_start, n_residues))
        
        return {
            "helix": helices,
            "sheet": sheets,
            "coil": coils,
        }
    
    def save_analysis_report(self, output_path: str) -> None:
        """Save comprehensive analysis report.
        
        Args:
            output_path: Path to save report.
        """
        quality = self.assess_quality()
        binding_sites = self.find_binding_sites()
        secondary = self.get_secondary_structure()
        
        report = []
        report.append("=" * 60)
        report.append("STRUCTURE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Quality metrics
        report.append("QUALITY METRICS")
        report.append("-" * 40)
        report.append(f"Mean pLDDT Score: {quality.mean_plddt:.1f}")
        report.append(f"High Quality: {'Yes' if quality.is_high_quality() else 'No'}")
        report.append(f"Low Confidence Regions: {len(quality.low_confidence_regions)}")
        for start, end in quality.low_confidence_regions:
            report.append(f"  - Residues {start}-{end}")
        report.append("")
        
        # Secondary structure
        report.append("SECONDARY STRUCTURE")
        report.append("-" * 40)
        report.append(f"Helices: {len(secondary['helix'])}")
        report.append(f"Sheets: {len(secondary['sheet'])}")
        report.append(f"Coils: {len(secondary['coil'])}")
        report.append("")
        
        # Binding sites
        report.append("BINDING SITES")
        report.append("-" * 40)
        report.append(f"Total Sites Found: {len(binding_sites)}")
        for i, site in enumerate(binding_sites, 1):
            report.append(f"\nSite {i}:")
            report.append(f"  Center: ({site.center[0]:.1f}, {site.center[1]:.1f}, {site.center[2]:.1f})")
            report.append(f"  Residues: {len(site.residues)}")
            report.append(f"  Volume: {site.volume:.1f} Å³")
            report.append(f"  Druggability: {site.druggability_score:.2f}")
            report.append(f"  Key Residues: {site.get_residue_string()[:50]}...")
        
        report.append("")
        report.append("=" * 60)
        
        # Write report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(report))
        
        logger.info(f"Analysis report saved to {output_path}")
