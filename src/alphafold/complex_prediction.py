"""Protein-ligand complex prediction and analysis.

This module provides specialized tools for predicting and analyzing
protein-ligand complexes using AlphaFold 3 and complementary methods.

Features:
- Complex structure prediction
- Binding pose analysis
- Interaction fingerprinting
- Comparison with docking results

Example:
    >>> predictor = ComplexPredictor(alphafold_client)
    >>> complex = predictor.predict(protein_seq, ligand_smiles)
    >>> interactions = predictor.analyze_interactions(complex)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

from .client import AlphaFoldClient, PredictionResult
from .structure_analysis import StructureAnalyzer, BindingSite


@dataclass
class LigandInteraction:
    """Represents an interaction between ligand and protein.
    
    Args:
        interaction_type: Type (hydrogen_bond, hydrophobic, pi_stacking, etc.)
        ligand_atom: Ligand atom index or name.
        protein_residue: Protein residue (name + number).
        protein_atom: Protein atom name.
        distance: Interaction distance in angstroms.
        strength: Estimated interaction strength (0-1).
    """
    interaction_type: str
    ligand_atom: str
    protein_residue: str
    protein_atom: str
    distance: float
    strength: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.interaction_type,
            "ligand_atom": self.ligand_atom,
            "protein_residue": self.protein_residue,
            "protein_atom": self.protein_atom,
            "distance": self.distance,
            "strength": self.strength,
        }


@dataclass
class ComplexAnalysis:
    """Analysis results for a protein-ligand complex.
    
    Args:
        ligand_smiles: Original ligand SMILES.
        binding_site: Identified binding site.
        interactions: List of protein-ligand interactions.
        binding_energy_estimate: Estimated binding energy (kcal/mol).
        ligand_efficiency: Ligand efficiency (LE).
        contact_residues: Residues in contact with ligand.
    """
    ligand_smiles: str
    binding_site: Optional[BindingSite]
    interactions: List[LigandInteraction]
    binding_energy_estimate: float = 0.0
    ligand_efficiency: float = 0.0
    contact_residues: List[str] = field(default_factory=list)
    
    def get_interaction_summary(self) -> Dict[str, int]:
        """Count interactions by type."""
        summary = {}
        for interaction in self.interactions:
            int_type = interaction.interaction_type
            summary[int_type] = summary.get(int_type, 0) + 1
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ligand_smiles": self.ligand_smiles,
            "binding_site": self.binding_site.to_dict() if self.binding_site else None,
            "interactions": [i.to_dict() for i in self.interactions],
            "binding_energy_estimate": self.binding_energy_estimate,
            "ligand_efficiency": self.ligand_efficiency,
            "contact_residues": self.contact_residues,
            "interaction_summary": self.get_interaction_summary(),
        }


class ComplexPredictor:
    """Predictor for protein-ligand complex structures.
    
    Combines AlphaFold 3 predictions with interaction analysis
    to provide comprehensive complex characterization.
    
    Args:
        client: AlphaFoldClient instance.
        
    Example:
        >>> client = AlphaFoldClient()
        >>> predictor = ComplexPredictor(client)
        >>> result = predictor.predict_and_analyze(
        ...     protein_sequence="MKFLILLFNILC...",
        ...     ligand_smiles="CC(=O)Oc1ccccc1C(=O)O"
        ... )
    """
    
    def __init__(self, client: Optional[AlphaFoldClient] = None):
        self.client = client or AlphaFoldClient()
        
        # Interaction distance cutoffs
        self.cutoffs = {
            "hydrogen_bond": 3.5,
            "salt_bridge": 4.0,
            "hydrophobic": 4.5,
            "pi_stacking": 5.0,
            "pi_cation": 6.0,
            "halogen_bond": 3.5,
        }
        
        logger.info("ComplexPredictor initialized")
    
    def predict(
        self,
        protein_sequence: str,
        ligand_smiles: str,
        name: str = "complex",
    ) -> PredictionResult:
        """Predict protein-ligand complex structure.
        
        Args:
            protein_sequence: Protein amino acid sequence.
            ligand_smiles: Ligand SMILES string.
            name: Name for the prediction job.
            
        Returns:
            PredictionResult with complex structure.
        """
        return self.client.predict_complex(protein_sequence, ligand_smiles, name)
    
    def predict_and_analyze(
        self,
        protein_sequence: str,
        ligand_smiles: str,
        name: str = "complex",
    ) -> Tuple[PredictionResult, ComplexAnalysis]:
        """Predict complex and perform full analysis.
        
        Args:
            protein_sequence: Protein sequence.
            ligand_smiles: Ligand SMILES.
            name: Job name.
            
        Returns:
            Tuple of (PredictionResult, ComplexAnalysis).
        """
        # Predict structure
        result = self.predict(protein_sequence, ligand_smiles, name)
        
        # Analyze complex
        analysis = self.analyze_complex(result, ligand_smiles)
        
        return result, analysis
    
    def analyze_complex(
        self,
        prediction: PredictionResult,
        ligand_smiles: str,
    ) -> ComplexAnalysis:
        """Analyze protein-ligand interactions in predicted complex.
        
        Args:
            prediction: AlphaFold prediction result.
            ligand_smiles: Original ligand SMILES.
            
        Returns:
            ComplexAnalysis with interaction details.
        """
        # Parse structure
        analyzer = StructureAnalyzer(prediction.pdb_string, prediction.plddt_scores)
        
        # Find binding site containing ligand
        binding_sites = analyzer.find_binding_sites()
        ligand_binding_site = binding_sites[0] if binding_sites else None
        
        # Identify interactions
        interactions = self._find_interactions(prediction.pdb_string, ligand_smiles)
        
        # Get contact residues
        contact_residues = self._get_contact_residues(prediction.pdb_string)
        
        # Estimate binding energy
        binding_energy = self._estimate_binding_energy(interactions, ligand_smiles)
        
        # Calculate ligand efficiency
        ligand_efficiency = self._calculate_ligand_efficiency(binding_energy, ligand_smiles)
        
        return ComplexAnalysis(
            ligand_smiles=ligand_smiles,
            binding_site=ligand_binding_site,
            interactions=interactions,
            binding_energy_estimate=binding_energy,
            ligand_efficiency=ligand_efficiency,
            contact_residues=contact_residues,
        )
    
    def _find_interactions(
        self,
        pdb_string: str,
        ligand_smiles: str,
    ) -> List[LigandInteraction]:
        """Identify protein-ligand interactions from structure."""
        interactions = []
        
        # Parse PDB to extract ligand and protein atoms
        ligand_atoms, protein_atoms = self._parse_complex_atoms(pdb_string)
        
        if not ligand_atoms or not protein_atoms:
            return interactions
        
        # Check each ligand-protein atom pair
        for lig_atom in ligand_atoms:
            for prot_atom in protein_atoms:
                interaction = self._check_interaction(lig_atom, prot_atom)
                if interaction is not None:
                    interactions.append(interaction)
        
        return interactions
    
    def _parse_complex_atoms(
        self,
        pdb_string: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Parse atoms from PDB string, separating ligand and protein."""
        ligand_atoms = []
        protein_atoms = []
        
        standard_residues = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
            "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
            "THR", "TRP", "TYR", "VAL"
        }
        
        for line in pdb_string.split("\n"):
            if not line.startswith("ATOM") and not line.startswith("HETATM"):
                continue
            
            try:
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain = line[21]
                res_num = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = line[76:78].strip() if len(line) > 76 else atom_name[0]
                
                atom_info = {
                    "name": atom_name,
                    "residue": res_name,
                    "chain": chain,
                    "res_num": res_num,
                    "coords": np.array([x, y, z]),
                    "element": element,
                }
                
                if res_name in standard_residues:
                    protein_atoms.append(atom_info)
                else:
                    ligand_atoms.append(atom_info)
                    
            except (ValueError, IndexError):
                continue
        
        return ligand_atoms, protein_atoms
    
    def _check_interaction(
        self,
        ligand_atom: Dict,
        protein_atom: Dict,
    ) -> Optional[LigandInteraction]:
        """Check if two atoms form an interaction."""
        distance = np.linalg.norm(ligand_atom["coords"] - protein_atom["coords"])
        
        lig_elem = ligand_atom["element"]
        prot_elem = protein_atom["element"]
        
        # Check hydrogen bond
        if distance < self.cutoffs["hydrogen_bond"]:
            if self._is_hbond_donor_acceptor(lig_elem, prot_elem):
                return LigandInteraction(
                    interaction_type="hydrogen_bond",
                    ligand_atom=ligand_atom["name"],
                    protein_residue=f"{protein_atom['residue']}{protein_atom['res_num']}",
                    protein_atom=protein_atom["name"],
                    distance=distance,
                    strength=self._hbond_strength(distance),
                )
        
        # Check hydrophobic
        if distance < self.cutoffs["hydrophobic"]:
            if self._is_hydrophobic(lig_elem, prot_elem):
                return LigandInteraction(
                    interaction_type="hydrophobic",
                    ligand_atom=ligand_atom["name"],
                    protein_residue=f"{protein_atom['residue']}{protein_atom['res_num']}",
                    protein_atom=protein_atom["name"],
                    distance=distance,
                    strength=self._hydrophobic_strength(distance),
                )
        
        # Check salt bridge
        if distance < self.cutoffs["salt_bridge"]:
            if self._is_salt_bridge(ligand_atom, protein_atom):
                return LigandInteraction(
                    interaction_type="salt_bridge",
                    ligand_atom=ligand_atom["name"],
                    protein_residue=f"{protein_atom['residue']}{protein_atom['res_num']}",
                    protein_atom=protein_atom["name"],
                    distance=distance,
                    strength=0.8,
                )
        
        return None
    
    def _is_hbond_donor_acceptor(self, elem1: str, elem2: str) -> bool:
        """Check if elements can form hydrogen bond."""
        donors_acceptors = {"N", "O", "S", "F"}
        return elem1 in donors_acceptors or elem2 in donors_acceptors
    
    def _is_hydrophobic(self, elem1: str, elem2: str) -> bool:
        """Check if elements can form hydrophobic contact."""
        return elem1 == "C" and elem2 == "C"
    
    def _is_salt_bridge(self, atom1: Dict, atom2: Dict) -> bool:
        """Check if atoms can form salt bridge."""
        charged_groups = {
            "positive": ["NZ", "NH1", "NH2", "NE"],  # Lys, Arg
            "negative": ["OD1", "OD2", "OE1", "OE2"],  # Asp, Glu
        }
        
        a1_positive = atom1["name"] in charged_groups["positive"]
        a1_negative = atom1["name"] in charged_groups["negative"]
        a2_positive = atom2["name"] in charged_groups["positive"]
        a2_negative = atom2["name"] in charged_groups["negative"]
        
        return (a1_positive and a2_negative) or (a1_negative and a2_positive)
    
    def _hbond_strength(self, distance: float) -> float:
        """Estimate hydrogen bond strength from distance."""
        if distance < 2.5:
            return 1.0
        elif distance < 3.0:
            return 0.8
        elif distance < 3.5:
            return 0.5
        return 0.2
    
    def _hydrophobic_strength(self, distance: float) -> float:
        """Estimate hydrophobic interaction strength."""
        if distance < 3.5:
            return 0.8
        elif distance < 4.0:
            return 0.5
        return 0.3
    
    def _get_contact_residues(self, pdb_string: str, cutoff: float = 4.0) -> List[str]:
        """Get protein residues in contact with ligand."""
        ligand_atoms, protein_atoms = self._parse_complex_atoms(pdb_string)
        
        contact_residues = set()
        
        for lig_atom in ligand_atoms:
            for prot_atom in protein_atoms:
                distance = np.linalg.norm(lig_atom["coords"] - prot_atom["coords"])
                if distance < cutoff:
                    res_id = f"{prot_atom['residue']}{prot_atom['res_num']}"
                    contact_residues.add(res_id)
        
        return sorted(list(contact_residues))
    
    def _estimate_binding_energy(
        self,
        interactions: List[LigandInteraction],
        ligand_smiles: str,
    ) -> float:
        """Estimate binding energy from interactions.
        
        Uses simplified empirical scoring function.
        """
        # Interaction energy contributions (kcal/mol)
        energy_contributions = {
            "hydrogen_bond": -2.5,
            "salt_bridge": -4.0,
            "hydrophobic": -0.5,
            "pi_stacking": -1.5,
            "pi_cation": -2.0,
            "halogen_bond": -1.5,
        }
        
        total_energy = 0.0
        
        for interaction in interactions:
            contribution = energy_contributions.get(interaction.interaction_type, 0.0)
            total_energy += contribution * interaction.strength
        
        # Entropy penalty (simplified)
        if HAS_RDKIT:
            try:
                mol = Chem.MolFromSmiles(ligand_smiles)
                if mol:
                    n_rotatable = Descriptors.NumRotatableBonds(mol)
                    total_energy += 0.5 * n_rotatable  # Penalty for flexibility
            except:
                pass
        
        return total_energy
    
    def _calculate_ligand_efficiency(
        self,
        binding_energy: float,
        ligand_smiles: str,
    ) -> float:
        """Calculate ligand efficiency (LE = -ΔG / N_heavy_atoms)."""
        if not HAS_RDKIT:
            return 0.0
        
        try:
            mol = Chem.MolFromSmiles(ligand_smiles)
            if mol is None:
                return 0.0
            
            n_heavy = mol.GetNumHeavyAtoms()
            if n_heavy == 0:
                return 0.0
            
            return -binding_energy / n_heavy
            
        except Exception:
            return 0.0
    
    def compare_with_docking(
        self,
        alphafold_prediction: PredictionResult,
        docking_pose_pdb: str,
        ligand_smiles: str,
    ) -> Dict[str, Any]:
        """Compare AlphaFold prediction with docking results.
        
        Args:
            alphafold_prediction: AlphaFold complex prediction.
            docking_pose_pdb: Docked pose in PDB format.
            ligand_smiles: Ligand SMILES.
            
        Returns:
            Comparison metrics dictionary.
        """
        # Analyze both structures
        af_analysis = self.analyze_complex(alphafold_prediction, ligand_smiles)
        
        # Create temporary prediction for docking pose
        docking_prediction = PredictionResult(
            job_id="docking",
            sequence="",
            pdb_string=docking_pose_pdb,
        )
        docking_analysis = self.analyze_complex(docking_prediction, ligand_smiles)
        
        # Compare contact residues
        af_contacts = set(af_analysis.contact_residues)
        dock_contacts = set(docking_analysis.contact_residues)
        
        shared_contacts = af_contacts & dock_contacts
        contact_overlap = len(shared_contacts) / max(len(af_contacts | dock_contacts), 1)
        
        # Compare interactions
        af_int_types = af_analysis.get_interaction_summary()
        dock_int_types = docking_analysis.get_interaction_summary()
        
        return {
            "contact_overlap": contact_overlap,
            "shared_contacts": list(shared_contacts),
            "alphafold_only_contacts": list(af_contacts - dock_contacts),
            "docking_only_contacts": list(dock_contacts - af_contacts),
            "alphafold_interactions": af_int_types,
            "docking_interactions": dock_int_types,
            "alphafold_binding_energy": af_analysis.binding_energy_estimate,
            "docking_binding_energy": docking_analysis.binding_energy_estimate,
        }
    
    def batch_predict(
        self,
        protein_sequence: str,
        ligand_smiles_list: List[str],
        max_concurrent: int = 5,
    ) -> List[Tuple[str, PredictionResult, ComplexAnalysis]]:
        """Predict complexes for multiple ligands.
        
        Args:
            protein_sequence: Protein sequence.
            ligand_smiles_list: List of ligand SMILES.
            max_concurrent: Maximum concurrent jobs.
            
        Returns:
            List of (smiles, prediction, analysis) tuples.
        """
        results = []
        
        for i, smiles in enumerate(ligand_smiles_list):
            logger.info(f"Processing ligand {i + 1}/{len(ligand_smiles_list)}")
            
            try:
                prediction, analysis = self.predict_and_analyze(
                    protein_sequence,
                    smiles,
                    name=f"complex_{i}",
                )
                results.append((smiles, prediction, analysis))
                
            except Exception as e:
                logger.error(f"Failed to process {smiles}: {e}")
                continue
        
        return results
    
    def rank_ligands(
        self,
        results: List[Tuple[str, PredictionResult, ComplexAnalysis]],
    ) -> List[Tuple[str, float, ComplexAnalysis]]:
        """Rank ligands by predicted binding affinity.
        
        Args:
            results: Output from batch_predict.
            
        Returns:
            Sorted list of (smiles, score, analysis) tuples.
        """
        scored = []
        
        for smiles, prediction, analysis in results:
            # Combined score: binding energy + interaction count
            n_interactions = len(analysis.interactions)
            score = analysis.binding_energy_estimate - 0.5 * n_interactions
            scored.append((smiles, score, analysis))
        
        # Sort by score (lower is better)
        scored.sort(key=lambda x: x[1])
        
        return scored
    
    def generate_report(
        self,
        results: List[Tuple[str, PredictionResult, ComplexAnalysis]],
        output_path: str,
    ) -> None:
        """Generate comprehensive analysis report.
        
        Args:
            results: Prediction results.
            output_path: Path to save report.
        """
        ranked = self.rank_ligands(results)
        
        lines = []
        lines.append("=" * 70)
        lines.append("PROTEIN-LIGAND COMPLEX PREDICTION REPORT")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Total Ligands Analyzed: {len(results)}")
        lines.append("")
        
        lines.append("TOP RANKED COMPOUNDS")
        lines.append("-" * 70)
        
        for rank, (smiles, score, analysis) in enumerate(ranked[:10], 1):
            lines.append(f"\nRank {rank}:")
            lines.append(f"  SMILES: {smiles[:60]}...")
            lines.append(f"  Binding Energy: {analysis.binding_energy_estimate:.2f} kcal/mol")
            lines.append(f"  Ligand Efficiency: {analysis.ligand_efficiency:.3f}")
            lines.append(f"  Total Interactions: {len(analysis.interactions)}")
            lines.append(f"  Interaction Types: {analysis.get_interaction_summary()}")
            lines.append(f"  Contact Residues: {', '.join(analysis.contact_residues[:10])}...")
        
        lines.append("")
        lines.append("=" * 70)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        
        logger.info(f"Report saved to {output_path}")
