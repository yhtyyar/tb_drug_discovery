"""Unit tests for molecular docking modules.

Tests cover:
- VinaDocker initialization and configuration
- ProteinPreparator functionality
- Binding site calculations
- Result parsing
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.docking.protein_prep import ProteinPreparator, TB_TARGETS, prepare_tb_target
from src.docking.vina_docker import DockingResult, VinaDocker


class TestDockingResult:
    """Tests for DockingResult dataclass."""
    
    def test_docking_result_creation(self):
        """Test creating a DockingResult."""
        result = DockingResult(
            ligand_name="test_ligand",
            smiles="CCO",
            affinity=-7.5,
            num_poses=5,
            all_affinities=[-7.5, -7.2, -6.8, -6.5, -6.0]
        )
        
        assert result.ligand_name == "test_ligand"
        assert result.smiles == "CCO"
        assert result.affinity == -7.5
        assert result.num_poses == 5
        assert len(result.all_affinities) == 5
    
    def test_docking_result_to_dict(self):
        """Test converting DockingResult to dictionary."""
        result = DockingResult(
            ligand_name="ethanol",
            smiles="CCO",
            affinity=-5.0,
        )
        
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert d["ligand_name"] == "ethanol"
        assert d["smiles"] == "CCO"
        assert d["affinity"] == -5.0


class TestVinaDocker:
    """Tests for VinaDocker class."""
    
    def test_vina_docker_initialization(self):
        """Test VinaDocker initialization."""
        docker = VinaDocker(
            vina_path="/usr/bin/vina",
            exhaustiveness=16,
            num_modes=5,
        )
        
        assert docker.vina_path == "/usr/bin/vina"
        assert docker.exhaustiveness == 16
        assert docker.num_modes == 5
        assert docker.receptor_path is None
    
    def test_set_receptor(self):
        """Test setting receptor."""
        docker = VinaDocker()
        
        # Create dummy receptor file
        with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as f:
            f.write(b"ATOM  1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n")
            receptor_path = f.name
        
        try:
            docker.set_receptor(
                receptor_path,
                center=(10.0, 20.0, 30.0),
                size=(25.0, 25.0, 25.0)
            )
            
            assert docker.receptor_path == receptor_path
            assert docker.center == (10.0, 20.0, 30.0)
            assert docker.size == (25.0, 25.0, 25.0)
        finally:
            os.unlink(receptor_path)
    
    def test_set_receptor_file_not_found(self):
        """Test error when receptor file not found."""
        docker = VinaDocker()
        
        with pytest.raises(FileNotFoundError):
            docker.set_receptor(
                "nonexistent.pdbqt",
                center=(0, 0, 0)
            )
    
    def test_check_dependencies(self):
        """Test dependency checking."""
        docker = VinaDocker()
        status = docker.check_dependencies()
        
        assert isinstance(status, dict)
        assert "vina" in status
        assert "obabel" in status
        assert "rdkit" in status
    
    def test_parse_vina_output(self):
        """Test parsing Vina log file."""
        docker = VinaDocker()
        
        # Create mock log file
        log_content = """
AutoDock Vina v1.2.3

mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1         -7.5      0.000      0.000
   2         -7.2      1.234      2.345
   3         -6.8      2.456      3.567
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".txt", delete=False) as f:
            f.write(log_content)
            log_path = f.name
        
        try:
            affinities = docker._parse_vina_output(log_path)
            
            assert len(affinities) == 3
            assert affinities[0] == -7.5
            assert affinities[1] == -7.2
            assert affinities[2] == -6.8
        finally:
            os.unlink(log_path)
    
    def test_cleanup(self):
        """Test temp directory cleanup."""
        docker = VinaDocker()
        temp_dir = docker._temp_dir
        
        assert os.path.exists(temp_dir)
        
        docker.cleanup()
        
        assert not os.path.exists(temp_dir)


class TestProteinPreparator:
    """Tests for ProteinPreparator class."""
    
    def test_protein_preparator_initialization(self):
        """Test ProteinPreparator initialization."""
        prep = ProteinPreparator()
        
        assert prep.work_dir.exists()
    
    def test_protein_preparator_custom_dir(self):
        """Test initialization with custom directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_dir = os.path.join(tmpdir, "protein_work")
            prep = ProteinPreparator(work_dir=custom_dir)
            
            assert prep.work_dir == Path(custom_dir)
            assert prep.work_dir.exists()
    
    def test_clean_protein(self):
        """Test protein cleaning."""
        prep = ProteinPreparator()
        
        # Create test PDB content
        pdb_content = """HEADER    TEST PROTEIN
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
HETATM  100  O   HOH A 101      10.000  10.000  10.000  1.00  0.00           O
HETATM  101  NA  NA  A 102      15.000  15.000  15.000  1.00  0.00          NA
HETATM  200  C1  LIG A 200      20.000  20.000  20.000  1.00  0.00           C
END
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".pdb", delete=False) as f:
            f.write(pdb_content)
            pdb_path = f.name
        
        try:
            clean_path = prep.clean_protein(
                pdb_path,
                remove_water=True,
                remove_ions=True,
                remove_ligands=True
            )
            
            # Check cleaned file
            with open(clean_path, 'r') as f:
                content = f.read()
            
            assert "ALA" in content  # Protein kept
            assert "HOH" not in content  # Water removed
            assert " NA " not in content  # Ion removed
            assert "LIG" not in content  # Ligand removed
            
        finally:
            os.unlink(pdb_path)
    
    def test_clean_protein_keep_ligand(self):
        """Test keeping specific ligand during cleaning."""
        prep = ProteinPreparator()
        
        pdb_content = """HEADER    TEST
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
HETATM  100  O   HOH A 101      10.000  10.000  10.000  1.00  0.00           O
HETATM  200  C1  TCL A 200      20.000  20.000  20.000  1.00  0.00           C
END
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".pdb", delete=False) as f:
            f.write(pdb_content)
            pdb_path = f.name
        
        try:
            clean_path = prep.clean_protein(
                pdb_path,
                remove_water=True,
                remove_ligands=True,
                keep_ligand="TCL"
            )
            
            with open(clean_path, 'r') as f:
                content = f.read()
            
            assert "TCL" in content  # Kept ligand
            assert "HOH" not in content  # Water removed
            
        finally:
            os.unlink(pdb_path)
    
    def test_get_binding_site(self):
        """Test binding site extraction."""
        prep = ProteinPreparator()
        
        # Create PDB with ligand
        pdb_content = """HETATM    1  C1  LIG A   1      10.000  20.000  30.000  1.00  0.00           C
HETATM    2  C2  LIG A   1      12.000  22.000  32.000  1.00  0.00           C
HETATM    3  C3  LIG A   1      14.000  24.000  34.000  1.00  0.00           C
END
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".pdb", delete=False) as f:
            f.write(pdb_content)
            pdb_path = f.name
        
        try:
            center, size = prep.get_binding_site(pdb_path, "LIG", padding=5.0)
            
            # Center should be centroid
            assert abs(center[0] - 12.0) < 0.1
            assert abs(center[1] - 22.0) < 0.1
            assert abs(center[2] - 32.0) < 0.1
            
            # Size should be extent + 2*padding
            assert size[0] >= 10.0  # 4 + 10 padding
            
        finally:
            os.unlink(pdb_path)
    
    def test_get_binding_site_ligand_not_found(self):
        """Test error when ligand not found."""
        prep = ProteinPreparator()
        
        pdb_content = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
END
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".pdb", delete=False) as f:
            f.write(pdb_content)
            pdb_path = f.name
        
        try:
            with pytest.raises(ValueError, match="not found"):
                prep.get_binding_site(pdb_path, "LIG")
        finally:
            os.unlink(pdb_path)
    
    def test_get_structure_info(self):
        """Test structure information extraction."""
        prep = ProteinPreparator()
        
        pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  N   GLY A   2       3.000   0.000   0.000  1.00  0.00           N
ATOM      4  CA  GLY B   1       5.000   0.000   0.000  1.00  0.00           C
HETATM    5  C1  LIG A 100      10.000  10.000  10.000  1.00  0.00           C
END
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".pdb", delete=False) as f:
            f.write(pdb_content)
            pdb_path = f.name
        
        try:
            info = prep.get_structure_info(pdb_path)
            
            assert info["n_atoms"] == 5
            assert info["n_chains"] == 2
            assert "A" in info["chains"]
            assert "B" in info["chains"]
            assert "LIG" in info["ligands"]
            
        finally:
            os.unlink(pdb_path)
    
    def test_extract_chain(self):
        """Test chain extraction."""
        prep = ProteinPreparator()
        
        pdb_content = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  ALA B   1       5.000   0.000   0.000  1.00  0.00           C
END
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".pdb", delete=False) as f:
            f.write(pdb_content)
            pdb_path = f.name
        
        try:
            chain_a_path = prep.extract_chain(pdb_path, "A")
            
            with open(chain_a_path, 'r') as f:
                content = f.read()
            
            # Should only have chain A
            lines = [l for l in content.split('\n') if l.startswith("ATOM")]
            assert len(lines) == 1
            assert "0.000   0.000   0.000" in lines[0]
            
        finally:
            os.unlink(pdb_path)


class TestTBTargets:
    """Tests for TB-specific functionality."""
    
    def test_tb_targets_defined(self):
        """Test that TB targets are properly defined."""
        assert "InhA" in TB_TARGETS
        assert "KatG" in TB_TARGETS
        assert "DprE1" in TB_TARGETS
        assert "MmpL3" in TB_TARGETS
    
    def test_tb_target_structure(self):
        """Test TB target dictionary structure."""
        for name, target in TB_TARGETS.items():
            assert "pdb_id" in target
            assert "ligand" in target
            assert "chain" in target
            assert "description" in target
            
            # PDB ID should be 4 characters
            assert len(target["pdb_id"]) == 4


class TestIntegration:
    """Integration tests (require external tools)."""
    
    @pytest.mark.skip(reason="Requires Vina and OpenBabel installed")
    def test_full_docking_workflow(self):
        """Test complete docking workflow."""
        pass
    
    @pytest.mark.skip(reason="Requires internet connection")
    def test_pdb_download(self):
        """Test PDB download from RCSB."""
        prep = ProteinPreparator()
        pdb_path = prep.download_pdb("4TZK")
        
        assert Path(pdb_path).exists()
        assert Path(pdb_path).stat().st_size > 0
