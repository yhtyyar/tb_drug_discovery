"""Tests for AlphaFold integration module.

Tests cover:
- AlphaFoldClient
- StructureAnalyzer
- ComplexPredictor
- Binding site detection
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.alphafold.client import (
    AlphaFoldConfig,
    AlphaFoldClient,
    PredictionResult,
)
from src.alphafold.structure_analysis import (
    StructureAnalyzer,
    BindingSite,
    StructureQuality,
)
from src.alphafold.complex_prediction import (
    ComplexPredictor,
    LigandInteraction,
    ComplexAnalysis,
)


# Sample PDB string for testing
SAMPLE_PDB = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 50.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 50.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 50.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00 50.00           O
ATOM      5  CB  ALA A   1       1.986  -0.760  -1.217  1.00 50.00           C
ATOM      6  N   GLY A   2       3.310   1.530   0.000  1.00 50.00           N
ATOM      7  CA  GLY A   2       3.970   2.830   0.000  1.00 50.00           C
ATOM      8  C   GLY A   2       5.470   2.710   0.000  1.00 50.00           C
ATOM      9  O   GLY A   2       6.030   1.610   0.000  1.00 50.00           O
ATOM     10  N   LEU A   3       6.110   3.870   0.000  1.00 50.00           N
ATOM     11  CA  LEU A   3       7.560   3.980   0.000  1.00 50.00           C
ATOM     12  C   LEU A   3       8.110   5.400   0.000  1.00 50.00           C
ATOM     13  O   LEU A   3       7.350   6.370   0.000  1.00 50.00           O
END
"""


class TestAlphaFoldConfig:
    """Tests for AlphaFoldConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AlphaFoldConfig()
        
        assert config.server_url == "https://alphafoldserver.com/api"
        assert config.cache_dir == "data/alphafold_cache"
        assert config.timeout == 300
        assert config.use_cache is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AlphaFoldConfig(
            server_url="http://localhost:8000",
            cache_dir="/tmp/af_cache",
            timeout=600,
        )
        
        assert config.server_url == "http://localhost:8000"
        assert config.cache_dir == "/tmp/af_cache"
        assert config.timeout == 600


class TestPredictionResult:
    """Tests for PredictionResult dataclass."""
    
    def test_create_result(self):
        """Test creating prediction result."""
        result = PredictionResult(
            job_id="test_123",
            sequence="AGLV",
            pdb_string=SAMPLE_PDB,
            plddt_scores=[80.0, 75.0, 90.0, 85.0],
        )
        
        assert result.job_id == "test_123"
        assert result.sequence == "AGLV"
        assert len(result.plddt_scores) == 4
    
    def test_mean_plddt(self):
        """Test mean pLDDT calculation."""
        result = PredictionResult(
            job_id="test",
            sequence="AGLV",
            pdb_string="",
            plddt_scores=[80.0, 70.0, 90.0, 60.0],
        )
        
        assert result.get_mean_plddt() == 75.0
    
    def test_confident_residues(self):
        """Test getting confident residues."""
        result = PredictionResult(
            job_id="test",
            sequence="AGLV",
            pdb_string="",
            plddt_scores=[80.0, 50.0, 90.0, 60.0],
        )
        
        confident = result.get_confident_residues(threshold=70.0)
        
        assert 0 in confident  # 80.0
        assert 2 in confident  # 90.0
        assert 1 not in confident  # 50.0
    
    def test_to_dict(self):
        """Test serialization to dict."""
        result = PredictionResult(
            job_id="test",
            sequence="AGLV",
            pdb_string=SAMPLE_PDB,
        )
        
        data = result.to_dict()
        
        assert data["job_id"] == "test"
        assert data["sequence"] == "AGLV"
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "job_id": "test",
            "sequence": "AGLV",
            "pdb_string": SAMPLE_PDB,
            "plddt_scores": [80.0],
            "pae_matrix": None,
            "metadata": {},
        }
        
        result = PredictionResult.from_dict(data)
        
        assert result.job_id == "test"
        assert result.sequence == "AGLV"
    
    def test_save_pdb(self, tmp_path):
        """Test saving PDB file."""
        result = PredictionResult(
            job_id="test",
            sequence="AGLV",
            pdb_string=SAMPLE_PDB,
        )
        
        pdb_path = str(tmp_path / "test.pdb")
        result.save_pdb(pdb_path)
        
        assert Path(pdb_path).exists()
        with open(pdb_path) as f:
            content = f.read()
        assert "ATOM" in content


class TestAlphaFoldClient:
    """Tests for AlphaFoldClient."""
    
    def test_init(self, tmp_path):
        """Test client initialization."""
        config = AlphaFoldConfig(cache_dir=str(tmp_path / "cache"))
        client = AlphaFoldClient(config)
        
        assert client.cache_dir.exists()
    
    def test_clean_sequence(self, tmp_path):
        """Test sequence cleaning."""
        config = AlphaFoldConfig(cache_dir=str(tmp_path / "cache"))
        client = AlphaFoldClient(config)
        
        # Test whitespace removal
        cleaned = client._clean_sequence("A G L V")
        assert cleaned == "AGLV"
        
        # Test uppercase
        cleaned = client._clean_sequence("aglv")
        assert cleaned == "AGLV"
        
        # Test invalid characters
        cleaned = client._clean_sequence("AGL123V")
        assert cleaned == "AGLV"
    
    def test_cache_key(self, tmp_path):
        """Test cache key generation."""
        config = AlphaFoldConfig(cache_dir=str(tmp_path / "cache"))
        client = AlphaFoldClient(config)
        
        key1 = client._get_cache_key("AGLV", "structure")
        key2 = client._get_cache_key("AGLV", "structure")
        key3 = client._get_cache_key("MGLV", "structure")
        
        assert key1 == key2  # Same content = same key
        assert key1 != key3  # Different content = different key
    
    def test_cache_save_load(self, tmp_path):
        """Test cache save and load."""
        config = AlphaFoldConfig(cache_dir=str(tmp_path / "cache"))
        client = AlphaFoldClient(config)
        
        result = PredictionResult(
            job_id="test",
            sequence="AGLV",
            pdb_string=SAMPLE_PDB,
        )
        
        key = "test_key"
        client._save_to_cache(key, result)
        
        loaded = client._load_from_cache(key)
        
        assert loaded is not None
        assert loaded.job_id == result.job_id
        assert loaded.sequence == result.sequence
    
    def test_get_cached_results(self, tmp_path):
        """Test listing cached results."""
        config = AlphaFoldConfig(cache_dir=str(tmp_path / "cache"))
        client = AlphaFoldClient(config)
        
        # Add some cache entries
        result = PredictionResult(job_id="test", sequence="A", pdb_string="")
        client._save_to_cache("key1", result)
        client._save_to_cache("key2", result)
        
        cached = client.get_cached_results()
        
        assert len(cached) == 2
        assert "key1" in cached
        assert "key2" in cached


class TestBindingSite:
    """Tests for BindingSite dataclass."""
    
    def test_create_binding_site(self):
        """Test creating binding site."""
        site = BindingSite(
            center=(10.0, 20.0, 30.0),
            residues=[1, 5, 10, 15],
            residue_names=["ALA", "GLY", "LEU", "VAL"],
            volume=500.0,
            druggability_score=0.75,
        )
        
        assert site.center == (10.0, 20.0, 30.0)
        assert len(site.residues) == 4
        assert site.druggability_score == 0.75
    
    def test_residue_string(self):
        """Test residue string generation."""
        site = BindingSite(
            center=(0, 0, 0),
            residues=[1, 5],
            residue_names=["ALA", "GLY"],
        )
        
        res_str = site.get_residue_string()
        
        assert "ALA1" in res_str
        assert "GLY5" in res_str
    
    def test_to_dict(self):
        """Test serialization."""
        site = BindingSite(
            center=(1.0, 2.0, 3.0),
            residues=[1],
            residue_names=["ALA"],
            druggability_score=0.5,
        )
        
        data = site.to_dict()
        
        assert data["center"] == [1.0, 2.0, 3.0]
        assert data["druggability_score"] == 0.5


class TestStructureQuality:
    """Tests for StructureQuality dataclass."""
    
    def test_is_high_quality(self):
        """Test quality threshold check."""
        high_quality = StructureQuality(
            mean_plddt=85.0,
            low_confidence_regions=[],
        )
        
        low_quality = StructureQuality(
            mean_plddt=55.0,
            low_confidence_regions=[(10, 20), (30, 40)],
        )
        
        assert high_quality.is_high_quality(threshold=70.0)
        assert not low_quality.is_high_quality(threshold=70.0)


class TestStructureAnalyzer:
    """Tests for StructureAnalyzer."""
    
    def test_init(self):
        """Test analyzer initialization."""
        analyzer = StructureAnalyzer(SAMPLE_PDB, plddt_scores=[80.0, 75.0, 90.0])
        
        assert analyzer.pdb_string == SAMPLE_PDB
        assert len(analyzer.plddt_scores) == 3
    
    def test_assess_quality(self):
        """Test quality assessment."""
        analyzer = StructureAnalyzer(
            SAMPLE_PDB,
            plddt_scores=[80.0, 40.0, 45.0, 90.0, 85.0],
        )
        
        quality = analyzer.assess_quality()
        
        assert quality.mean_plddt == 68.0
        assert len(quality.low_confidence_regions) > 0
    
    def test_get_contact_map(self):
        """Test contact map calculation."""
        analyzer = StructureAnalyzer(SAMPLE_PDB)
        
        if analyzer.structure is not None:
            contact_map = analyzer.get_contact_map(distance_cutoff=10.0)
            
            # Should be symmetric
            if contact_map.size > 0:
                np.testing.assert_array_equal(contact_map, contact_map.T)


class TestLigandInteraction:
    """Tests for LigandInteraction dataclass."""
    
    def test_create_interaction(self):
        """Test creating interaction."""
        interaction = LigandInteraction(
            interaction_type="hydrogen_bond",
            ligand_atom="O1",
            protein_residue="ALA10",
            protein_atom="N",
            distance=2.8,
            strength=0.9,
        )
        
        assert interaction.interaction_type == "hydrogen_bond"
        assert interaction.distance == 2.8
    
    def test_to_dict(self):
        """Test serialization."""
        interaction = LigandInteraction(
            interaction_type="hydrophobic",
            ligand_atom="C1",
            protein_residue="LEU5",
            protein_atom="CD1",
            distance=3.5,
        )
        
        data = interaction.to_dict()
        
        assert data["type"] == "hydrophobic"
        assert data["distance"] == 3.5


class TestComplexAnalysis:
    """Tests for ComplexAnalysis dataclass."""
    
    def test_create_analysis(self):
        """Test creating analysis."""
        interactions = [
            LigandInteraction("hydrogen_bond", "O1", "ALA10", "N", 2.8),
            LigandInteraction("hydrogen_bond", "N1", "GLY15", "O", 3.0),
            LigandInteraction("hydrophobic", "C1", "LEU5", "CD1", 3.5),
        ]
        
        analysis = ComplexAnalysis(
            ligand_smiles="CCO",
            binding_site=None,
            interactions=interactions,
            binding_energy_estimate=-5.5,
        )
        
        summary = analysis.get_interaction_summary()
        
        assert summary["hydrogen_bond"] == 2
        assert summary["hydrophobic"] == 1
    
    def test_to_dict(self):
        """Test serialization."""
        analysis = ComplexAnalysis(
            ligand_smiles="CCO",
            binding_site=None,
            interactions=[],
            binding_energy_estimate=-3.0,
            ligand_efficiency=0.5,
        )
        
        data = analysis.to_dict()
        
        assert data["ligand_smiles"] == "CCO"
        assert data["binding_energy_estimate"] == -3.0


class TestComplexPredictor:
    """Tests for ComplexPredictor."""
    
    def test_init(self, tmp_path):
        """Test predictor initialization."""
        config = AlphaFoldConfig(cache_dir=str(tmp_path / "cache"))
        client = AlphaFoldClient(config)
        predictor = ComplexPredictor(client)
        
        assert predictor.client is not None
        assert "hydrogen_bond" in predictor.cutoffs
    
    def test_parse_complex_atoms(self, tmp_path):
        """Test PDB atom parsing."""
        config = AlphaFoldConfig(cache_dir=str(tmp_path / "cache"))
        client = AlphaFoldClient(config)
        predictor = ComplexPredictor(client)
        
        ligand_atoms, protein_atoms = predictor._parse_complex_atoms(SAMPLE_PDB)
        
        # All atoms in sample are protein (standard residues)
        assert len(protein_atoms) > 0
    
    def test_hbond_strength(self, tmp_path):
        """Test H-bond strength estimation."""
        config = AlphaFoldConfig(cache_dir=str(tmp_path / "cache"))
        client = AlphaFoldClient(config)
        predictor = ComplexPredictor(client)
        
        # Closer = stronger
        strong = predictor._hbond_strength(2.5)
        medium = predictor._hbond_strength(3.0)
        weak = predictor._hbond_strength(3.5)
        
        assert strong > medium > weak


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
