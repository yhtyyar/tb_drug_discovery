"""Unit tests for data loading and preprocessing modules."""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.chembl_loader import ChEMBLLoader
from data.descriptor_calculator import DescriptorCalculator
from data.data_preprocessor import DataPreprocessor


class TestChEMBLLoader:
    """Tests for ChEMBLLoader class."""
    
    @pytest.fixture
    def loader(self):
        """Create loader instance."""
        return ChEMBLLoader()
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            "canonical_smiles": [
                "CCO",  # Ethanol
                "CC(=O)O",  # Acetic acid
                "c1ccccc1",  # Benzene
                "INVALID_SMILES",
                None,
            ],
            "standard_value": [100, 200, 50, 300, 400],
            "standard_type": ["IC50", "IC50", "IC50", "IC50", "IC50"],
        })
    
    def test_validate_smiles_valid(self, loader):
        """Test SMILES validation with valid molecules."""
        assert loader.validate_smiles("CCO") is True
        assert loader.validate_smiles("c1ccccc1") is True
        assert loader.validate_smiles("CC(=O)O") is True
    
    def test_validate_smiles_invalid(self, loader):
        """Test SMILES validation with invalid inputs."""
        assert loader.validate_smiles("INVALID") is False
        assert loader.validate_smiles("") is False
        assert loader.validate_smiles(None) is False
    
    def test_standardize_smiles(self, loader):
        """Test SMILES standardization."""
        # Canonical SMILES should be returned
        result = loader.standardize_smiles("CCO")
        assert result == "CCO"
        
        # Invalid SMILES should return None
        result = loader.standardize_smiles("INVALID")
        assert result is None
    
    def test_calculate_pic50(self, loader):
        """Test pIC50 calculation from IC50 (nM)."""
        # IC50 = 1 nM → pIC50 = 9
        assert loader.calculate_pic50(1) == pytest.approx(9.0)
        
        # IC50 = 1000 nM (1 µM) → pIC50 = 6
        assert loader.calculate_pic50(1000) == pytest.approx(6.0)
        
        # IC50 = 10 nM → pIC50 = 8
        assert loader.calculate_pic50(10) == pytest.approx(8.0)
        
        # Invalid values
        assert np.isnan(loader.calculate_pic50(0))
        assert np.isnan(loader.calculate_pic50(-1))
    
    def test_preprocess(self, loader, sample_df):
        """Test data preprocessing pipeline."""
        result = loader.preprocess(sample_df)
        
        # Should have 3 valid compounds (excluding invalid SMILES and None)
        assert len(result) == 3
        
        # Should have required columns
        assert "smiles" in result.columns
        assert "pIC50" in result.columns
    
    def test_create_activity_labels(self, loader, sample_df):
        """Test binary activity label creation."""
        df_clean = loader.preprocess(sample_df)
        result = loader.create_activity_labels(df_clean, threshold=7.0)
        
        assert "active" in result.columns
        assert result["active"].dtype in [np.int32, np.int64, int]
        assert set(result["active"].unique()).issubset({0, 1})
    
    def test_get_statistics(self, loader, sample_df):
        """Test statistics calculation."""
        df_clean = loader.preprocess(sample_df)
        stats = loader.get_statistics(df_clean)
        
        assert "n_compounds" in stats
        assert "pIC50_mean" in stats
        assert "pIC50_std" in stats
        assert stats["n_compounds"] == 3


class TestDescriptorCalculator:
    """Tests for DescriptorCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        return DescriptorCalculator(
            lipinski=True,
            topological=True,
            extended=False,  # Faster tests
        )
    
    def test_calculate_single(self, calculator):
        """Test descriptor calculation for single molecule."""
        result = calculator.calculate("CCO")  # Ethanol
        
        assert result is not None
        assert "MolWt" in result
        assert "LogP" in result
        assert "HBD" in result
        assert "HBA" in result
        
        # Check reasonable values for ethanol
        assert result["MolWt"] == pytest.approx(46.07, rel=0.01)
    
    def test_calculate_invalid(self, calculator):
        """Test handling of invalid SMILES."""
        result = calculator.calculate("INVALID_SMILES")
        assert result is None
    
    def test_calculate_batch(self, calculator):
        """Test batch descriptor calculation."""
        smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
        result = calculator.calculate_batch(smiles_list, show_progress=False)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "smiles" in result.columns
        assert "MolWt" in result.columns
    
    def test_descriptor_names(self, calculator):
        """Test descriptor name retrieval."""
        names = calculator.descriptor_names
        
        assert isinstance(names, list)
        assert len(names) > 0
        assert "MolWt" in names
    
    def test_check_lipinski(self, calculator):
        """Test Lipinski Rule of 5 checking."""
        # Ethanol should pass all rules
        result = calculator.check_lipinski("CCO")
        
        assert result["MolWt_ok"] is True
        assert result["LogP_ok"] is True
        assert result["passes_ro5"] is True


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return DataPreprocessor(random_seed=42)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        return X, y
    
    def test_split_data(self, preprocessor, sample_data):
        """Test train/val/test splitting."""
        X, y = sample_data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
            X, y, test_size=0.15, val_size=0.15
        )
        
        # Check sizes
        assert len(X_train) == pytest.approx(70, abs=5)
        assert len(X_val) == pytest.approx(15, abs=5)
        assert len(X_test) == pytest.approx(15, abs=5)
        
        # Check no data leakage
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(X)
    
    def test_fit_transform(self, preprocessor, sample_data):
        """Test scaler fitting and transformation."""
        X, _ = sample_data
        
        X_scaled = preprocessor.fit_transform(X)
        
        # Should have mean ~0 and std ~1
        assert np.abs(X_scaled.mean()) < 0.1
        assert np.abs(X_scaled.std() - 1.0) < 0.1
    
    def test_transform_before_fit(self, preprocessor, sample_data):
        """Test error when transforming before fitting."""
        X, _ = sample_data
        
        with pytest.raises(ValueError, match="not fitted"):
            preprocessor.transform(X)
    
    def test_handle_missing_values(self, preprocessor):
        """Test missing value imputation."""
        X = np.array([
            [1.0, np.nan, 3.0],
            [4.0, 5.0, np.nan],
            [7.0, 8.0, 9.0],
        ])
        
        result = preprocessor.handle_missing_values(X, strategy="mean")
        
        # No NaN values should remain
        assert not np.isnan(result).any()
    
    def test_reproducibility(self, sample_data):
        """Test that random seed ensures reproducibility."""
        X, y = sample_data
        
        prep1 = DataPreprocessor(random_seed=42)
        prep2 = DataPreprocessor(random_seed=42)
        
        X1_train, _, _, _, _, _ = prep1.split_data(X, y)
        X2_train, _, _, _, _, _ = prep2.split_data(X, y)
        
        np.testing.assert_array_equal(X1_train, X2_train)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
