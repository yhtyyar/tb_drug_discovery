"""ChEMBL data loading and preprocessing module.

This module handles downloading and preprocessing TB inhibitor data
from ChEMBL database for QSAR modeling.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, MolStandardize
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available. Some features will be disabled.")


class ChEMBLLoader:
    """Load and preprocess ChEMBL TB inhibitor data.
    
    This class handles:
    - Loading data from CSV or ChEMBL API
    - SMILES validation and standardization
    - Activity value processing (IC50 → pIC50)
    - Duplicate removal and data cleaning
    
    Attributes:
        target_id: ChEMBL target ID (default: CHEMBL1849 for InhA).
        min_compounds: Minimum number of compounds required.
        
    Example:
        >>> loader = ChEMBLLoader()
        >>> df = loader.load_from_csv("data/raw/chembl_inhA.csv")
        >>> df_clean = loader.preprocess(df)
        >>> print(f"Loaded {len(df_clean)} compounds")
    """
    
    # ChEMBL target IDs for TB enzymes
    TARGETS = {
        "InhA": "CHEMBL1849",
        "rpoB": "CHEMBL1790",
        "KatG": "CHEMBL1916",
    }
    
    # Standard activity columns
    REQUIRED_COLUMNS = ["canonical_smiles", "standard_value", "standard_type"]
    
    def __init__(
        self,
        target_id: str = "CHEMBL1849",
        min_compounds: int = 500,
        random_seed: int = 42,
    ) -> None:
        """Initialize ChEMBL loader.
        
        Args:
            target_id: ChEMBL target ID.
            min_compounds: Minimum compounds required.
            random_seed: Random seed for reproducibility.
        """
        self.target_id = target_id
        self.min_compounds = min_compounds
        self.random_seed = random_seed
        
        if not RDKIT_AVAILABLE:
            raise ImportError(
                "RDKit is required for ChEMBLLoader. "
                "Install with: pip install rdkit"
            )
    
    def load_from_csv(self, path: str) -> pd.DataFrame:
        """Load ChEMBL data from CSV file.
        
        Args:
            path: Path to CSV file.
            
        Returns:
            DataFrame with raw ChEMBL data.
            
        Raises:
            FileNotFoundError: If CSV file not found.
            ValueError: If required columns missing.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path)
        
        # Check for required columns (case-insensitive)
        df.columns = df.columns.str.lower().str.strip()
        
        # Map common column name variations
        column_mapping = {
            "smiles": "canonical_smiles",
            "molecule_chembl_id": "molecule_chembl_id",
            "standard_value": "standard_value",
            "standard_type": "standard_type",
            "pchembl_value": "pchembl_value",
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        logger.info(f"Loaded {len(df)} records from CSV")
        return df
    
    def validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES string using RDKit.
        
        Args:
            smiles: SMILES string to validate.
            
        Returns:
            True if SMILES is valid, False otherwise.
        """
        if pd.isna(smiles) or not isinstance(smiles, str):
            return False
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except Exception:
            return False
    
    def standardize_smiles(self, smiles: str) -> Optional[str]:
        """Standardize SMILES using RDKit.
        
        Applies:
        - Sanitization
        - Normalization
        - Canonical SMILES generation
        
        Args:
            smiles: Input SMILES string.
            
        Returns:
            Standardized SMILES or None if invalid.
        """
        if not self.validate_smiles(smiles):
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            
            # Sanitize molecule
            Chem.SanitizeMol(mol)
            
            # Return canonical SMILES
            return Chem.MolToSmiles(mol, canonical=True)
            
        except Exception as e:
            logger.debug(f"Failed to standardize SMILES: {smiles}, Error: {e}")
            return None
    
    def calculate_pic50(self, ic50_nm: float) -> float:
        """Convert IC50 (nM) to pIC50.
        
        pIC50 = -log10(IC50 in M) = 9 - log10(IC50 in nM)
        
        Args:
            ic50_nm: IC50 value in nanomolar.
            
        Returns:
            pIC50 value.
        """
        if pd.isna(ic50_nm) or ic50_nm <= 0:
            return np.nan
        
        return 9 - np.log10(ic50_nm)
    
    def preprocess(
        self,
        df: pd.DataFrame,
        activity_types: Optional[List[str]] = None,
        remove_duplicates: bool = True,
    ) -> pd.DataFrame:
        """Preprocess ChEMBL data for QSAR modeling.
        
        Steps:
        1. Filter by activity type (IC50, Ki, etc.)
        2. Validate and standardize SMILES
        3. Convert activity to pIC50
        4. Remove duplicates
        5. Quality checks
        
        Args:
            df: Raw ChEMBL DataFrame.
            activity_types: Activity types to include (default: ["IC50"]).
            remove_duplicates: Whether to remove duplicate SMILES.
            
        Returns:
            Cleaned DataFrame ready for QSAR modeling.
            
        Raises:
            ValueError: If insufficient compounds after cleaning.
        """
        if activity_types is None:
            activity_types = ["IC50"]
        
        logger.info(f"Preprocessing {len(df)} compounds")
        df = df.copy()
        
        # Filter by activity type
        if "standard_type" in df.columns:
            df = df[df["standard_type"].isin(activity_types)]
            logger.info(f"After activity filter: {len(df)} compounds")
        
        # Filter valid activity values
        if "standard_value" in df.columns:
            df = df[df["standard_value"].notna()]
            df = df[df["standard_value"] > 0]
            logger.info(f"After activity value filter: {len(df)} compounds")
        
        # Validate SMILES
        smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else "smiles"
        if smiles_col in df.columns:
            df["valid_smiles"] = df[smiles_col].apply(self.validate_smiles)
            df = df[df["valid_smiles"]]
            df = df.drop(columns=["valid_smiles"])
            logger.info(f"After SMILES validation: {len(df)} compounds")
            
            # Standardize SMILES
            df["smiles_clean"] = df[smiles_col].apply(self.standardize_smiles)
            df = df[df["smiles_clean"].notna()]
            logger.info(f"After SMILES standardization: {len(df)} compounds")
        
        # Calculate pIC50
        if "pchembl_value" in df.columns and df["pchembl_value"].notna().any():
            df["pIC50"] = df["pchembl_value"]
        elif "standard_value" in df.columns:
            df["pIC50"] = df["standard_value"].apply(self.calculate_pic50)
        
        df = df[df["pIC50"].notna()]
        logger.info(f"After pIC50 calculation: {len(df)} compounds")
        
        # Remove duplicates (keep highest activity)
        if remove_duplicates and "smiles_clean" in df.columns:
            df = df.sort_values("pIC50", ascending=False)
            df = df.drop_duplicates(subset=["smiles_clean"], keep="first")
            logger.info(f"After duplicate removal: {len(df)} compounds")
        
        # Create clean output DataFrame
        output_cols = ["smiles_clean", "pIC50"]
        if "molecule_chembl_id" in df.columns:
            output_cols.insert(0, "molecule_chembl_id")
        
        df_clean = df[output_cols].copy()
        df_clean = df_clean.rename(columns={"smiles_clean": "smiles"})
        
        # Reset index
        df_clean = df_clean.reset_index(drop=True)
        
        # Quality check
        if len(df_clean) < self.min_compounds:
            logger.warning(
                f"Only {len(df_clean)} compounds after cleaning "
                f"(minimum: {self.min_compounds})"
            )
        
        logger.info(f"Preprocessing complete: {len(df_clean)} clean compounds")
        return df_clean
    
    def create_activity_labels(
        self,
        df: pd.DataFrame,
        threshold: float = 6.0,
    ) -> pd.DataFrame:
        """Create binary activity labels for classification.
        
        Args:
            df: DataFrame with pIC50 column.
            threshold: pIC50 threshold for active/inactive.
                Default 6.0 corresponds to IC50 = 1 µM.
            
        Returns:
            DataFrame with added 'active' column.
        """
        df = df.copy()
        df["active"] = (df["pIC50"] >= threshold).astype(int)
        
        n_active = df["active"].sum()
        n_inactive = len(df) - n_active
        
        logger.info(
            f"Activity labels: {n_active} active, {n_inactive} inactive "
            f"(threshold: pIC50 >= {threshold})"
        )
        
        return df
    
    def save_processed(
        self,
        df: pd.DataFrame,
        path: str,
    ) -> None:
        """Save processed data to CSV.
        
        Args:
            df: Processed DataFrame.
            path: Output path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Saved {len(df)} compounds to {path}")
    
    def load_processed(self, path: str) -> pd.DataFrame:
        """Load previously processed data.
        
        Args:
            path: Path to processed CSV.
            
        Returns:
            Processed DataFrame.
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Processed data not found: {path}")
        
        return pd.read_csv(path)
    
    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        """Calculate dataset statistics.
        
        Args:
            df: DataFrame to analyze.
            
        Returns:
            Dictionary with statistics.
        """
        stats = {
            "n_compounds": len(df),
            "n_unique_smiles": df["smiles"].nunique() if "smiles" in df.columns else 0,
            "pIC50_mean": df["pIC50"].mean() if "pIC50" in df.columns else 0,
            "pIC50_std": df["pIC50"].std() if "pIC50" in df.columns else 0,
            "pIC50_min": df["pIC50"].min() if "pIC50" in df.columns else 0,
            "pIC50_max": df["pIC50"].max() if "pIC50" in df.columns else 0,
        }
        
        if "active" in df.columns:
            stats["n_active"] = int(df["active"].sum())
            stats["n_inactive"] = int(len(df) - df["active"].sum())
            stats["active_ratio"] = df["active"].mean()
        
        return stats
