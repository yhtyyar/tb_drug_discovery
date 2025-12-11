"""Molecular descriptor calculation module.

This module calculates RDKit molecular descriptors for QSAR modeling,
including Lipinski descriptors, topological features, and extended properties.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Lipinski, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available. Descriptor calculation disabled.")


class DescriptorCalculator:
    """Calculate molecular descriptors for QSAR modeling.
    
    Computes various descriptor sets:
    - Lipinski descriptors (drug-likeness)
    - Topological descriptors (connectivity)
    - Extended descriptors (additional properties)
    
    Attributes:
        descriptor_sets: List of descriptor sets to calculate.
        
    Example:
        >>> calc = DescriptorCalculator()
        >>> descriptors = calc.calculate("CCO")  # Ethanol
        >>> print(descriptors["MolWt"])
        46.07
    """
    
    # Lipinski Rule of 5 descriptors
    LIPINSKI_DESCRIPTORS = [
        ("MolWt", Descriptors.MolWt),
        ("LogP", Descriptors.MolLogP),
        ("TPSA", Descriptors.TPSA),
        ("HBD", Descriptors.NumHDonors),
        ("HBA", Descriptors.NumHAcceptors),
    ]
    
    # Topological descriptors
    TOPOLOGICAL_DESCRIPTORS = [
        ("NumRotatableBonds", Descriptors.NumRotatableBonds),
        ("RingCount", Descriptors.RingCount),
        ("NumAromaticRings", Descriptors.NumAromaticRings),
        ("NumAliphaticRings", Descriptors.NumAliphaticRings),
        ("NumHeteroatoms", Descriptors.NumHeteroatoms),
        ("NumHeavyAtoms", Descriptors.HeavyAtomCount),
        ("FractionCSP3", Descriptors.FractionCSP3),
    ]
    
    # Extended descriptors
    EXTENDED_DESCRIPTORS = [
        ("LabuteASA", Descriptors.LabuteASA),
        ("BalabanJ", Descriptors.BalabanJ),
        ("BertzCT", Descriptors.BertzCT),
        ("Chi0", Descriptors.Chi0),
        ("Chi1", Descriptors.Chi1),
        ("Kappa1", Descriptors.Kappa1),
        ("Kappa2", Descriptors.Kappa2),
        ("MaxPartialCharge", Descriptors.MaxPartialCharge),
        ("MinPartialCharge", Descriptors.MinPartialCharge),
        ("MaxAbsPartialCharge", Descriptors.MaxAbsPartialCharge),
        ("MinAbsPartialCharge", Descriptors.MinAbsPartialCharge),
    ]
    
    def __init__(
        self,
        lipinski: bool = True,
        topological: bool = True,
        extended: bool = True,
    ) -> None:
        """Initialize descriptor calculator.
        
        Args:
            lipinski: Include Lipinski descriptors.
            topological: Include topological descriptors.
            extended: Include extended descriptors.
        """
        if not RDKIT_AVAILABLE:
            raise ImportError(
                "RDKit is required for DescriptorCalculator. "
                "Install with: pip install rdkit"
            )
        
        self.descriptors = []
        
        if lipinski:
            self.descriptors.extend(self.LIPINSKI_DESCRIPTORS)
        if topological:
            self.descriptors.extend(self.TOPOLOGICAL_DESCRIPTORS)
        if extended:
            self.descriptors.extend(self.EXTENDED_DESCRIPTORS)
        
        logger.info(f"Initialized calculator with {len(self.descriptors)} descriptors")
    
    @property
    def descriptor_names(self) -> List[str]:
        """Get list of descriptor names."""
        return [name for name, _ in self.descriptors]
    
    def calculate(self, smiles: str) -> Optional[Dict[str, float]]:
        """Calculate descriptors for a single molecule.
        
        Args:
            smiles: SMILES string.
            
        Returns:
            Dictionary of descriptor values, or None if calculation fails.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            result = {}
            for name, func in self.descriptors:
                try:
                    value = func(mol)
                    # Handle NaN and Inf values
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    result[name] = float(value)
                except Exception:
                    result[name] = 0.0
            
            return result
            
        except Exception as e:
            logger.debug(f"Failed to calculate descriptors for {smiles}: {e}")
            return None
    
    def calculate_batch(
        self,
        smiles_list: List[str],
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Calculate descriptors for multiple molecules.
        
        Args:
            smiles_list: List of SMILES strings.
            show_progress: Show progress bar.
            
        Returns:
            DataFrame with descriptors for each molecule.
            Rows with failed calculations are filled with NaN.
        """
        results = []
        iterator = tqdm(smiles_list, desc="Calculating descriptors") if show_progress else smiles_list
        
        for smiles in iterator:
            desc = self.calculate(smiles)
            if desc is not None:
                desc["smiles"] = smiles
                results.append(desc)
            else:
                # Create row with NaN for failed molecules
                row = {name: np.nan for name in self.descriptor_names}
                row["smiles"] = smiles
                results.append(row)
        
        df = pd.DataFrame(results)
        
        # Reorder columns: SMILES first
        cols = ["smiles"] + self.descriptor_names
        df = df[cols]
        
        # Report statistics
        valid_count = df.dropna().shape[0]
        logger.info(
            f"Calculated descriptors for {valid_count}/{len(smiles_list)} molecules"
        )
        
        return df
    
    def calculate_from_dataframe(
        self,
        df: pd.DataFrame,
        smiles_col: str = "smiles",
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Calculate descriptors from DataFrame.
        
        Args:
            df: Input DataFrame with SMILES column.
            smiles_col: Name of SMILES column.
            show_progress: Show progress bar.
            
        Returns:
            DataFrame with original data plus descriptors.
        """
        if smiles_col not in df.columns:
            raise ValueError(f"Column '{smiles_col}' not found in DataFrame")
        
        smiles_list = df[smiles_col].tolist()
        desc_df = self.calculate_batch(smiles_list, show_progress)
        
        # Merge with original data
        # Remove smiles column from descriptors to avoid duplication
        desc_df = desc_df.drop(columns=["smiles"])
        
        result = pd.concat([df.reset_index(drop=True), desc_df], axis=1)
        
        return result
    
    def check_lipinski(self, smiles: str) -> Dict[str, bool]:
        """Check Lipinski Rule of 5 compliance.
        
        Args:
            smiles: SMILES string.
            
        Returns:
            Dictionary with rule compliance for each criterion.
        """
        desc = self.calculate(smiles)
        if desc is None:
            return {"valid": False}
        
        rules = {
            "MolWt_ok": desc["MolWt"] <= 500,
            "LogP_ok": desc["LogP"] <= 5,
            "HBD_ok": desc["HBD"] <= 5,
            "HBA_ok": desc["HBA"] <= 10,
        }
        
        rules["n_violations"] = sum(not v for v in rules.values())
        rules["passes_ro5"] = rules["n_violations"] <= 1
        
        return rules
    
    def filter_druglike(
        self,
        df: pd.DataFrame,
        max_violations: int = 1,
    ) -> pd.DataFrame:
        """Filter molecules by drug-likeness.
        
        Args:
            df: DataFrame with descriptor columns.
            max_violations: Maximum Lipinski violations allowed.
            
        Returns:
            Filtered DataFrame.
        """
        if "MolWt" not in df.columns:
            raise ValueError("Descriptors not calculated. Run calculate_from_dataframe first.")
        
        mask = (
            ((df["MolWt"] > 500).astype(int) +
             (df["LogP"] > 5).astype(int) +
             (df["HBD"] > 5).astype(int) +
             (df["HBA"] > 10).astype(int))
            <= max_violations
        )
        
        result = df[mask].copy()
        logger.info(f"Drug-like filter: {len(result)}/{len(df)} molecules passed")
        
        return result
    
    def get_feature_matrix(
        self,
        df: pd.DataFrame,
        drop_na: bool = True,
    ) -> np.ndarray:
        """Extract feature matrix for ML training.
        
        Args:
            df: DataFrame with descriptors.
            drop_na: Drop rows with NaN values.
            
        Returns:
            NumPy array of shape (n_samples, n_features).
        """
        feature_cols = [col for col in self.descriptor_names if col in df.columns]
        
        if not feature_cols:
            raise ValueError("No descriptor columns found in DataFrame")
        
        X = df[feature_cols].copy()
        
        if drop_na:
            X = X.dropna()
        
        return X.values
