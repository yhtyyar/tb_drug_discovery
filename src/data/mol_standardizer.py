"""Molecule standardization: tautomers, salts, stereo canonicalization.

Pipeline (in order):
1. Parse SMILES
2. Remove salts / largest fragment
3. Neutralize common charges
4. Canonicalize tautomer (RDKit MolStandardize)
5. Assign stereochemistry
6. Return canonical SMILES

Without standardization, tautomers of the same molecule (e.g. keto/enol)
produce different descriptors and different scaffold assignments, inflating
apparent dataset diversity and hurting model generalization.
"""

from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class MoleculeStandardizer:
    """Canonical molecule standardization pipeline.

    Attributes:
        remove_salts: Strip counter-ions / keep largest fragment.
        neutralize: Neutralize common charged groups.
        canonicalize_tautomer: Collapse tautomeric forms to canonical one.
        assign_stereo: Re-assign stereo from 2D coordinates.

    Example:
        >>> std = MoleculeStandardizer()
        >>> canon = std.standardize("CC(=O)O.[Na+]")
        >>> assert canon == "CC(=O)O"  # salt stripped

        >>> # Tautomer collapse: 1H-pyridin-2-one ↔ 2-hydroxypyridine
        >>> s1 = std.standardize("O=c1cccc[nH]1")
        >>> s2 = std.standardize("Oc1ccccn1")
        >>> assert s1 == s2
    """

    def __init__(
        self,
        remove_salts: bool = True,
        neutralize: bool = True,
        canonicalize_tautomer: bool = True,
        assign_stereo: bool = True,
    ) -> None:
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit required: pip install rdkit")

        self.remove_salts = remove_salts
        self.neutralize = neutralize
        self.canonicalize_tautomer = canonicalize_tautomer
        self.assign_stereo = assign_stereo

        # Build RDKit standardization components once
        self._largest_frag = rdMolStandardize.LargestFragmentChooser()
        self._uncharger = rdMolStandardize.Uncharger()
        self._tautomer_enum = rdMolStandardize.TautomerEnumerator()

    def standardize(self, smiles: str) -> Optional[str]:
        """Standardize a single SMILES string.

        Args:
            smiles: Input SMILES (may be non-canonical, charged, multi-fragment).

        Returns:
            Canonical standardized SMILES, or None if the molecule is invalid.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # 1. Remove salts — keep largest fragment
            if self.remove_salts:
                mol = self._largest_frag.choose(mol)
                if mol is None:
                    return None

            # 2. Neutralize charges (e.g. COO- → COOH, N+ → N)
            if self.neutralize:
                mol = self._uncharger.uncharge(mol)

            # 3. Canonical tautomer (e.g. keto/enol, imine/enamine)
            if self.canonicalize_tautomer:
                mol = self._tautomer_enum.Canonicalize(mol)

            # 4. Assign/perceive stereo from current connectivity
            if self.assign_stereo:
                Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

            return Chem.MolToSmiles(mol, isomericSmiles=True)

        except Exception as e:
            logger.debug(f"Standardization failed for '{smiles}': {e}")
            return None

    def standardize_batch(
        self,
        smiles_list: List[str],
        drop_invalid: bool = False,
    ) -> List[Optional[str]]:
        """Standardize a list of SMILES strings.

        Args:
            smiles_list: Input SMILES.
            drop_invalid: If True, filter out None results.

        Returns:
            List of standardized SMILES (None for invalid inputs unless dropped).
        """
        results = [self.standardize(smi) for smi in smiles_list]
        if drop_invalid:
            return [s for s in results if s is not None]
        return results

    def standardize_dataframe(
        self,
        df,
        smiles_col: str = "smiles",
        output_col: str = "canonical_smiles",
        drop_invalid: bool = True,
    ):
        """Add standardized SMILES column to a DataFrame.

        Args:
            df: Input DataFrame.
            smiles_col: Column with raw SMILES.
            output_col: Column name for standardized output.
            drop_invalid: Drop rows where standardization failed.

        Returns:
            DataFrame with ``output_col`` column added.
        """
        import pandas as pd

        df = df.copy()
        df[output_col] = [self.standardize(s) for s in df[smiles_col]]

        n_failed = df[output_col].isna().sum()
        if n_failed:
            logger.warning(f"Standardization: {n_failed}/{len(df)} molecules failed")

        if drop_invalid:
            df = df.dropna(subset=[output_col]).reset_index(drop=True)

        return df

    def are_same_molecule(self, smiles_a: str, smiles_b: str) -> bool:
        """Check if two SMILES represent the same molecule after standardization.

        Useful for deduplication that accounts for tautomers.
        """
        a = self.standardize(smiles_a)
        b = self.standardize(smiles_b)
        if a is None or b is None:
            return False
        return a == b

    def deduplicate(self, smiles_list: List[str]) -> Tuple[List[str], List[int]]:
        """Remove duplicates accounting for tautomers/salts.

        Args:
            smiles_list: Input SMILES.

        Returns:
            Tuple of (unique_canonical_smiles, kept_indices).
        """
        seen = {}
        unique_smiles = []
        kept_indices = []

        for i, smi in enumerate(smiles_list):
            canon = self.standardize(smi)
            if canon is None:
                continue
            if canon not in seen:
                seen[canon] = i
                unique_smiles.append(canon)
                kept_indices.append(i)

        removed = len(smiles_list) - len(unique_smiles)
        if removed:
            logger.info(f"Deduplication: removed {removed} duplicates "
                        f"({len(unique_smiles)} unique from {len(smiles_list)})")

        return unique_smiles, kept_indices
