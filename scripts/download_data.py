#!/usr/bin/env python
"""Download ChEMBL TB inhibitor data.

This script downloads bioactivity data for Mycobacterium tuberculosis
InhA enzyme from ChEMBL database.

Usage:
    python scripts/download_data.py --target CHEMBL1849 --output data/raw/

Example:
    python scripts/download_data.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


def download_chembl_activities(
    target_id: str = "CHEMBL1849",
    activity_type: str = "IC50",
    output_dir: str = "data/raw",
) -> pd.DataFrame:
    """Download bioactivity data from ChEMBL API.
    
    Args:
        target_id: ChEMBL target ID (default: CHEMBL1849 for InhA).
        activity_type: Activity type to filter (default: IC50).
        output_dir: Directory to save the downloaded data.
        
    Returns:
        DataFrame with downloaded bioactivity data.
    """
    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    
    params = {
        "target_chembl_id": target_id,
        "standard_type": activity_type,
        "limit": 1000,
    }
    
    all_activities = []
    offset = 0
    
    print(f"Downloading {activity_type} data for target {target_id}...")
    
    pbar = tqdm(desc="Records downloaded", unit=" records")
    
    while True:
        params["offset"] = offset
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"\nError downloading data: {e}")
            break
        
        data = response.json()
        activities = data.get("activities", [])
        
        if not activities:
            break
        
        all_activities.extend(activities)
        pbar.update(len(activities))
        
        offset += len(activities)
        
        if len(activities) < 1000:
            break
    
    pbar.close()
    
    if not all_activities:
        print("No data downloaded!")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_activities)
    
    # Save to file
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"chembl_{target_id.lower().replace('chembl', '')}_{activity_type.lower()}.csv"
    filepath = output_path / filename
    
    df.to_csv(filepath, index=False)
    print(f"\nSaved {len(df)} records to: {filepath}")
    
    # Print summary
    print("\nDataset Summary:")
    print(f"  Total records: {len(df)}")
    
    if "canonical_smiles" in df.columns:
        unique_smiles = df["canonical_smiles"].nunique()
        print(f"  Unique molecules: {unique_smiles}")
    
    if "standard_value" in df.columns:
        valid_values = df["standard_value"].notna().sum()
        print(f"  Valid activity values: {valid_values}")
    
    return df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download ChEMBL TB inhibitor data"
    )
    
    parser.add_argument(
        "--target",
        type=str,
        default="CHEMBL1849",
        help="ChEMBL target ID (default: CHEMBL1849 for InhA)",
    )
    
    parser.add_argument(
        "--activity-type",
        type=str,
        default="IC50",
        help="Activity type to download (default: IC50)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory (default: data/raw)",
    )
    
    args = parser.parse_args()
    
    # Download data
    df = download_chembl_activities(
        target_id=args.target,
        activity_type=args.activity_type,
        output_dir=args.output,
    )
    
    if df.empty:
        return 1
    
    print("\n✅ Download complete!")
    print("\nNext steps:")
    print("  1. Run notebook: notebooks/01_data_loading.ipynb")
    print("  2. Or run: python scripts/train_qsar.py --data data/raw/chembl_*.csv")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
