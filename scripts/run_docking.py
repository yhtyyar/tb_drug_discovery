#!/usr/bin/env python
"""Molecular docking pipeline script.

This script runs AutoDock Vina docking for TB drug discovery.

Usage:
    # Prepare TB target and dock compounds
    python scripts/run_docking.py --target InhA --compounds data/processed/cleaned_chembl_inhA.csv
    
    # Use custom receptor
    python scripts/run_docking.py --receptor protein.pdb --center 10 20 30 --compounds ligands.csv

Requirements:
    - AutoDock Vina installed and in PATH
    - Open Babel installed and in PATH
    - RDKit for SMILES processing
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.docking.protein_prep import ProteinPreparator, prepare_tb_target, TB_TARGETS
from src.docking.vina_docker import VinaDocker
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run molecular docking with AutoDock Vina",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dock against InhA (automatic setup)
    python scripts/run_docking.py --target InhA --compounds compounds.csv
    
    # Custom receptor with specified binding site
    python scripts/run_docking.py --receptor receptor.pdb \\
        --center 10.5 20.3 15.7 --size 25 25 25 \\
        --compounds compounds.csv
    
    # High exhaustiveness for final results
    python scripts/run_docking.py --target InhA --compounds compounds.csv \\
        --exhaustiveness 32 --num-modes 20

Available TB targets: InhA, KatG, DprE1, MmpL3
        """
    )
    
    # Target options (mutually exclusive groups)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target",
        type=str,
        choices=list(TB_TARGETS.keys()),
        help="TB target name (auto-downloads and prepares structure)"
    )
    target_group.add_argument(
        "--receptor",
        type=str,
        help="Path to receptor PDB/PDBQT file"
    )
    
    # Binding site (required if using --receptor)
    parser.add_argument(
        "--center",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Docking box center coordinates (required with --receptor)"
    )
    parser.add_argument(
        "--size",
        type=float,
        nargs=3,
        default=[25, 25, 25],
        metavar=("X", "Y", "Z"),
        help="Docking box size in Angstroms (default: 25 25 25)"
    )
    
    # Input compounds
    parser.add_argument(
        "--compounds",
        type=str,
        required=True,
        help="CSV file with compounds (must have 'smiles' column)"
    )
    parser.add_argument(
        "--smiles-col",
        type=str,
        default="smiles",
        help="Column name for SMILES (default: smiles)"
    )
    parser.add_argument(
        "--name-col",
        type=str,
        default=None,
        help="Column name for compound names (optional)"
    )
    parser.add_argument(
        "--max-compounds",
        type=int,
        default=None,
        help="Maximum number of compounds to dock (for testing)"
    )
    
    # Docking parameters
    parser.add_argument(
        "--exhaustiveness",
        type=int,
        default=8,
        help="Search exhaustiveness (default: 8, higher = slower but better)"
    )
    parser.add_argument(
        "--num-modes",
        type=int,
        default=9,
        help="Number of binding modes to generate (default: 9)"
    )
    parser.add_argument(
        "--energy-range",
        type=float,
        default=3.0,
        help="Maximum energy difference between poses (default: 3.0)"
    )
    parser.add_argument(
        "--cpu",
        type=int,
        default=0,
        help="Number of CPUs to use (default: 0 = auto)"
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="results/docking/docking_results.csv",
        help="Output CSV file for results"
    )
    parser.add_argument(
        "--save-poses",
        action="store_true",
        help="Save docked poses to files"
    )
    
    # Tool paths
    parser.add_argument(
        "--vina-path",
        type=str,
        default="vina",
        help="Path to Vina executable"
    )
    parser.add_argument(
        "--obabel-path",
        type=str,
        default="obabel",
        help="Path to Open Babel executable"
    )
    
    return parser.parse_args()


def check_dependencies(vina_path: str, obabel_path: str) -> bool:
    """Check if required tools are installed."""
    docker = VinaDocker(vina_path=vina_path, obabel_path=obabel_path)
    status = docker.check_dependencies()
    
    logger.info("Dependency check:")
    for tool, available in status.items():
        status_str = "✅ Available" if available else "❌ Not found"
        logger.info(f"  {tool}: {status_str}")
    
    if not status["vina"]:
        logger.error("AutoDock Vina not found. Install from: https://vina.scripps.edu/")
        return False
    
    if not status["obabel"]:
        logger.error("Open Babel not found. Install from: https://openbabel.org/")
        return False
    
    return True


def main():
    """Main docking pipeline."""
    args = parse_args()
    
    # Setup logging
    setup_logger(log_file="logs/docking.log")
    logger.info("=" * 60)
    logger.info("MOLECULAR DOCKING PIPELINE")
    logger.info("=" * 60)
    
    # Check dependencies
    if not check_dependencies(args.vina_path, args.obabel_path):
        return 1
    
    # Prepare output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Prepare receptor
    logger.info("\nStep 1: Preparing receptor...")
    
    if args.target:
        # Use predefined TB target
        logger.info(f"Using TB target: {args.target}")
        target_info = prepare_tb_target(args.target, output_dir="data/structures")
        receptor_path = target_info["receptor_pdb"]
        center = target_info["center"]
        size = target_info["size"]
        logger.info(f"  PDB ID: {target_info['pdb_id']}")
        logger.info(f"  Description: {target_info['description']}")
    else:
        # Use custom receptor
        if not args.center:
            logger.error("--center is required when using --receptor")
            return 1
        receptor_path = args.receptor
        center = tuple(args.center)
        size = tuple(args.size)
    
    logger.info(f"  Receptor: {receptor_path}")
    logger.info(f"  Center: {center}")
    logger.info(f"  Size: {size}")
    
    # Step 2: Load compounds
    logger.info("\nStep 2: Loading compounds...")
    
    df = pd.read_csv(args.compounds)
    logger.info(f"  Loaded {len(df)} compounds from {args.compounds}")
    
    if args.max_compounds:
        df = df.head(args.max_compounds)
        logger.info(f"  Limited to {len(df)} compounds")
    
    smiles_list = df[args.smiles_col].dropna().tolist()
    
    if args.name_col and args.name_col in df.columns:
        names = df[args.name_col].tolist()
    else:
        names = [f"compound_{i}" for i in range(len(smiles_list))]
    
    logger.info(f"  Valid SMILES: {len(smiles_list)}")
    
    # Step 3: Initialize docking
    logger.info("\nStep 3: Initializing docking...")
    
    docker = VinaDocker(
        vina_path=args.vina_path,
        obabel_path=args.obabel_path,
        num_modes=args.num_modes,
        exhaustiveness=args.exhaustiveness,
        energy_range=args.energy_range,
        cpu=args.cpu,
    )
    
    # Prepare receptor PDBQT
    prep = ProteinPreparator()
    receptor_pdbqt = prep.work_dir / "receptor.pdbqt"
    
    docker.prepare_receptor(receptor_path, str(receptor_pdbqt))
    docker.set_receptor(str(receptor_pdbqt), center, size)
    
    logger.info(f"  Exhaustiveness: {args.exhaustiveness}")
    logger.info(f"  Num modes: {args.num_modes}")
    
    # Step 4: Run docking
    logger.info("\nStep 4: Running docking...")
    
    results = docker.dock_batch(smiles_list, names, progress=True)
    
    # Step 5: Analyze results
    logger.info("\nStep 5: Analyzing results...")
    
    valid_results = results.dropna(subset=["affinity"])
    
    logger.info(f"  Successful docking: {len(valid_results)}/{len(results)}")
    
    if len(valid_results) > 0:
        best_affinity = valid_results["affinity"].min()
        mean_affinity = valid_results["affinity"].mean()
        
        logger.info(f"  Best affinity: {best_affinity:.2f} kcal/mol")
        logger.info(f"  Mean affinity: {mean_affinity:.2f} kcal/mol")
        
        # Top 10 compounds
        top10 = valid_results.nsmallest(10, "affinity")
        logger.info("\n  Top 10 compounds:")
        for _, row in top10.iterrows():
            logger.info(f"    {row['ligand_name']}: {row['affinity']:.2f} kcal/mol")
    
    # Step 6: Save results
    logger.info("\nStep 6: Saving results...")
    
    results.to_csv(output_path, index=False)
    logger.info(f"  Results saved to: {output_path}")
    
    # Save summary
    summary = {
        "target": args.target or args.receptor,
        "n_compounds": len(smiles_list),
        "n_successful": len(valid_results),
        "best_affinity": float(valid_results["affinity"].min()) if len(valid_results) > 0 else None,
        "mean_affinity": float(valid_results["affinity"].mean()) if len(valid_results) > 0 else None,
        "parameters": {
            "exhaustiveness": args.exhaustiveness,
            "num_modes": args.num_modes,
            "center": list(center),
            "size": list(size),
        }
    }
    
    summary_path = output_path.parent / "docking_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Summary saved to: {summary_path}")
    
    # Cleanup
    docker.cleanup()
    
    logger.info("\n" + "=" * 60)
    logger.info("DOCKING COMPLETE")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
