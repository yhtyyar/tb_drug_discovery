# Setup Guide

## Prerequisites

- Python 3.10 or higher
- Conda (recommended) or pip
- Git

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/tb_drug_discovery.git
cd tb_drug_discovery

# Create conda environment
conda create -n tb_discovery python=3.10 -y
conda activate tb_discovery

# Install RDKit (best via conda)
conda install -c conda-forge rdkit -y

# Install remaining dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Option 2: Using pip only

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/tb_drug_discovery.git
cd tb_drug_discovery

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Verify Installation

```bash
# Test RDKit
python -c "from rdkit import Chem; print('✅ RDKit OK')"

# Test PyTorch (optional)
python -c "import torch; print(f'✅ PyTorch OK, CUDA: {torch.cuda.is_available()}')"

# Run tests
pytest tests/ -v
```

## Download Data

```bash
# Download ChEMBL InhA inhibitor data
python scripts/download_data.py

# Or download manually from ChEMBL website
# Target: CHEMBL1849 (InhA)
```

## Quick Start

```bash
# Train QSAR model
python scripts/train_qsar.py --data data/raw/chembl_inhA.csv

# Or use Jupyter notebooks
jupyter notebook notebooks/
```

## GPU Setup (Optional)

For GNN training (Phase 3+), GPU acceleration is recommended:

### CUDA (NVIDIA GPU)
```bash
# Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric
```

### Google Colab
The notebooks are Colab-compatible. Simply upload to Colab and run.

## Troubleshooting

### RDKit import error
```bash
# Try conda installation
conda install -c conda-forge rdkit
```

### CUDA not detected
```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch for your CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
```

### Memory issues
- Reduce batch size in config
- Use Google Colab for larger computations
- Consider AWS/GCP instances for production runs
