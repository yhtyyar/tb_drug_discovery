# TB Drug Discovery ML Pipeline

**Machine Learning pipeline for Tuberculosis drug discovery targeting InhA enzyme.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/yhtyyar/tb_drug_discovery/actions/workflows/pytest.yml/badge.svg)](https://github.com/yhtyyar/tb_drug_discovery/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Machine Learning Pipeline for Tuberculosis Drug Discovery**

An integrated ML pipeline for predicting activity and de novo design of Mycobacterium tuberculosis enzyme inhibitors (InhA, rpoB).

## 🎯 Project Overview

This project implements:
- **QSAR Modeling**: Random Forest baseline for activity prediction
- **Graph Neural Networks**: GCN/MPNN for molecular property prediction
- **Molecular Docking**: AutoDock Vina integration for binding analysis
- **Generative Models**: VAE and Diffusion models for molecular design
- **Structure Prediction**: AlphaFold 3 integration for complex prediction

## 📊 Target Metrics

| Model | Metric | Target |
|-------|--------|--------|
| QSAR | ROC-AUC | > 0.75 |
| GNN | ROC-AUC | > 0.80 |
| Docking | R² correlation | > 0.50 |
| Generation | SMILES validity | > 90% |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/tb_drug_discovery.git
cd tb_drug_discovery

# Create conda environment
conda create -n tb_discovery python=3.10 -y
conda activate tb_discovery

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from rdkit import Chem; print('✅ RDKit OK')"
```

### Run QSAR Training

```bash
# Train QSAR model
python scripts/train_qsar.py

# Or use Jupyter notebook
jupyter notebook notebooks/03_qsar_training.ipynb
```

## 📁 Project Structure

```
tb_drug_discovery/
├── src/                    # Source code
│   ├── data/              # Data loading and processing
│   ├── models/            # ML models (QSAR, GNN, VAE)
│   ├── docking/           # Molecular docking
│   ├── evaluation/        # Metrics and validation
│   └── utils/             # Utilities
├── notebooks/             # Jupyter notebooks
├── data/                  # Data files
│   ├── raw/              # Original data
│   └── processed/        # Cleaned data
├── models/                # Trained models
├── results/               # Results and figures
├── tests/                 # Unit tests
├── scripts/               # CLI scripts
└── config/                # Configuration files
```

## 📈 Results

*Results will be updated as the project progresses.*

| Phase | Status | Metrics |
|-------|--------|---------|
| QSAR Baseline | 🔄 In Progress | - |
| Molecular Docking | ⏳ Pending | - |
| GNN Training | ⏳ Pending | - |
| Molecular Generation | ⏳ Pending | - |

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📚 Documentation

See the `documents/` folder for detailed documentation:
- `TECHNICAL_SPECIFICATION.md` - Full technical specification
- `QUICK_START_TODAY.md` - Getting started guide
- `GIT_WORKFLOW_SETUP.md` - Git workflow and conventions

## 🤝 Contributing

See `CONTRIBUTING.md` for contribution guidelines.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{tb_drug_discovery,
  title = {TB Drug Discovery ML Pipeline},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/tb_drug_discovery}
}
```

## 👤 Author

PhD Candidate - Machine Learning for Drug Discovery

---

**Version:** 0.1.0 (Phase 1: QSAR Baseline)
