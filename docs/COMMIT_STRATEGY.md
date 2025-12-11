# GitHub Commit Strategy

## Phase 1 Commits (Weeks 1-2)

Execute these commits sequentially as you complete each component:

### Commit 1: Initial Project Setup
```bash
git init
git add .gitignore README.md LICENSE requirements.txt setup.py pyproject.toml CONTRIBUTING.md
git add config/ docs/
git commit -m "chore: initial project structure and configuration

- Add project documentation (README, LICENSE, CONTRIBUTING)
- Configure Python environment (requirements.txt, setup.py)
- Add code quality tools configuration (pyproject.toml)
- Set up directory structure for ML pipeline"
```

### Commit 2: Core Source Structure
```bash
git add src/
git commit -m "feat: add core source code structure

- Implement data loading module (ChEMBLLoader)
- Implement descriptor calculator (RDKit descriptors)
- Implement data preprocessor (splitting, scaling)
- Add utility modules (config, logger)

Modules: src/data/, src/utils/"
```

### Commit 3: QSAR Model Implementation
```bash
git add src/models/ src/evaluation/
git commit -m "feat: implement QSAR model with Random Forest

- Add QSARModel class with regression/classification support
- Implement cross-validation utilities
- Add comprehensive evaluation metrics
- Include ROC-AUC, precision, recall, F1 calculations

Target: ROC-AUC > 0.75 on test set"
```

### Commit 4: Unit Tests
```bash
git add tests/
git commit -m "test: add comprehensive unit tests

- Test data loader (SMILES validation, pIC50 calculation)
- Test descriptor calculator (Lipinski, batch processing)
- Test QSAR model (training, prediction, serialization)
- Test data preprocessor (splitting, scaling)

Coverage target: > 80%"
```

### Commit 5: Jupyter Notebooks
```bash
git add notebooks/
git commit -m "docs: add training notebooks with visualizations

- 01_data_loading.ipynb: ChEMBL data preprocessing
- 02_descriptor_calculation.ipynb: RDKit descriptors
- 03_qsar_training.ipynb: Model training with 5-fold CV

Includes ROC curves, confusion matrices, feature importance"
```

### Commit 6: Training Scripts
```bash
git add scripts/
git commit -m "feat: add CLI training scripts

- download_data.py: ChEMBL data downloader
- train_qsar.py: Complete QSAR training pipeline

Usage: python scripts/train_qsar.py --data data/raw/chembl_inhA.csv"
```

### Commit 7: CI/CD Configuration
```bash
git add .github/
git commit -m "ci: add GitHub Actions for testing and quality

- pytest.yml: Run tests on push/PR
- code_quality.yml: Black, flake8, isort, mypy
- Issue templates for bugs and features

Triggers: push/PR to main and dev branches"
```

### Commit 8: Phase 1 Complete (Tag v0.1)
```bash
git add .
git commit -m "feat: complete Phase 1 QSAR baseline

Phase 1 deliverables:
- QSAR model achieving ROC-AUC target
- 5-fold cross-validation
- Full test suite
- Documentation and notebooks
- CI/CD pipeline

Ready for Phase 2: Molecular Docking"

git tag -a v0.1 -m "Phase 1: QSAR Baseline Complete"
git push origin main --tags
```

## Branch Strategy

```
main (production)
  │
  ├── v0.1 (Phase 1: QSAR)
  ├── v0.2 (Phase 2: Docking)
  ├── v0.3 (Phase 3: GNN)
  └── v1.0 (First Publication)
  
dev (development)
  │
  ├── feature/qsar_baseline
  ├── feature/docking_pipeline
  ├── feature/gnn_architecture
  └── fix/descriptor_nan
```

## Commit Message Format

```
<type>(<scope>): <description>

[body]

[footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `chore`: Maintenance
- `ci`: CI/CD changes

**Examples:**
```bash
feat(qsar): add hyperparameter tuning with GridSearchCV

- Implement grid search for n_estimators, max_depth
- Add early stopping based on validation loss
- Update config with tunable parameters

Improves ROC-AUC from 0.75 to 0.82

Closes #5
```

## Quick Commands

```bash
# Create feature branch
git checkout -b feature/your_feature dev

# Stage and commit
git add -A
git commit -m "feat: your message"

# Push branch
git push -u origin feature/your_feature

# Create pull request (via GitHub)

# After merge, update local
git checkout dev
git pull origin dev

# For releases
git checkout main
git merge dev
git tag -a v0.X -m "Phase X: Description"
git push origin main --tags
```
