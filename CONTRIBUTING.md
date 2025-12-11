# Contributing to TB Drug Discovery

Thank you for your interest in contributing to this project!

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/tb_drug_discovery.git
   cd tb_drug_discovery
   ```

2. **Create a virtual environment:**
   ```bash
   conda create -n tb_discovery python=3.10 -y
   conda activate tb_discovery
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Install pre-commit hooks (recommended):**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Code Standards

### Style Guide
- Follow PEP 8 with a line length of 100 characters
- Use [Black](https://github.com/psf/black) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- All functions must have type hints
- All public functions must have docstrings (Google style)

### Before Submitting

1. **Format code:**
   ```bash
   black --line-length=100 src/ tests/ scripts/
   isort --profile=black src/ tests/ scripts/
   ```

2. **Run linting:**
   ```bash
   flake8 src/ tests/ scripts/ --max-line-length=100
   mypy src/ --ignore-missing-imports
   ```

3. **Run tests:**
   ```bash
   pytest tests/ -v --cov=src
   ```

4. **Ensure all tests pass and coverage is adequate (>80%)**

## Git Workflow

### Branch Naming
- `feature/` - New features (e.g., `feature/gnn_attention`)
- `fix/` - Bug fixes (e.g., `fix/descriptor_nan`)
- `docs/` - Documentation updates (e.g., `docs/api_reference`)
- `refactor/` - Code refactoring (e.g., `refactor/data_loader`)

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or modifying tests
- `refactor`: Code refactoring
- `chore`: Maintenance tasks

Examples:
```
feat: add GNN attention visualization

- Implement attention weight extraction
- Add visualization function for molecular graphs
- Include unit tests

Closes #15
```

### Pull Request Process

1. Create a feature branch from `dev`
2. Make your changes
3. Ensure all tests pass
4. Update documentation if needed
5. Create a PR to `dev` branch
6. Wait for code review
7. Address any feedback
8. Merge after approval

## Testing

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_qsar_model.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Writing Tests
- Place tests in `tests/` directory
- Mirror the source structure (e.g., `src/models/` → `tests/test_models/`)
- Use descriptive test names: `test_<function>_<scenario>_<expected>`
- Use fixtures for shared setup

## Documentation

- Update docstrings for any modified functions
- Update README.md if adding new features
- Add examples for new functionality
- Keep notebooks up to date

## Questions?

Feel free to open an issue for any questions or concerns.
