# GIT WORKFLOW & GITHUB CI/CD SETUP
## TB Drug Discovery PhD Project

**Версия:** 2.0  
**Дата:** Декабрь 2025  
**Статус:** Production Ready

---

## ЧАСТЬ 1: ИНИЦИАЛИЗАЦИЯ GITHUB РЕПОЗИТОРИЯ

### Шаг 1: Создание репозитория (в веб-интерфейсе GitHub)

```
1. Зайти на github.com/new
2. Назвать репозиторий: tb_drug_discovery
3. Описание: 
   "ML Pipeline for TB Drug Discovery: QSAR → GNN → Generative Models → AlphaFold"
4. Выбрать Public (открытый для науки)
5. Инициализировать с README.md
6. Добавить .gitignore Python
7. Выбрать MIT License
8. Создать
```

### Шаг 2: Локальная инициализация

```bash
# Клонировать репозиторий
git clone https://github.com/YOUR_USERNAME/tb_drug_discovery.git
cd tb_drug_discovery

# Проверить remote
git remote -v
# Должно быть:
# origin  https://github.com/YOUR_USERNAME/tb_drug_discovery.git (fetch)
# origin  https://github.com/YOUR_USERNAME/tb_drug_discovery.git (push)

# Создать основные ветки
git checkout -b dev  # development branch
git push -u origin dev

git checkout main
# Создать protected rules в GitHub Settings:
# - Require pull request reviews before merging
# - Require status checks to pass before merging
# - Require branches to be up to date before merging
```

---

## ЧАСТЬ 2: COMMIT STRATEGY

### 2.1 Семантические типы commits

```
feat:     Новая функция (feature)
          feat: add GNN model architecture
          
fix:      Исправление бага
          fix: handle NaN values in descriptor calculation
          
docs:     Документация
          docs: update README with installation steps
          
test:     Добавление/изменение тестов
          test: add unit tests for QSAR model
          
refactor: Переструктурирование кода (без изменения функции)
          refactor: reorganize data loading pipeline
          
perf:     Улучшение производительности
          perf: optimize GNN inference (100x speedup)
          
chore:    Служебные изменения (dependencies, config)
          chore: update requirements.txt with torch==2.0
          
ci:       CI/CD конфигурация
          ci: add GitHub Actions workflow for pytest
```

### 2.2 Структура commit сообщения

```
feat: add Graph Neural Network implementation

- Implement GCN architecture with 3 convolutional layers
- Add global mean pooling and readout layers  
- Support batch processing for DataLoader
- Add comprehensive docstrings and type hints
- Include 100% unit test coverage

Performance: 
- Training: 30-45 min on T4 GPU for 500 molecules
- Inference: <2 min for 5000 molecules

Issues closed: #5 #6
```

### 2.3 Правила committing

**ОБЯЗАТЕЛЬНО:**
- [ ] 1 commit = 1 логическая функция (atomic commits)
- [ ] Commit сообщение четкое и информативное (вторая строка - пустая)
- [ ] Тесты MUST passing перед commit
- [ ] Code MUST быть отформатирован (black, flake8)
- [ ] Нет credentials или API keys в коде

**НЕ КОМИТИТЬ:**
```
❌ .env files (credentials)
❌ model.pkl > 100 MB (используйте Git LFS)
❌ raw data files (используйте только links)
❌ generated plots (regenerate из кода)
❌ IDE files (.vscode, .idea)
```

---

## ЧАСТЬ 3: BRANCH STRATEGY (Git Flow)

### 3.1 Основные ветки

```
main (production)
  ├─ Только release commits
  ├─ Все тесты passing
  ├─ Версионирование (v0.1, v1.0, v2.0)
  └─ Automatic deployment (опционально)

dev (development)
  ├─ Feature branches отсюда
  ├─ Может быть нестабильна
  └─ Регулярные merges из features
```

### 3.2 Feature ветки (для работы)

```
Правило именования: feature/<ISSUE_NUMBER>_<DESCRIPTION>

Примеры:
  feature/1_qsar_baseline
  feature/2_data_loader
  feature/3_gnn_architecture
  feature/4_molecular_docking
  feature/5_molecular_generation
  feature/6_alphafold_integration
  
Bugfix ветки:
  fix/7_nan_in_descriptors
  fix/8_docking_score_calculation
  
Документация:
  docs/9_readme_update
  docs/10_api_documentation
```

### 3.3 Workflow (для каждой feature)

```bash
# 1. Создать feature ветку из dev
git checkout dev
git pull origin dev
git checkout -b feature/3_gnn_architecture

# 2. Работать локально, регулярно коммитить
git add src/models/gnn_model.py
git commit -m "feat: implement GCN architecture"
git add tests/test_gnn_model.py
git commit -m "test: add unit tests for GNN"

# 3. Убедиться что тесты passing
pytest tests/test_gnn_model.py -v

# 4. Запушить ветку
git push origin feature/3_gnn_architecture

# 5. Создать Pull Request на GitHub (веб-интерфейс)
# - Описать изменения
# - Ссылаться на issue
# - Запросить review

# 6. После approved: слить в dev
# (в веб-интерфейсе GitHub)
git checkout dev
git pull origin dev

# 7. Удалить локальную feature ветку
git branch -d feature/3_gnn_architecture
```

---

## ЧАСТЬ 4: PULL REQUEST TEMPLATE

**Создать файл: `.github/pull_request_template.md`**

```markdown
## Description
Brief description of changes

## Type of change
- [ ] New feature
- [ ] Bug fix
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement

## Related Issues
Closes #[issue_number]

## Changes made
- Change 1
- Change 2
- Change 3

## Testing done
- [ ] Unit tests added/updated
- [ ] All tests passing (pytest -v)
- [ ] Code formatted (black, flake8)
- [ ] Type hints added (mypy)

## Documentation
- [ ] README updated if needed
- [ ] Docstrings added
- [ ] API documentation updated

## Performance impact
- [ ] No impact
- [ ] Improved (describe how)
- [ ] Degraded (describe why and mitigation)

## Screenshots (if applicable)
[Add plots, tables, or results]

## Checklist
- [ ] Code follows project style
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] No hardcoded values
- [ ] Reproducible (seeds, versions)
```

---

## ЧАСТЬ 5: GITHUB ACTIONS CI/CD

### 5.1 Test Workflow

**Создать файл: `.github/workflows/pytest.yml`**

```yaml
name: Tests

on:
  push:
    branches: [ main, dev ]
  pull_request:
    branches: [ main, dev ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: ['3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests with coverage
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

### 5.2 Code Quality Workflow

**Создать файл: `.github/workflows/code_quality.yml`**

```yaml
name: Code Quality

on:
  push:
    branches: [ main, dev ]
  pull_request:
    branches: [ main, dev ]

jobs:
  quality:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install linting tools
      run: |
        pip install black flake8 mypy isort
    
    - name: Check formatting with black
      run: black --check src/ tests/
    
    - name: Check with flake8
      run: flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Check type hints with mypy
      run: mypy src/ --ignore-missing-imports
    
    - name: Check import ordering
      run: isort --check-only src/ tests/
```

### 5.3 Documentation Workflow

**Создать файл: `.github/workflows/docs.yml`**

```yaml
name: Documentation

on:
  push:
    branches: [ main, dev ]
    paths: ['docs/**', 'src/**']

jobs:
  docs:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install sphinx sphinx-rtd-theme
        pip install -r requirements.txt
    
    - name: Build documentation
      run: cd docs && make html
    
    - name: Upload documentation
      uses: actions/upload-artifact@v3
      with:
        name: documentation
        path: docs/_build/html/
```

---

## ЧАСТЬ 6: VERSION MANAGEMENT

### 6.1 Versioning scheme (Semantic Versioning)

```
v MAJOR . MINOR . PATCH

v0.1.0  - QSAR baseline ready
v0.2.0  - Docking pipeline ready
v0.3.0  - GNN model trained
v0.4.0  - Molecular generation ready
v1.0.0  - Publication 1 ready (full pipeline)
v1.1.0  - GNN improvements
v2.0.0  - Publication 2 ready (Diffusion + Transformers)
v3.0.0  - Publication 3 ready (AlphaFold + MD)
```

### 6.2 Создание release

```bash
# 1. Обновить версию в коде
# Файл: src/__init__.py
__version__ = "1.0.0"

# 2. Обновить CHANGELOG.md
# - List all changes
# - Credits contributors

# 3. Создать tag
git tag -a v1.0.0 -m "Release version 1.0.0: Full QSAR+GNN+Docking pipeline"

# 4. Запушить tag
git push origin v1.0.0

# 5. Создать Release на GitHub (веб)
# - Draft release notes
# - Upload documentation
# - Link to publication

# 6. Создать DOI на Zenodo (для цитирования)
# https://guides.github.com/features/mastering-markdown/#github-flavored-markdown
```

---

## ЧАСТЬ 7: MONTHLY GITHUB WORKFLOWS

### Месячный commit review

**Каждый месяц:**
```bash
# 1. Посмотреть месячный diff
git log --oneline --since="2025-01-01" --until="2025-02-01"

# 2. Подсчитать commits
git log --since="2025-01-01" --until="2025-02-01" --oneline | wc -l

# 3. Статистика изменений
git log --since="2025-01-01" --until="2025-02-01" --stat

# 4. Визуализация
git log --since="2025-01-01" --until="2025-02-01" --graph --oneline --all
```

### Месячный отчет (шаблон)

```markdown
# Month 1 Progress Report (January 2025)

## Commits Summary
- Total commits: 24
- Feature commits: 16
- Test commits: 5
- Docs commits: 3

## Completed Tasks
- [ ] Data loading pipeline
- [ ] QSAR model training
- [ ] First synthesis batch (5 compounds)

## Key Metrics
- QSAR ROC-AUC: 0.78 (target: 0.75) ✅
- Test coverage: 82% (target: 80%) ✅
- Code quality: 0 flake8 errors ✅

## GitHub Stats
- PR merged: 5
- Issues closed: 8
- Stars: 2

## Next Month Goals
- Implement docking pipeline
- Prepare publication draft
- Synthesize 5 more compounds

## Blockers
- None currently
```

---

## ЧАСТЬ 8: ВАЖНЫЕ ФАЙЛЫ GIT ПРОЕКТА

### .gitignore

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data
data/raw/
*.csv
*.xlsx

# Models (> 100 MB используйте Git LFS)
*.pkl
*.pt
*.pth

# Results
results/
plots/*.png
plots/*.pdf

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# OS
.DS_Store
Thumbs.db

# Credentials
.env
credentials.yaml
```

### requirements.txt

```
# Core
python=3.10

# Chemistry
rdkit>=2024.03
openbabel>=3.1

# ML/DL
torch>=2.0.0
pytorch-geometric>=2.4.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0

# ML Specialized
deepchem>=4.0
transformers>=4.30

# Molecular Dynamics
openmm>=8.0
mdtraj>=1.9

# Utilities
jupyter>=7.0
matplotlib>=3.7
seaborn>=0.12
plotly>=5.14

# Testing
pytest>=7.4
pytest-cov>=4.1

# Code Quality
black>=23.0
flake8>=6.0
mypy>=1.5
isort>=5.12

# Documentation
sphinx>=7.0
sphinx-rtd-theme>=1.3
```

---

## ЧАСТЬ 9: ПРАКТИЧЕСКИЙ ПРИМЕР

### Новая feature: GNN model

```bash
# 1. Создать issue на GitHub
# Title: "Implement Graph Neural Network for TB activity prediction"
# Label: feature, gnn

# 2. Создать ветку
git checkout dev
git pull origin dev
git checkout -b feature/5_gnn_implementation

# 3. Работать на feature ветке (примеры коммитов)

# Commit 1: Молекулярное представление
git add src/models/molecular_graph.py
git add tests/test_molecular_graph.py
git commit -m "feat: implement molecular graph representation
- Convert SMILES to RDKit graphs
- Extract node features (atomic number, degree, etc)
- Create PyTorch Geometric Data objects
- 100% test coverage"

# Commit 2: GNN архитектура
git add src/models/gnn_model.py
git add tests/test_gnn_architecture.py
git commit -m "feat: implement Graph Convolutional Network
- 3 convolutional layers with ReLU activation
- Global mean pooling for graph-level representation
- Readout layer for IC50 regression
- Docstrings and type hints included"

# Commit 3: Обучение
git add src/models/gnn_training.py
git add notebooks/05_gnn_training.ipynb
git commit -m "feat: add GNN training pipeline
- DataLoader setup with batching
- Loss function (MSELoss) and optimizer (Adam)
- Early stopping and cross-validation
- Training notebook with examples"

# Commit 4: Оценка
git add src/evaluation/gnn_metrics.py
git commit -m "feat: add GNN evaluation metrics
- ROC-AUC calculation
- Confusion matrix
- Feature importance from attention
- Visualization functions"

# 4. Убедиться что всё работает
pytest tests/test_gnn_*.py -v --cov=src/models

# 5. Форматировать код
black src/
flake8 src/
mypy src/

# 6. Запушить ветку
git push origin feature/5_gnn_implementation

# 7. Создать Pull Request на GitHub
# Заполнить template
# Request review из advisor/collaborator

# 8. После одобрения: merge в dev
# (в GitHub веб-интерфейсе)

# 9. Обновить локальный dev
git checkout dev
git pull origin dev

# 10. Удалить feature ветку
git branch -d feature/5_gnn_implementation
git push origin --delete feature/5_gnn_implementation

# 11. После завершения месяца: merge dev → main и создать release
git checkout main
git pull origin main
git merge dev
git tag -a v0.3.0 -m "Release v0.3.0: GNN model implementation"
git push origin main v0.3.0
```

---

## ЧАСТЬ 10: ИНСТРУМЕНТЫ

### GitHub Desktop (если не хотите command line)

```
1. Скачать: desktop.github.com
2. Аутентификация с GitHub
3. Clone repository
4. Переключение между ветками (GUI)
5. Commit с интерфейсом
6. Push/pull кнопками
```

### VS Code Extensions

```
- GitHub Pull Requests and Issues
- GitLens (просмотр истории)
- Python (Pylance)
- Jupyter
```

### Команды для быстрого старта

```bash
# День 1: Setup
git clone https://github.com/YOUR_USERNAME/tb_drug_discovery.git
cd tb_drug_discovery
git checkout -b feature/1_qsar_baseline

# Вся неделя: Regular workflow
git status
git add .
git commit -m "feat: [description]"
git push origin feature/1_qsar_baseline

# Конец недели: Pull Request
# Создать на GitHub веб-интерфейсе

# После одобрения: Merge и cleanup
git checkout dev
git pull origin dev
git branch -d feature/1_qsar_baseline
```

---

## ЧАСТЬ 11: TROUBLESHOOTING

### Отменить последний commit (не pushed)
```bash
git reset --soft HEAD~1
# Коммит отменен, но изменения сохранены
```

### Отменить последний commit (уже pushed)
```bash
git revert HEAD
# Создает новый commit, который отменяет предыдущий
```

### Переименовать последний commit message
```bash
git commit --amend -m "new message"
git push origin feature/branch_name --force-with-lease
```

### Слить несколько коммитов (squash)
```bash
git rebase -i HEAD~3
# Выбрать 'squash' для 2-го и 3-го коммита
```

### Синхронизировать с удаленной веткой
```bash
git fetch origin
git rebase origin/dev
# Или просто merge (safe):
git merge origin/dev
```

---

## ФИНАЛЬНЫЙ ЧЕКЛИСТ

**Перед каждым commits:**
- [ ] Код отформатирован (`black`, `flake8`)
- [ ] Тесты passing (`pytest -v`)
- [ ] Type hints добавлены (`mypy`)
- [ ] Docstrings написаны
- [ ] Commit message informative
- [ ] Нет credentials в коде

**Перед каждым Pull Request:**
- [ ] Branch updated с dev
- [ ] Все тесты passing в CI/CD
- [ ] Code review requested
- [ ] PR description заполнена
- [ ] Related issue linked

**Перед каждым Release:**
- [ ] Все tests passing
- [ ] Documentation updated
- [ ] Version bumped
- [ ] CHANGELOG updated
- [ ] Tag created и pushed
- [ ] Release notes на GitHub

---

**Версия:** 2.0  
**Дата:** Декабрь 2025  
**Статус:** Ready for PhD project
