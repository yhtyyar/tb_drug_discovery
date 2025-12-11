# ТЕХНИЧЕСКОЕ ЗАДАНИЕ (ТЗ)
## TB Drug Discovery: Machine Learning Pipeline Project

**Версия:** 1.0 (Production Ready)  
**Дата создания:** Декабрь 2025  
**Статус:** Утверждено  
**Разработчик:** Кандидат PhD (химия + ML)  
**Руководитель:** Prof. [Ваш руководитель]

---

## 1. ОБЗОР ПРОЕКТА

### 1.1 Название
**"Интегрированный pipeline машинного обучения для разработки противотуберкулезных препаратов с применением Graph Neural Networks и молекулярного дизайна"**

### 1.2 Цель
Разработать и валидировать машинное обучение pipeline для предсказания активности и de novo дизайна ингибиторов ферментов Mycobacterium tuberculosis (InhA, rpoB), интегрировав:
- QSAR моделирование
- Graph Neural Networks (GNN)
- Молекулярный дозинг
- Генеративные модели (VAE, Diffusion)
- AlphaFold 3 структурное предсказание

### 1.3 Значимость
- **Научная:** Интеграция органической химии + современного ML (2024-2025 методы)
- **Медицинская:** Разработка препаратов против MDR-TB/XDR-TB (резистентные штаммы)
- **Практическая:** Минимум 3 публикации в топ-20 журналах + PhD диссертация

### 1.4 Целевые метрики успеха
- [ ] **QSAR модель:** ROC-AUC > 0.75 на test set
- [ ] **GNN модель:** ROC-AUC > 0.80 (лучше QSAR)
- [ ] **Синтезировано молекул:** >= 15-20 соединений
- [ ] **Экспериментальная активность:** Хотя бы 1-2 соединения с IC50 < 10 μM
- [ ] **Публикации:** 2-3 статьи в журналах IF > 5
- [ ] **Код качество:** 100% тестовое покрытие критических функций
- [ ] **GitHub:** All commits pushed, CI/CD passing

---

## 2. ТРЕБОВАНИЯ К ФУНКЦИОНАЛЬНОСТИ

### 2.1 Модуль 1: Data Pipeline & QSAR (Месяцы 1-2)

#### Требования:
- [x] **Загрузка данных:**
  - [ ] Парсинг ChEMBL API (или CSV)
  - [ ] Фильтрация: только InhA ингибиторы Mtb
  - [ ] Стандартизация SMILES (удаление duplicate)
  - [ ] Валидация структур (RDKit санитизация)
  - [ ] Результат: чистый датасет >= 500 соединений

- [x] **Расчет дескрипторов (RDKit):**
  - [ ] Липинский дескрипторы: MW, LogP, TPSA, HBA, HBD
  - [ ] Топологические: RotBonds, RingCount, AromaticRings
  - [ ] Расширенные: MolFormula, BPM, LabuteASA
  - [ ] Output: pandas DataFrame (соединение × дескриптор)

- [x] **QSAR модель базовая:**
  - [ ] Алгоритм: Random Forest Regressor (n_estimators=100)
  - [ ] Targets: Log(IC50) или pIC50
  - [ ] Split: Train (70%) / Val (15%) / Test (15%)
  - [ ] Метрики: R², ROC-AUC, RMSE, MAE
  - [ ] Целевой результат: R² > 0.65, ROC-AUC > 0.75
  - [ ] Валидация: 5-fold cross-validation

- [x] **Vis & Analysis:**
  - [ ] ROC кривая (matplotlib/plotly)
  - [ ] Scatter plot (pred vs actual)
  - [ ] Feature importance (top 10 дескрипторов)
  - [ ] Residual анализ (для outliers)

#### Deliverables:
```
data/
├── raw/
│   └── chembl_inhA.csv (оригинальные данные)
├── processed/
│   ├── chembl_inhA_clean.csv
│   ├── descriptors.csv
│   └── train_test_split.pkl
models/
├── qsar_rf_model.pkl
├── scaler.pkl
└── qsar_metrics.json
notebooks/
├── 01_data_loading.ipynb
├── 02_descriptor_calculation.ipynb
└── 03_qsar_training.ipynb
plots/
├── qsar_roc.png
├── scatter_pred_vs_actual.png
└── feature_importance.png
```

---

### 2.2 Модуль 2: Молекулярный дозинг (Месяцы 2-4, параллельно)

#### Требования:
- [x] **Подготовка структур белков:**
  - [ ] Загрузить PDB структуры мишеней (InhA: 1ENX, rpoB: 6F0M)
  - [ ] Удалить лиганды, молекулы воды
  - [ ] Добавить водороды (MolKit или программно)
  - [ ] Параметризация (PDBQT формат)
  - [ ] Результат: cleaned_protein.pdbqt

- [x] **Подготовка лигандов:**
  - [ ] SMILES → 3D конформер (RDKit.AllChem.EmbedMolecule)
  - [ ] Минимизация энергии (MMFF94, 500 steps)
  - [ ] Конвертация в PDBQT (Meeko)
  - [ ] Batch processing (параллельный)

- [x] **Молекулярный дозинг (AutoDock Vina):**
  - [ ] Определение binding pocket (CarveOut или вручную)
  - [ ] Дозинг параметры: exhaust=32, num_modes=20
  - [ ] Запуск для всех лигандов (100+ молекул)
  - [ ] Сбор результатов: docking_score, RMSD, binding_mode

- [x] **Анализ результатов:**
  - [ ] Корреляция docking_score vs экспериментальные IC50
  - [ ] Целевой результат: R² > 0.5-0.6
  - [ ] Визуализация комплексов (PyMOL)
  - [ ] SAR анализ (какие взаимодействия важны)

#### Deliverables:
```
data/
├── structures/
│   ├── 1ENX_cleaned.pdbqt
│   ├── 1ENX_binding_pocket.txt
│   └── ligands/
│       ├── mol_001.pdbqt
│       ├── mol_002.pdbqt
│       └── ...
results/
├── docking_scores.csv
├── docking_vs_ic50_correlation.png
└── top_10_poses/
    ├── mol_001_pose.pdb
    └── ...
```

---

### 2.3 Модуль 3: Graph Neural Networks (Месяцы 4-8)

#### Требования:
- [x] **Молекулярное представление (RDKit):**
  - [ ] SMILES → Граф (атомы = узлы, связи = ребра)
  - [ ] Node features: atomic number, degree, charge, aromaticity
  - [ ] Edge features (опционально): bond type, is_aromatic
  - [ ] Формат: PyTorch Geometric Data objects

- [x] **GNN архитектура:**
  - [ ] Архитектура: Graph Convolutional Network (GCN) или Message Passing NN (MPNN)
  - [ ] Слои: 3-4 convolutional layers + global pooling + 1-2 linear layers
  - [ ] Activation: ReLU
  - [ ] Pooling: global_mean_pool или global_add_pool
  - [ ] Output: регрессия (IC50) или классификация (active/inactive)

- [x] **Обучение:**
  - [ ] Dataset: PyTorch Geometric DataLoader (batch_size=32-64)
  - [ ] Train/Val/Test: 70/15/15 split
  - [ ] Loss function: MSELoss (для регрессии) или CrossEntropyLoss (классификация)
  - [ ] Optimizer: Adam (lr=0.001, weight_decay=1e-5)
  - [ ] Epochs: 50-100 (с early stopping)
  - [ ] Hardware: Google Colab T4 GPU (sufficient)

- [x] **Метрики:**
  - [ ] Регрессия: R², RMSE, MAE
  - [ ] Классификация: ROC-AUC, Precision, Recall, F1
  - [ ] Целевой результат: ROC-AUC > 0.80
  - [ ] Кросс-валидация: 5-fold CV (опционально)

- [x] **Интерпретируемость:**
  - [ ] Attention weights (если используется GAT)
  - [ ] Saliency maps (какие атомы важны)
  - [ ] SHAP значения (для интерпретации)

#### Deliverables:
```
models/
├── gnn_model_v1.pt (trained model)
├── gnn_config.yaml (архитектура)
├── gnn_metrics.json (ROC-AUC, R², etc)
└── gnn_scaler.pkl (для нормализации)
notebooks/
├── 04_gnn_implementation.ipynb
├── 05_gnn_training.ipynb
└── 06_gnn_evaluation.ipynb
plots/
├── gnn_roc_curve.png
├── gnn_loss_curves.png
├── attention_weights.png
└── saliency_maps.pdf
```

---

### 2.4 Модуль 4: Молекулярный дизайн (Месяцы 7-12)

#### Требования:
- [x] **VAE для молекулярной генерации:**
  - [ ] Использовать pre-trained D-MolVAE или собственное обучение
  - [ ] SMILES кодирование (latent space dimension: 64-256)
  - [ ] Генерация: sampling из latent space, декодирование в SMILES
  - [ ] Валидность: >= 90% generated SMILES должны быть valid

- [x] **Diffusion Models (новейший метод):**
  - [ ] Фреймворк: Hugging Face Diffusers или custom PyTorch
  - [ ] Модель: PMDM или DiffSMol (молекулярная генерация)
  - [ ] Обучение (опционально): transfer learning от pre-trained
  - [ ] Генерация: 100-500 молекул

- [x] **Reinforcement Learning (опционально):**
  - [ ] Policy: Guided generation (reward function)
  - [ ] Reward: GNN score + ADME свойства + синтезируемость
  - [ ] Итеративная оптимизация (200-300 итераций)

- [x] **Отбор кандидатов:**
  - [ ] Фильтрация: Липинский правило 5 + TPSA 20-130
  - [ ] Ранжирование: GNN score + docking score + ADME
  - [ ] Финальный список: top 10-20 кандидатов для синтеза
  - [ ] Таблица: молекула | GNN | Docking | MW | LogP | TPSA | ...

#### Deliverables:
```
models/
├── vae_model.pt
├── diffusion_model.pt
└── generated_molecules.csv (1000 молекул)
results/
├── top_20_candidates.csv
├── property_distribution_plots.png
├── vs_known_drugs_comparison.png
└── synthesis_recommendations.txt
```

---

### 2.5 Модуль 5: AlphaFold 3 & MD валидация (Месяцы 9-12)

#### Требования:
- [x] **AlphaFold 3 структурное предсказание:**
  - [ ] Использовать: alphafoldserver.com (веб) или локальная версия
  - [ ] Input: InhA структура + topN лигандов из генерации
  - [ ] Output: protein-ligand комплекс структуры (PDB)
  - [ ] Сравнение с молекулярным дозингом (RMSD)

- [x] **Молекулярная динамика (MD) валидация:**
  - [ ] Инструмент: OpenMM или GROMACS
  - [ ] Для top 3-5 комплексов:
    - [ ] Energy minimization (500 steps)
    - [ ] Heating 0K → 310K (2000 steps)
    - [ ] Production MD (100 ps)
  - [ ] Анализ: RMSD, H-bonds stability, лиганд остается в кармане?

#### Deliverables:
```
structures/
├── alphafold_predictions/
│   ├── inhA_ligand001_complex.pdb
│   ├── inhA_ligand002_complex.pdb
│   └── ...
md_trajectories/
├── ligand001_md_trajectory.xtc
├── ligand001_md_analysis.csv
└── stability_plots.png
```

---

### 2.6 Модуль 6: Синтез и экспериментальная валидация (Parallel)

#### Требования:
- [x] **Синтез соединений:**
  - [ ] Синтезировать 15-20 отобранных молекул
  - [ ] Стандартная органическая химия (>60% выход)
  - [ ] Спектральная характеризация (ЯМР, МС)
  - [ ] Чистота >= 95% (HPLC)
  - [ ] Таблица: соединение | синтез_схема | выход | точка_плавления | спектры

- [x] **Экспериментальное тестирование:**
  - [ ] IC50 assay против InхА (если лаборатория есть)
  - [ ] Цитотоксичность на HEK293 (если есть)
  - [ ] Активность против резистентных штаммов (если есть)
  - [ ] Результат: таблица экспериментальных активностей

- [x] **Обновление ML моделей:**
  - [ ] Добавить новые экспериментальные данные в датасет
  - [ ] Переобучить GNN (with new data)
  - [ ] Проверить улучшилась ли точность
  - [ ] Итеративный процесс (синтез → тест → update model)

#### Deliverables:
```
chemistry/
├── synthesis_protocols/
│   ├── compound_001_synthesis.pdf
│   ├── compound_002_synthesis.pdf
│   └── ...
├── spectra/
│   ├── compound_001_NMR.pdf
│   ├── compound_001_MS.pdf
│   └── ...
└── experimental_data.csv (IC50, purity, etc)
```

---

## 3. АРХИТЕКТУРА ПРОЕКТА

### 3.1 Структура репозитория (GitHub)

```
tb_drug_discovery/
│
├── README.md (overview)
├── CONTRIBUTING.md (contribution guidelines)
├── LICENSE (MIT)
├── .gitignore
├── requirements.txt (Python dependencies)
├── setup.py (для pip install)
│
├── docs/
│   ├── ARCHITECTURE.md (система проекта)
│   ├── SETUP.md (how to setup)
│   ├── API.md (code documentation)
│   ├── PUBLICATIONS.md (статьи)
│   └── images/
│       └── workflow_diagram.png
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── chembl_loader.py (загрузка данных)
│   │   ├── descriptor_calculator.py (QSAR дескрипторы)
│   │   └── data_preprocessor.py (очистка, валидация)
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── qsar_model.py (Random Forest QSAR)
│   │   ├── gnn_model.py (Graph Neural Network)
│   │   ├── vae_model.py (Variational AutoEncoder)
│   │   └── diffusion_model.py (Diffusion Models)
│   │
│   ├── docking/
│   │   ├── __init__.py
│   │   ├── prepare_protein.py (подготовка PDB)
│   │   ├── prepare_ligand.py (подготовка лигандов)
│   │   ├── vina_runner.py (запуск AutoDock Vina)
│   │   └── docking_analysis.py (анализ результатов)
│   │
│   ├── alphafold/
│   │   ├── __init__.py
│   │   ├── alphafold_runner.py (запуск AF3)
│   │   └── structure_analysis.py (анализ структур)
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py (ROC-AUC, R², и т.д.)
│   │   ├── cross_validation.py (CV логика)
│   │   └── interpretability.py (SHAP, attention)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py (конфигурация)
│       ├── logger.py (логирование)
│       └── file_handlers.py (работа с файлами)
│
├── notebooks/
│   ├── 01_data_loading.ipynb (QSAR prep)
│   ├── 02_descriptor_calculation.ipynb
│   ├── 03_qsar_training.ipynb
│   ├── 04_gnn_implementation.ipynb
│   ├── 05_gnn_training.ipynb
│   ├── 06_gnn_evaluation.ipynb
│   ├── 07_docking_analysis.ipynb
│   ├── 08_molecular_generation.ipynb
│   ├── 09_alphafold_validation.ipynb
│   └── 10_full_pipeline.ipynb
│
├── data/
│   ├── raw/
│   │   ├── chembl_inhA.csv
│   │   └── structures/
│   │       ├── 1ENX.pdb (InhA)
│   │       └── 6F0M.pdb (rpoB)
│   │
│   ├── processed/
│   │   ├── cleaned_molecules.csv
│   │   ├── descriptors.csv
│   │   ├── generated_molecules.csv
│   │   └── experimental_results.csv
│   │
│   └── external/
│       └── literature_references.bib
│
├── models/
│   ├── qsar_rf_model.pkl
│   ├── qsar_scaler.pkl
│   ├── gnn_model_v1.pt
│   ├── gnn_config.yaml
│   ├── vae_model.pt
│   └── diffusion_model.pt
│
├── results/
│   ├── metrics/
│   │   ├── qsar_metrics.json
│   │   ├── gnn_metrics.json
│   │   └── ...
│   │
│   ├── figures/
│   │   ├── qsar_roc.png
│   │   ├── gnn_roc.png
│   │   ├── scatter_pred_vs_actual.png
│   │   ├── feature_importance.png
│   │   ├── docking_analysis.png
│   │   └── ...
│   │
│   └── reports/
│       ├── project_summary.md
│       ├── monthly_progress.md
│       └── final_report.pdf
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_qsar_model.py
│   ├── test_gnn_model.py
│   ├── test_docking.py
│   └── test_integration.py
│
├── scripts/
│   ├── setup_environment.sh
│   ├── download_data.py
│   ├── train_qsar.py
│   ├── train_gnn.py
│   ├── run_docking.py
│   ├── generate_molecules.py
│   └── run_pipeline.py (full pipeline)
│
├── .github/
│   ├── workflows/
│   │   ├── pytest.yml (unit tests)
│   │   ├── code_quality.yml (black, flake8)
│   │   └── docs.yml (documentation build)
│   │
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
│
└── config/
    ├── default_config.yaml
    ├── gnn_config.yaml
    ├── docking_config.yaml
    └── credentials.example.yaml (for cloud APIs)
```

---

## 4. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ

### 4.1 Stack технологий

```yaml
Language:
  Python: 3.10+
  
Core Libraries:
  RDKit: >= 2024.03 (молекулярная химия)
  PyTorch: >= 2.0 (deep learning)
  PyTorch Geometric: >= 2.4 (GNN)
  scikit-learn: >= 1.3 (классический ML)
  pandas: >= 2.0 (data manipulation)
  numpy: >= 1.24 (numerical computing)
  
ML/AI:
  DeepChem: >= 4.0 (drug discovery)
  Hugging Face Transformers: >= 4.30 (LLM)
  Diffusers: >= 0.21 (diffusion models)
  
Molecular Dynamics:
  OpenMM: >= 8.0 (MD simulations)
  MDTraj: >= 1.9 (trajectory analysis)
  
Visualization:
  Matplotlib: >= 3.7
  Plotly: >= 5.14
  Seaborn: >= 0.12
  Pymol: >= 3.0 (3D structures)
  
Utilities:
  Jupyter: >= 7.0 (notebooks)
  tqdm: >= 4.65 (progress bars)
  pyyaml: >= 6.0 (config files)
  requests: >= 2.31 (API calls)
  
Testing:
  pytest: >= 7.4
  pytest-cov: >= 4.1 (coverage)
  black: >= 23.0 (code formatting)
  flake8: >= 6.0 (linting)
  mypy: >= 1.5 (type checking)
  
Documentation:
  Sphinx: >= 7.0
  sphinx-rtd-theme: >= 1.3
  
Cloud/DevOps:
  AWS SDK (boto3): >= 1.28
  Google Cloud Client: >= 2.45
  Docker: >= 24.0 (containers, опционально)
  GitHub Actions: (CI/CD)
```

### 4.2 Версионирование

```
v0.1 (Месяц 1-2)   - QSAR базовая + данные
v0.2 (Месяц 3-4)   - Дозинг + синтез
v0.3 (Месяц 5-6)   - GNN обучение
v0.4 (Месяц 7-8)   - Молекулярная генерация
v1.0 (Месяц 9-10)  - Полный pipeline (Publication 1)
v1.1 (Месяц 11-12) - GNN улучшения + новые данные
v2.0 (Месяц 13-18) - Трансформеры + Diffusion (Publication 2)
v3.0 (Месяц 19-24) - AlphaFold интеграция + финал (Publication 3)
```

---

## 5. ПРОЦЕСС РАЗРАБОТКИ

### 5.1 Git workflow

```
Main branch (production):
  ├── Только release commits
  ├── Все тесты passing
  └── Версии (v1.0, v2.0, и т.д.)

Dev branch (development):
  ├── Ежедневные commits
  ├── Может быть нестабильно
  └── Merges в main при готовности

Feature branches:
  ├── feature/qsar_baseline
  ├── feature/gnn_architecture
  ├── feature/molecular_generation
  ├── feature/alphafold_integration
  ├── fix/bug_in_descriptor_calc
  └── docs/readme_update

Commit Message Format:
  feat: добавить GNN архитектуру
  fix: исправить баг в QSAR валидации
  docs: обновить README
  test: добавить unit tests для docking
  refactor: переструктурировать data loader
  chore: обновить requirements.txt
```

### 5.2 Testing & Quality

```
Unit Tests (pytest):
  ✅ Все functions имеют tests
  ✅ Coverage >= 80% (критичные функции 100%)
  ✅ Runs locally перед commits
  
Code Quality:
  ✅ black --line-length=100 (formatting)
  ✅ flake8 (linting)
  ✅ mypy (type checking)
  ✅ isort (import sorting)
  
CI/CD (GitHub Actions):
  ✅ Auto-run tests на каждый push
  ✅ Auto-run code quality checks
  ✅ Build documentation
  ✅ Approve merge при passing всех checks
```

### 5.3 Документирование

```
README.md:
  ├── Project description
  ├── Quick start
  ├── Installation
  ├── Usage examples
  ├── Results summary
  └── References

Docstrings (Google style):
  ├── Все функции имеют docstrings
  ├── Все классы документированы
  ├── Examples включены
  └── Type hints обязательны

Notebooks:
  ├── Markdown cells объясняют логику
  ├── Code хорошо аннотирован
  ├── Результаты и графики показаны
  └── Execution reproducible

Blogs/Articles:
  └── Monthly progress blogs (Medium или Substack)
```

---

## 6. ВРЕМЕННОЙ ПЛАН

### 6.1 Гант-диаграмма (36 месяцев PhD)

```
МЕСЯЦЫ:    1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36

Модуль 1   [===================]                       (QSAR + синтез)
Модуль 2   [===========================]                (Дозинг + синтез 2)
Модуль 3   [================================]           (GNN обучение + валидация)
Модуль 4   [========================================]    (Молекулярная генерация)
Модуль 5   [===========================]                (AlphaFold + MD)
Модуль 6   [===================================]         (Синтез + экспресс)

Publication #1        [=====]                           (Месяцы 10-11)
Publication #2                            [=====]       (Месяцы 18-19)
Publication #3                                    [=====] (Месяцы 24-25)

Dissertation                                                        [=====] (Месяцы 31-33)
Defense                                                                   [=] (Месяц 34-35)
```

### 6.2 Key Milestones

| Дата | Milestone | Deliverable |
|------|-----------|------------|
| Неделя 1 | Project kickoff | GitHub repo, Trello board |
| Неделя 4 | Data loading complete | cleaned_chembl_inhA.csv |
| Неделя 8 | QSAR baseline ready | QSAR model (R² > 0.75) |
| Неделя 12 | Docking pipeline | docking_scores.csv |
| Месяц 3 | First synthesis batch | 5-7 compounds characterized |
| Месяц 5 | GNN model trained | GNN model (ROC-AUC > 0.80) |
| Месяц 7 | Generation pipeline | top_20_candidates.csv |
| Месяц 9 | AlphaFold predictions | AF3 structures |
| Месяц 10 | Publication #1 submitted | Paper on arXiv/journal |
| Месяц 12 | Second synthesis batch | 5-10 more compounds |
| Месяц 18 | Publication #2 submitted | Paper (GNN focus) |
| Месяц 24 | Publication #3 submitted | Paper (integrated pipeline) |
| Месяц 30 | Dissertation draft | Complete thesis |
| Месяц 34 | PhD defense | ✅ Defended successfully |

---

## 7. КРИТЕРИИ ПРИЕМКИ

### 7.1 Функциональные требования

- [x] **QSAR модель**
  - [ ] Обучена на >= 500 соединениях
  - [ ] ROC-AUC >= 0.75 на test set
  - [ ] Код полностью задокументирован

- [x] **GNN модель**
  - [ ] Архитектура выбрана и обоснована
  - [ ] ROC-AUC >= 0.80 (лучше чем QSAR)
  - [ ] Интерпретируемость реализована (attention/saliency)
  - [ ] Tests passing

- [x] **Молекулярный дозинг**
  - [ ] AutoDock Vina настроен правильно
  - [ ] Результаты коррелируют с активностью (R² > 0.5)
  - [ ] Visualization готовых комплексов

- [x] **Молекулярный дизайн**
  - [ ] Сгенерировано >= 1000 молекул
  - [ ] Валидность SMILES >= 90%
  - [ ] Top 20 кандидатов отобраны по скорингу

- [x] **Синтез**
  - [ ] >= 15 соединений синтезировано
  - [ ] Чистота >= 95%
  - [ ] Экспериментальные данные собраны

- [x] **Публикации**
  - [ ] 3+ статьи submitted/accepted
  - [ ] IF журналов >= 4.0 (в среднем)
  - [ ] Все коды open-source на GitHub

### 7.2 Non-Functional требования

- [x] **Код качество**
  - [ ] 100% прохождение pytest
  - [ ] Code coverage >= 80%
  - [ ] Zero black/flake8 errors
  - [ ] Type hints везде

- [x] **Документация**
  - [ ] README полный и clear
  - [ ] Все functions имеют docstrings
  - [ ] API документ сгенерирован (Sphinx)
  - [ ] 5+ tutorials/examples

- [x] **Performance**
  - [ ] QSAR < 1 мин на 1000 молекул
  - [ ] GNN < 2 мин на 5000 молекул
  - [ ] Docking 100 молекул < 1 час
  - [ ] Inference GPU < 10 сек

- [x] **Reproducibility**
  - [ ] Random seeds зафиксированы
  - [ ] requirements.txt точен
  - [ ] Docker container (опционально)
  - [ ] All results reproducible

---

## 8. УПРАВЛЕНИЕ РИСКАМИ

### 8.1 Potential Issues

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| GNN не сходится | High | Medium | Туне гиперпараметры, попробовать другую архитектуру |
| Молекулы синтезируются плохо | Medium | High | Use только predicted молекулы с high synthesis score |
| Экспериментальная активность низкая | Medium | High | Iterate ML модели, добавлять данные |
| Нет доступа к GPU | Medium | High | Use Google Colab Pro (€10/мес) |
| Публикации rejected | Low | Medium | Улучшить методологию, попробовать другие журналы |

### 8.2 Contingency Plans

- GPU fails → AWS cloud machine (pay-per-hour)
- Experiments fail → Larger virtual screening (in silico only)
- Publication rejected → Split на 2-3 smaller papers
- Time running out → Focus на GNN + docking (main contributions)

---

## 9. CONSTRAINTS & ASSUMPTIONS

### 9.1 Constraints
- Бюджет: €10-15k (максимум за 3 года)
- Time: 36 месяцев PhD программа
- Лаборатория: Может быть доступна не всегда
- GPU: T4 на Colab достаточна (можно использовать)

### 9.2 Assumptions
- ChEMBL данные доступны (public)
- PDB структуры доступны (public)
- Python environment можно настроить
- Синтез осуществим (структуры not too complex)
- Publications можно опубликовать (новая область)

---

## 10. SIGN-OFF

**Подготовлено:** [Ваше имя]  
**Дата:** Декабрь 2025  
**Статус:** ✅ Approved for implementation  

**Утверждено:** Prof. [Руководитель]  
**Дата:** [День месяц год]  
**Подпись:** _________________

---

**Версия:** 1.0  
**Последнее обновление:** Декабрь 2025  
**Следующее review:** Месяц 3 PhD программы
