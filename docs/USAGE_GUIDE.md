# Полное руководство по использованию

## Содержание

1. [Установка](#установка)
2. [Phase 1: QSAR модель](#phase-1-qsar-модель)
3. [Phase 2: Молекулярный докинг](#phase-2-молекулярный-докинг)
4. [Команды консоли](#команды-консоли)
5. [Python API](#python-api)
6. [Jupyter Notebooks](#jupyter-notebooks)

---

## Установка

### Быстрая установка

```bash
# Клонирование репозитория
git clone https://github.com/yhtyyar/tb_drug_discovery.git
cd tb_drug_discovery

# Создание виртуального окружения
conda create -n tb_discovery python=3.10 -y
conda activate tb_discovery

# Установка зависимостей
pip install -r requirements.txt

# Установка пакета в режиме разработки
pip install -e .
```

### Проверка установки

```bash
# Проверить RDKit
python -c "from rdkit import Chem; print('✅ RDKit работает')"

# Проверить scikit-learn
python -c "from sklearn.ensemble import RandomForestClassifier; print('✅ scikit-learn работает')"

# Запустить тесты
pytest tests/ -v
```

---

## Phase 1: QSAR модель

### 1. Скачивание данных ChEMBL

```bash
# Скачать данные для InhA (CHEMBL1849)
python scripts/download_data.py

# Скачать для другой мишени
python scripts/download_data.py --target CHEMBL1234 --output data/raw/

# Опции
python scripts/download_data.py --help
```

**Результат:** `data/raw/chembl_1849_ic50.csv`

### 2. Обучение QSAR модели

#### Через консоль:

```bash
# Базовое обучение (classification)
python scripts/train_qsar.py --data data/raw/chembl_1849_ic50.csv

# Обучение для регрессии
python scripts/train_qsar.py --data data/raw/chembl_1849_ic50.csv --task regression

# С изменёнными параметрами
python scripts/train_qsar.py \
    --data data/raw/chembl_1849_ic50.csv \
    --task classification \
    --threshold 6.5 \
    --n-folds 10 \
    --seed 123 \
    --output models/custom_model
```

#### Все параметры train_qsar.py:

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--data` | Путь к CSV файлу с данными | **обязательный** |
| `--task` | `classification` или `regression` | `classification` |
| `--threshold` | Порог активности (pIC50) | `6.0` |
| `--n-folds` | Количество фолдов для CV | `5` |
| `--seed` | Random seed | `42` |
| `--output` | Директория для сохранения | `models/` |

### 3. Результаты обучения

После обучения создаются файлы:

```
models/
├── qsar_rf_model.pkl      # Обученная модель
├── qsar_scaler.pkl        # Скейлер признаков
├── qsar_metrics.json      # Метрики качества
└── feature_importance.csv # Важность признаков

data/processed/
└── cleaned_chembl_inhA.csv # Очищенные данные
```

### 4. Использование обученной модели

```python
from src.models import QSARModel
from src.data import DescriptorCalculator, DataPreprocessor
import pandas as pd

# Загрузка модели
model = QSARModel.load("models/qsar_rf_model.pkl")
preprocessor = DataPreprocessor.load("models/qsar_scaler.pkl")

# Подготовка нового соединения
smiles = "CC(=O)Nc1ccc(O)cc1"  # Парацетамол
calculator = DescriptorCalculator()
descriptors = calculator.calculate(smiles)

# Масштабирование
X = preprocessor.transform([list(descriptors.values())])

# Предсказание
prediction = model.predict(X)
probability = model.predict_proba(X)

print(f"Предсказание: {'Активный' if prediction[0] == 1 else 'Неактивный'}")
print(f"Вероятность активности: {probability[0][1]:.2%}")
```

### 5. Batch предсказание

```python
import pandas as pd
from src.models import QSARModel
from src.data import DescriptorCalculator, DataPreprocessor

# Загрузка
model = QSARModel.load("models/qsar_rf_model.pkl")
preprocessor = DataPreprocessor.load("models/qsar_scaler.pkl")
calculator = DescriptorCalculator()

# Загрузка соединений
df = pd.read_csv("my_compounds.csv")

# Вычисление дескрипторов
df_desc = calculator.calculate_from_dataframe(df, smiles_col="smiles")

# Предсказание
feature_cols = calculator.descriptor_names
X = df_desc[feature_cols].values
X_scaled = preprocessor.transform(X)

predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)[:, 1]

# Сохранение результатов
df["prediction"] = predictions
df["probability"] = probabilities
df.to_csv("predictions.csv", index=False)
```

---

## Phase 2: Молекулярный докинг

### Требования

1. **AutoDock Vina**: https://vina.scripps.edu/downloads/
2. **Open Babel**: https://openbabel.org/docs/Installation/install.html

```bash
# Проверка установки
vina --version
obabel -V
```

### 1. Подготовка белка

```bash
# Автоматическая подготовка TB мишени
python -c "
from src.docking import prepare_tb_target
result = prepare_tb_target('InhA', output_dir='data/structures')
print(result)
"
```

#### Доступные TB мишени:

| Мишень | PDB ID | Описание |
|--------|--------|----------|
| InhA | 4TZK | Enoyl-ACP reductase |
| KatG | 1SJ2 | Catalase-peroxidase |
| DprE1 | 4FDO | Decaprenylphosphoryl-beta-D-ribose oxidase |
| MmpL3 | 6AJG | Mycolic acid transporter |

### 2. Запуск докинга

#### Через консоль:

```bash
# Докинг против InhA (автоматическая настройка)
python scripts/run_docking.py \
    --target InhA \
    --compounds data/processed/cleaned_chembl_inhA.csv

# Докинг с кастомным белком
python scripts/run_docking.py \
    --receptor my_protein.pdb \
    --center 10.5 20.3 15.7 \
    --size 25 25 25 \
    --compounds compounds.csv

# Высокая точность
python scripts/run_docking.py \
    --target InhA \
    --compounds compounds.csv \
    --exhaustiveness 32 \
    --num-modes 20
```

#### Параметры run_docking.py:

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--target` | TB мишень (InhA, KatG, DprE1, MmpL3) | - |
| `--receptor` | Путь к PDB файлу белка | - |
| `--center` | Центр докинг-бокса (X Y Z) | автоопределение |
| `--size` | Размер докинг-бокса | `25 25 25` |
| `--compounds` | CSV с SMILES | **обязательный** |
| `--exhaustiveness` | Тщательность поиска | `8` |
| `--num-modes` | Количество поз | `9` |
| `--output` | Путь для результатов | `results/docking/` |

### 3. Результаты докинга

```
results/docking/
├── docking_results.csv    # Все результаты
└── docking_summary.json   # Статистика
```

**Формат docking_results.csv:**

| Колонка | Описание |
|---------|----------|
| ligand_name | Имя соединения |
| smiles | SMILES |
| affinity | Энергия связывания (kcal/mol) |
| num_poses | Количество найденных поз |

### 4. Python API для докинга

```python
from src.docking import VinaDocker, ProteinPreparator

# Подготовка белка
prep = ProteinPreparator(work_dir="data/structures")
pdb_path = prep.download_pdb("4TZK")
clean_path = prep.clean_protein(pdb_path, keep_ligand="TCL")
center, size = prep.get_binding_site(clean_path, "TCL")

# Инициализация докера
docker = VinaDocker(exhaustiveness=16)
receptor_pdbqt = docker.prepare_receptor(clean_path)
docker.set_receptor(receptor_pdbqt, center, size)

# Докинг одного соединения
result = docker.dock_smiles("CC(=O)Nc1ccc(O)cc1", name="paracetamol")
print(f"Энергия связывания: {result.affinity} kcal/mol")

# Batch докинг
smiles_list = ["CCO", "CCCO", "CCCCO"]
results_df = docker.dock_batch(smiles_list)
print(results_df)

# Очистка временных файлов
docker.cleanup()
```

---

## Команды консоли

### Основные команды

```bash
# === PHASE 1: QSAR ===

# Скачать данные ChEMBL
python scripts/download_data.py

# Обучить QSAR модель
python scripts/train_qsar.py --data data/raw/chembl_1849_ic50.csv

# Запустить тесты
pytest tests/ -v

# Запустить тесты с покрытием
pytest tests/ --cov=src --cov-report=html

# === PHASE 2: DOCKING ===

# Докинг (требует Vina + OpenBabel)
python scripts/run_docking.py --target InhA --compounds compounds.csv

# === ОБЩИЕ ===

# Jupyter notebooks
jupyter notebook notebooks/

# Форматирование кода
black src/ tests/ scripts/
isort src/ tests/ scripts/

# Проверка типов
mypy src/
```

### Быстрые примеры

```bash
# Полный пайплайн от скачивания до предсказания
python scripts/download_data.py
python scripts/train_qsar.py --data data/raw/chembl_1849_ic50.csv
python -c "
from src.models import QSARModel
from src.data import DescriptorCalculator, DataPreprocessor

model = QSARModel.load('models/qsar_rf_model.pkl')
prep = DataPreprocessor.load('models/qsar_scaler.pkl')
calc = DescriptorCalculator()

smiles = 'c1ccc2c(c1)c(=O)n(c(=O)[nH]2)C'
desc = calc.calculate(smiles)
X = prep.transform([list(desc.values())])
prob = model.predict_proba(X)[0][1]
print(f'Вероятность активности: {prob:.1%}')
"
```

---

## Python API

### Модули

```python
# Data loading
from src.data import ChEMBLLoader, DescriptorCalculator, DataPreprocessor

# QSAR model
from src.models import QSARModel

# Evaluation
from src.evaluation import cross_validate_model, calculate_classification_metrics

# Docking
from src.docking import VinaDocker, ProteinPreparator, prepare_tb_target

# Utilities
from src.utils import Config, setup_logger
```

### Примеры использования

```python
# === Загрузка и предобработка данных ===
from src.data import ChEMBLLoader, DescriptorCalculator

loader = ChEMBLLoader()
df = loader.load("data/raw/chembl_1849_ic50.csv")
df = loader.preprocess(df)
df = loader.create_activity_labels(df, threshold=6.0)

calculator = DescriptorCalculator(lipinski=True, topological=True)
df_desc = calculator.calculate_from_dataframe(df, smiles_col="smiles")


# === Обучение модели ===
from src.models import QSARModel
from src.data import DataPreprocessor

X = df_desc[calculator.descriptor_names].values
y = df_desc["active"].values

preprocessor = DataPreprocessor(random_seed=42)
X_train, X_test, y_train, y_test = preprocessor.split_data_simple(X, y)
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

model = QSARModel(task="classification", n_estimators=100)
model.fit(X_train_scaled, y_train)
metrics = model.evaluate(X_test_scaled, y_test)
print(f"ROC-AUC: {metrics['roc_auc']:.4f}")


# === Cross-validation ===
from src.evaluation import cross_validate_model

cv_results = cross_validate_model(
    model.model, X_train_scaled, y_train,
    n_folds=5, task="classification"
)
print(f"CV ROC-AUC: {cv_results['roc_auc_mean']:.4f} ± {cv_results['roc_auc_std']:.4f}")


# === Сохранение и загрузка ===
model.save("models/my_model.pkl")
preprocessor.save("models/my_scaler.pkl")

loaded_model = QSARModel.load("models/my_model.pkl")
loaded_prep = DataPreprocessor.load("models/my_scaler.pkl")
```

---

## Jupyter Notebooks

### Доступные notebooks

| Notebook | Описание |
|----------|----------|
| `01_data_loading.ipynb` | Загрузка и очистка данных ChEMBL |
| `02_descriptor_calculation.ipynb` | Вычисление молекулярных дескрипторов |
| `03_qsar_training.ipynb` | Обучение и оценка QSAR модели |

### Запуск

```bash
# Запуск Jupyter
jupyter notebook notebooks/

# Или JupyterLab
jupyter lab notebooks/
```

### Порядок выполнения

1. Сначала `01_data_loading.ipynb` → создаёт `data/processed/cleaned_chembl_inhA.csv`
2. Затем `02_descriptor_calculation.ipynb` → создаёт `data/processed/descriptors.csv`
3. Наконец `03_qsar_training.ipynb` → обучает модель и сохраняет в `models/`

---

## Устранение неполадок

### RDKit не устанавливается

```bash
# Используйте conda
conda install -c conda-forge rdkit
```

### ModuleNotFoundError

```bash
# Установите пакет в режиме разработки
pip install -e .
```

### Vina не найден

1. Скачайте с https://vina.scripps.edu/downloads/
2. Добавьте в PATH или укажите полный путь:
```bash
python scripts/run_docking.py --vina-path /path/to/vina ...
```

### Ошибки памяти

- Уменьшите размер батча
- Используйте `--max-compounds 100` для тестирования
