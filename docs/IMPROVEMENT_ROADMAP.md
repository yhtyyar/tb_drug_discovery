# TB Drug Discovery — Полный план улучшений

> Документ составлен на основе глубокого анализа кодовой базы.
> Каждый пункт привязан к конкретному файлу и содержит рабочий пример кода.
> Статус: 🔴 Не сделано | 🟡 Частично | 🟢 Сделано

---

## Содержание

1. [Критические исправления (неделя 1–2)](#1-критические-исправления)
2. [Тестирование и валидация (неделя 2–3)](#2-тестирование-и-валидация)
3. [Улучшение моделей (месяц 1–2)](#3-улучшение-моделей)
4. [MLOps и воспроизводимость (месяц 1–2)](#4-mlops-и-воспроизводимость)
5. [Production API и мониторинг (месяц 2–3)](#5-production-api-и-мониторинг)
6. [Расширение пайплайна (месяц 2–4)](#6-расширение-пайплайна)
7. [Документация и коллаборация (постоянно)](#7-документация-и-коллаборация)

---

## 1. Критические исправления

### 1.1 🟢 Scaffold Split для QSAR (СДЕЛАНО)

**Файл:** `src/data/scaffold_split.py` — создан  
**Проблема:** `data_preprocessor.py` использует случайное разбиение, что завышает метрики на 5–15% AUC.  
При случайном split структурно похожие молекулы попадают и в train, и в test, что даёт модели "подсказки".

**Что сделано:**
- `scaffold_split()` — разбивка индексов по scaffolds
- `scaffold_split_df()` — разбивка DataFrame
- `scaffold_k_fold()` — scaffold-aware k-fold CV

**Шаг 1: Обновить `scripts/train_qsar.py`**

```python
# БЫЛО (завышает метрики):
from src.data.data_preprocessor import DataPreprocessor
prep = DataPreprocessor(random_seed=42)
X_train, X_val, X_test, y_train, y_val, y_test = prep.split_data(X, y)

# ДОЛЖНО БЫТЬ:
from src.data.scaffold_split import scaffold_split_df
train_df, val_df, test_df = scaffold_split_df(
    df,
    smiles_col="smiles",
    frac_train=0.70,
    frac_val=0.15,
    frac_test=0.15,
    random_seed=42,
)
X_train = descriptor_matrix[train_df.index]
X_val   = descriptor_matrix[val_df.index]
X_test  = descriptor_matrix[test_df.index]
y_train, y_val, y_test = train_df["pIC50"], val_df["pIC50"], test_df["pIC50"]
```

**Шаг 2: Обновить `src/evaluation/cross_validation.py` — добавить scaffold k-fold**

```python
# В функцию cross_validate_model добавить параметр:
def cross_validate_model(
    model, X, y,
    smiles_list=None,        # <-- добавить
    use_scaffold_split=True, # <-- добавить
    n_folds=5, ...
):
    if use_scaffold_split and smiles_list is not None:
        from data.scaffold_split import scaffold_k_fold
        splits = scaffold_k_fold(smiles_list, n_folds=n_folds, random_seed=random_seed)
    else:
        kfold = KFold(n_splits=n_folds, ...)
        splits = list(kfold.split(X, y))
```

**Шаг 3: Пересчитать и задокументировать метрики**
```bash
# Запустить переобучение и сравнить метрики:
python scripts/train_qsar.py --split scaffold --log-results results/scaffold_vs_random.json
```

---

### 1.2 🔴 KL Annealing в VAE (КРИТИЧНО)

**Файл:** `src/generation/vae.py`, `scripts/train_vae.py`  
**Проблема:** Без KL annealing VAE страдает от posterior collapse — decoder игнорирует латентное пространство и генерирует одинаковые молекулы. KL loss в начале обучения "давит" на encoder, заставляя его обнулять информацию.

**Шаг 1: Добавить класс `KLAnnealer`**

```python
# src/generation/vae.py — добавить класс:
class KLAnnealer:
    """Linear or cyclical KL weight schedule."""

    def __init__(self, n_epochs: int, start: float = 0.0, end: float = 1.0,
                 strategy: str = "linear"):
        self.n_epochs = n_epochs
        self.start = start
        self.end = end
        self.strategy = strategy

    def get_weight(self, epoch: int) -> float:
        if self.strategy == "linear":
            return min(self.end, self.start + (self.end - self.start) * epoch / self.n_epochs)
        elif self.strategy == "cyclical":
            # 4 циклов на всё обучение
            cycle = self.n_epochs // 4
            pos = epoch % cycle
            return min(self.end, self.start + (self.end - self.start) * pos / cycle)
        return self.end
```

**Шаг 2: Обновить `scripts/train_vae.py`**

```python
from src.generation.vae import SmilesVAE, KLAnnealer

kl_annealer = KLAnnealer(n_epochs=epochs, start=0.0, end=1.0, strategy="linear")

for epoch in range(epochs):
    kl_weight = kl_annealer.get_weight(epoch)
    for batch in train_loader:
        outputs, mu, logvar = vae(batch)
        loss_dict = vae.loss_function(outputs, batch, mu, logvar, kl_weight=kl_weight)
        loss = loss_dict["loss"]
        # ...
```

**Шаг 3: Мониторинг KL collapse**
```python
# Признак posterior collapse: kl_loss < 0.1 при recon_loss > 5.0
# Добавить в логирование:
if loss_dict["kl_loss"].item() < 0.1:
    logger.warning(f"Potential KL collapse at epoch {epoch}! "
                   f"kl={loss_dict['kl_loss'].item():.4f}")
```

---

### 1.3 🔴 Fix AttentiveFP O(N×G) attention loop

**Файл:** `src/gnn/models.py`, строки 443–455  
**Проблема:** Python loop по числу графов в батче — крайне медленно на GPU.

**Шаг 1: Установить зависимость**
```bash
# В environment.yml / requirements.txt добавить:
pip install torch-scatter  # или через conda: conda install pyg -c pyg
```

**Шаг 2: Исправить `AttentiveFPModel.forward`**

```python
# БЫЛО (медленно):
attention_weights = torch.zeros_like(attention_scores)
for g in range(num_graphs):
    mask = (batch == g)
    attention_weights[mask] = F.softmax(attention_scores[mask], dim=0)

# ДОЛЖНО БЫТЬ (быстро, работает на GPU):
from torch_scatter import scatter_softmax
attention_weights = scatter_softmax(attention_scores, batch.unsqueeze(-1), dim=0)

# И аналогично для context:
# БЫЛО:
context = torch.zeros(num_graphs, self.hidden_dim, device=x.device)
for g in range(num_graphs):
    mask = (batch == g)
    context[g] = (attention_weights[mask] * x[mask]).sum(dim=0)

# ДОЛЖНО БЫТЬ:
from torch_scatter import scatter_add
context = scatter_add(attention_weights * x, batch.unsqueeze(-1).expand_as(x), dim=0)
```

---

### 1.4 🔴 Заменить pickle на joblib

**Файл:** `src/models/qsar_model.py:301`, `src/data/data_preprocessor.py:232`  
**Проблема:** pickle небезопасен при загрузке из недоверенных источников; также зависит от версии sklearn.

```python
# src/models/qsar_model.py — метод save/load:
import joblib  # вместо pickle

def save(self, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": self.model,
        "task": self.task,
        "params": self.params,
        "feature_names": self.feature_names,
        "training_metrics": self.training_metrics,
        "random_seed": self.random_seed,
        "sklearn_version": sklearn.__version__,  # для совместимости
    }
    joblib.dump(state, path, compress=3)  # compress=3 уменьшает размер ~3x
    logger.info(f"Model saved to {path} (joblib)")

@classmethod
def load(cls, path: str) -> "QSARModel":
    state = joblib.load(path)
    # проверить совместимость версий
    if state.get("sklearn_version") != sklearn.__version__:
        logger.warning(f"sklearn version mismatch: saved={state.get('sklearn_version')}, "
                       f"current={sklearn.__version__}")
    ...
```

---

### 1.5 🔴 Добавить PR-AUC в метрики классификации

**Файл:** `src/models/qsar_model.py:193`, `src/evaluation/metrics.py:66`  
**Проблема:** При несбалансированных классах (обычно 10–30% активных в ChEMBL) ROC-AUC оптимистичен; PR-AUC более информативен.

```python
# src/evaluation/metrics.py — в функцию calculate_classification_metrics:
from sklearn.metrics import average_precision_score

if y_proba is not None:
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))  # добавить
    # Для несбалансированных данных pr_auc более реалистичен
```

---

## 2. Тестирование и валидация

### 2.1 🟢 Property-based тесты (СДЕЛАНО)

**Файл:** `tests/test_property_based.py` — создан  

**Что покрыто:**
- Идемпотентность canonical SMILES
- Монотонность pIC50
- Отсутствие data leakage в split
- Scaffold split disjointness

**Шаг: Добавить в `requirements-dev.txt`:**
```
hypothesis>=6.100.0
pytest>=8.0.0
pytest-cov>=4.0.0
```

---

### 2.2 🟢 Integration тесты (СДЕЛАНО)

**Файл:** `tests/test_integration.py` — создан  

**Дополнительно нужно добавить GNN integration test:**

```python
# tests/test_integration.py — добавить класс:
@pytest.mark.skipif(not HAS_TORCH_GEOMETRIC, reason="torch_geometric not installed")
class TestGNNPipeline:
    def test_gcn_forward_pass(self):
        from gnn.featurizer import MoleculeFeaturizer
        from gnn.models import GCNModel

        featurizer = MoleculeFeaturizer()
        graphs = [featurizer.featurize(smi) for smi in MINI_SMILES[:5]]
        graphs = [g for g in graphs if g is not None]

        from torch_geometric.data import Batch
        batch = Batch.from_data_list(graphs)

        model = GCNModel(node_dim=featurizer.node_dim, hidden_dim=64, num_layers=2)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (len(graphs),)
        assert torch.isfinite(out).all()
```

---

### 2.3 🟢 Regression тесты (СДЕЛАНО)

**Файл:** `tests/test_regression.py`, `tests/baselines/regression_baselines.json` — созданы  

**Workflow для обновления baseline после намеренного улучшения:**
```bash
# 1. Убедиться, что улучшение реально:
pytest tests/test_regression.py -v  # должно упасть с "regressed"

# 2. Обновить baseline:
pytest tests/test_regression.py --update-baselines

# 3. Закоммитить обновлённый baseline:
git add tests/baselines/regression_baselines.json
git commit -m "chore: update regression baselines after scaffold split refactor"
```

---

### 2.4 🔴 Тест scaffold split vs random split

**Шаг: Добавить `tests/test_scaffold_split.py`**

```python
# tests/test_scaffold_split.py
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.scaffold_split import get_scaffold, scaffold_split, scaffold_k_fold

SMILES = [
    "CCO", "c1ccccc1", "CC(=O)O", "CCN", "c1ccc(O)cc1",
    "CC(=O)Nc1ccc(O)cc1", "Cc1ccccc1", "c1ccc(F)cc1",
    "c1ccc(Cl)cc1", "c1ccc(Br)cc1",  # benzene family
]

def test_benzene_family_same_scaffold():
    """Бензольные производные должны иметь один scaffold — c1ccccc1."""
    scaffolds = {get_scaffold(smi) for smi in ["c1ccccc1", "c1ccc(F)cc1", "c1ccc(Cl)cc1"]}
    assert len(scaffolds) == 1, f"Expected 1 scaffold, got {scaffolds}"

def test_split_sizes_approximately_correct():
    train, val, test = scaffold_split(SMILES * 10, 0.7, 0.15, 0.15)
    n = len(SMILES) * 10
    assert abs(len(train) / n - 0.7) < 0.15  # допуск из-за дискретности scaffold groups

def test_k_fold_coverage():
    splits = scaffold_k_fold(SMILES, n_folds=3)
    assert len(splits) == 3
    all_test = set()
    for _, test_idx in splits:
        all_test.update(test_idx)
    assert all_test == set(range(len(SMILES)))
```

---

### 2.5 🔴 Тест генеративных моделей (VAE reconstruction)

```python
# tests/test_vae_metrics.py
def test_vae_reconstruction_rate():
    """VAE должен восстанавливать >50% валидных молекул."""
    # Использует обученную модель из models/vae_final.pt
    import pytest
    if not Path("models/vae_final.pt").exists():
        pytest.skip("No trained VAE model found")

    from generation.vae import SmilesVAE
    from generation.tokenizer import SMILESTokenizer
    from evaluation.generation_metrics import compute_validity

    tokenizer = SMILESTokenizer.load("models/tokenizer.pkl")
    vae = SmilesVAE.load("models/vae_final.pt")

    generated = []
    for _ in range(100):
        tokens = vae.generate(num_samples=1, temperature=1.0)
        smi = tokenizer.decode(tokens[0].tolist())
        generated.append(smi)

    validity, _ = compute_validity(generated)
    assert validity >= 0.50, f"VAE validity too low: {validity:.1%}"
```

---

## 3. Улучшение моделей

### 3.1 🔴 SHAP интерпретация QSAR

**Файл:** `src/models/qsar_model.py`  
**Зачем:** Химики хотят понимать, какие структурные фрагменты влияют на активность.

**Шаг 1: Установить**
```bash
pip install shap
```

**Шаг 2: Добавить метод в `QSARModel`**

```python
# src/models/qsar_model.py — добавить метод:
def explain_shap(
    self,
    X: np.ndarray,
    n_background: int = 100,
    max_display: int = 20,
) -> "shap.Explanation":
    """Compute SHAP values for feature importance explanation.

    Args:
        X: Feature matrix to explain.
        n_background: Background dataset size for TreeExplainer.
        max_display: Number of top features to display.

    Returns:
        SHAP Explanation object.

    Example:
        >>> explainer_result = model.explain_shap(X_test[:50])
        >>> shap.summary_plot(explainer_result, X_test[:50],
        ...                   feature_names=model.feature_names)
    """
    import shap

    if not self.is_fitted:
        raise ValueError("Model not fitted")

    explainer = shap.TreeExplainer(self.model)
    shap_values = explainer.shap_values(X)

    # Для классификации shap_values — список [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # класс "active"

    return shap_values

def plot_shap_summary(self, X: np.ndarray, save_path: str = None) -> None:
    import shap
    import matplotlib.pyplot as plt

    values = self.explain_shap(X)
    shap.summary_plot(
        values, X,
        feature_names=self.feature_names,
        show=save_path is None,
    )
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
```

---

### 3.2 🔴 Applicability Domain (AD) для QSAR

**Зачем:** Предсказания вне applicability domain ненадёжны. Важно для коллаборации с химиками — они должны знать, когда модели доверять.

**Шаг: Создать `src/models/applicability_domain.py`**

```python
# src/models/applicability_domain.py
import numpy as np
from sklearn.preprocessing import StandardScaler

class BoundingBoxAD:
    """Простая AD на основе диапазонов дескрипторов обучающего набора.

    Молекула внутри AD если все её дескрипторы попадают в диапазон
    [mean - k*std, mean + k*std] обучающих данных.
    """

    def __init__(self, k: float = 3.0):
        self.k = k
        self.lower_: np.ndarray = None
        self.upper_: np.ndarray = None

    def fit(self, X_train: np.ndarray) -> "BoundingBoxAD":
        mean = X_train.mean(axis=0)
        std  = X_train.std(axis=0)
        self.lower_ = mean - self.k * std
        self.upper_ = mean + self.k * std
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns bool array: True = inside AD."""
        inside = np.all((X >= self.lower_) & (X <= self.upper_), axis=1)
        return inside

    def coverage(self, X: np.ndarray) -> float:
        return self.predict(X).mean()


class LeverageAD:
    """Williams plot: AD на основе leverage (hat matrix diagonal).

    Стандартный метод в QSAR. Молекула вне AD если h > h* = 3*(k+1)/n.
    """

    def __init__(self):
        self.X_train_: np.ndarray = None
        self.h_star_: float = None
        self.n_: int = None
        self.k_: int = None

    def fit(self, X_train: np.ndarray) -> "LeverageAD":
        self.X_train_ = X_train
        self.n_, self.k_ = X_train.shape
        self.h_star_ = 3.0 * (self.k_ + 1) / self.n_
        return self

    def leverage(self, X: np.ndarray) -> np.ndarray:
        """Compute leverage values h_i for test molecules."""
        XtX_inv = np.linalg.pinv(self.X_train_.T @ self.X_train_)
        h = np.array([x @ XtX_inv @ x for x in X])
        return h

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.leverage(X) <= self.h_star_
```

---

### 3.3 🔴 Uncertainty Quantification — Calibration

**Файл:** `src/evaluation/uncertainty.py` (существует, но что там — нужно проверить)

**Шаг: Добавить conformal prediction**

```python
# src/evaluation/conformal_prediction.py
import numpy as np
from typing import Tuple

class ConformalPredictor:
    """Split conformal prediction для QSAR regression.

    Даёт гарантированные (1-alpha) confidence intervals без
    параметрических предположений о распределении ошибок.

    References:
        Venn-ABERS predictor (Vovk et al. 2012)
        Barber et al. (2021) "Predictive Inference with the Jackknife+"
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha  # 1-alpha = coverage (0.1 → 90% coverage)
        self.nonconformity_scores_: np.ndarray = None
        self.threshold_: float = None

    def calibrate(self, y_cal: np.ndarray, y_pred_cal: np.ndarray) -> "ConformalPredictor":
        """Calibrate on a held-out calibration set."""
        self.nonconformity_scores_ = np.abs(y_cal - y_pred_cal)
        n = len(y_cal)
        q = np.ceil((1 - self.alpha) * (n + 1)) / n
        self.threshold_ = np.quantile(self.nonconformity_scores_, min(q, 1.0))
        return self

    def predict_interval(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) prediction intervals."""
        return y_pred - self.threshold_, y_pred + self.threshold_

    def coverage_empirical(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Check empirical coverage on test set (should be >= 1-alpha)."""
        lower, upper = self.predict_interval(y_pred)
        return float(((y_true >= lower) & (y_true <= upper)).mean())
```

---

### 3.4 🔴 Multi-objective молекулярная оптимизация

**Зачем:** Генерировать молекулы с одновременной высокой активностью, drug-likeness и синтетической доступностью.

**Шаг: Создать `src/generation/optimizer.py` (если не готов)**

```python
# src/generation/multi_objective.py
from dataclasses import dataclass
from typing import Callable, List
import numpy as np

@dataclass
class ObjectiveScore:
    activity: float      # pIC50 prediction (higher = better)
    qed: float           # drug-likeness 0-1 (higher = better)
    sa_score: float      # synthetic accessibility 1-10 (lower = better)
    novelty: float       # 1 if novel vs training set, else 0

    def weighted_sum(self, w_act=0.5, w_qed=0.3, w_sa=0.1, w_nov=0.1) -> float:
        sa_norm = 1.0 - (self.sa_score - 1) / 9  # normalize to 0-1
        return w_act * self.activity + w_qed * self.qed + w_sa * sa_norm + w_nov * self.novelty


def score_molecules(
    smiles_list: List[str],
    activity_predictor: Callable,
    training_smiles: set,
) -> List[ObjectiveScore]:
    """Score generated molecules on all objectives."""
    from rdkit import Chem
    from rdkit.Chem import QED, Descriptors
    from evaluation.generation_metrics import compute_validity

    _, valid_smiles = compute_validity(smiles_list)
    scores = []
    for smi in valid_smiles:
        mol = Chem.MolFromSmiles(smi)
        activity = float(activity_predictor([smi])[0])
        qed_score = QED.qed(mol)
        # SA score требует contrib
        try:
            import sascorer
            sa = sascorer.calculateScore(mol)
        except Exception:
            sa = 5.0  # нейтральное значение при отсутствии модуля
        novelty = 1.0 if smi not in training_smiles else 0.0
        scores.append(ObjectiveScore(activity, qed_score, sa, novelty))
    return scores
```

---

### 3.5 🔴 Внешняя валидация QSAR на независимом датасете

**Зачем:** Обязательно для публикации. Показывает, что модель обобщается за пределы ChEMBL.

**Шаг 1: Подготовить внешний датасет**
```
Источники для InhA:
- BindingDB: https://www.bindingdb.org (фильтр: target = InhA)
- PubChem BioAssay: AID 449764 (InhA inhibition)
- Literature: Sink et al. 2015, J. Med. Chem.
```

**Шаг 2: Скрипт валидации**
```python
# scripts/external_validation.py
from src.models.qsar_model import QSARModel
from src.data.chembl_loader import ChEMBLLoader
from src.data.descriptor_calculator import DescriptorCalculator
from src.evaluation.metrics import calculate_metrics, calculate_classification_metrics

def validate_external(model_path: str, external_csv: str):
    model = QSARModel.load(model_path)
    loader = ChEMBLLoader()
    df = loader.load_from_csv(external_csv)
    df_clean = loader.preprocess(df)

    calc = DescriptorCalculator()
    desc_df = calc.calculate_batch(df_clean["smiles"].tolist())
    X = desc_df[model.feature_names].fillna(0).values

    y_pred = model.predict(X)
    metrics = calculate_metrics(df_clean["pIC50"].values, y_pred)
    print(f"External validation: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.3f}")
    return metrics
```

---

## 4. MLOps и воспроизводимость

### 4.1 🔴 MLflow Experiment Tracking

**Зачем:** Без tracking невозможно воспроизвести лучший эксперимент через 3 месяца.

**Шаг 1: Установить**
```bash
pip install mlflow
# Добавить в requirements.txt: mlflow>=2.10.0
```

**Шаг 2: Обернуть `scripts/train_qsar.py`**

```python
# scripts/train_qsar.py — добавить MLflow:
import mlflow
import mlflow.sklearn

mlflow.set_experiment("tb-qsar-inha")

with mlflow.start_run(run_name=f"rf-scaffold-{datetime.now():%Y%m%d-%H%M}"):
    # Логировать конфигурацию
    mlflow.log_params({
        "n_estimators": config.get("qsar.n_estimators"),
        "split_type": "scaffold",
        "activity_threshold": config.get("qsar.activity_threshold"),
        "n_train": len(X_train),
        "n_test": len(X_test),
    })

    # Обучить модель
    model.fit(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)

    # Логировать метрики
    mlflow.log_metrics(metrics)

    # Сохранить модель в registry
    mlflow.sklearn.log_model(model.model, "qsar_model")

    # Сохранить артефакты
    mlflow.log_artifact("config/default_config.yaml")
```

**Шаг 3: Запустить UI**
```bash
mlflow ui --port 5000
# Открыть: http://localhost:5000
```

**Шаг 4: Добавить в `Makefile`**
```makefile
# Makefile
train-qsar:
	python scripts/train_qsar.py

mlflow-ui:
	mlflow ui --port 5000

.PHONY: train-qsar mlflow-ui
```

---

### 4.2 🔴 DVC для версионирования данных

**Зачем:** `data/` не в git (правильно!), но без DVC нет воспроизводимости экспериментов.

**Шаг 1: Инициализировать DVC**
```bash
pip install dvc[gdrive]  # или dvc[s3] если есть S3
dvc init
git add .dvc .dvcignore
git commit -m "chore: initialize DVC"
```

**Шаг 2: Добавить данные**
```bash
dvc add data/raw/chembl_inha.csv
dvc add data/processed/inha_clean.csv
git add data/raw/chembl_inha.csv.dvc data/processed/inha_clean.csv.dvc
git commit -m "data: add ChEMBL InhA dataset v1"
```

**Шаг 3: Настроить remote (Google Drive — бесплатно)**
```bash
dvc remote add -d gdrive gdrive://YOUR_FOLDER_ID
dvc push  # загрузить данные в облако
```

**Шаг 4: Воспроизводимость**
```bash
# На новой машине:
git clone https://github.com/yhtyyar/tb_drug_discovery
dvc pull  # скачать данные
```

---

### 4.3 🔴 Makefile для стандартных операций

```makefile
# Makefile
.PHONY: install test lint train clean

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src --cov-report=term-missing \
		--ignore=tests/test_alphafold.py \
		--ignore=tests/test_docking.py

test-property:
	pytest tests/test_property_based.py -v --hypothesis-seed=42

test-regression:
	pytest tests/test_regression.py -v

lint:
	black --check --line-length=100 src/ tests/ scripts/
	isort --check-only --profile=black src/ tests/ scripts/
	flake8 src/ tests/ --max-line-length=100
	mypy src/ --ignore-missing-imports

format:
	black --line-length=100 src/ tests/ scripts/
	isort --profile=black src/ tests/ scripts/

train-qsar:
	python scripts/train_qsar.py

train-gnn:
	python scripts/train_gnn.py

train-vae:
	python scripts/train_vae.py

train-diffusion:
	python scripts/train_diffusion.py

screen:
	python scripts/run_docking.py

api:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .coverage coverage.xml htmlcov/
```

---

### 4.4 🔴 conda environment.yml для воспроизводимости

```yaml
# environment.yml
name: tb-drug-discovery
channels:
  - conda-forge
  - pytorch
  - pyg
  - defaults

dependencies:
  - python=3.10
  - rdkit=2023.09.5
  - pytorch=2.1.0
  - pytorch-cuda=11.8  # убрать для CPU-only
  - pyg=2.4.0
  - numpy=1.26.0
  - pandas=2.1.0
  - scikit-learn=1.3.0
  - scipy=1.11.0
  - pip:
    - chembl-webresource-client==0.10.8
    - loguru==0.7.2
    - pydantic==2.5.0
    - fastapi==0.109.0
    - uvicorn[standard]==0.27.0
    - mlflow==2.10.0
    - optuna==3.5.0
    - hypothesis==6.100.0
    - shap==0.44.0
    - tqdm==4.66.0
    - pyyaml==6.0.1
    - joblib==1.3.2
```

```bash
# Создание окружения:
conda env create -f environment.yml
conda activate tb-drug-discovery

# Обновление:
conda env update -f environment.yml --prune
```

---

## 5. Production API и мониторинг

### 5.1 🟢 FastAPI inference service (СДЕЛАНО)

**Файл:** `src/api/app.py` — создан  

**Шаг: Добавить аутентификацию для коллаборации**

```python
# src/api/app.py — добавить перед endpoints:
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Security, Depends
import secrets

security = HTTPBearer()
API_TOKENS = {"collab-token-xyz", "internal-token-abc"}  # переместить в .env

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials not in API_TOKENS:
        raise HTTPException(status_code=401, detail="Invalid API token")
    return credentials.credentials

# Применить к защищённым endpoint'ам:
@app.post("/predict/activity", response_model=ActivityResponse)
async def predict_activity(request: ActivityRequest, token: str = Depends(verify_token)):
    ...
```

**Шаг: Dockerfile для деплоя**

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Системные зависимости для RDKit
RUN apt-get update && apt-get install -y \
    libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/
COPY config/ config/

ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Сборка и запуск:
docker build -t tb-drug-api:latest .
docker run -p 8000:8000 -v $(pwd)/models:/app/models tb-drug-api:latest
```

---

### 5.2 🔴 Мониторинг дрейфа данных

**Файл:** `src/evaluation/drift_detector.py`

```python
# src/evaluation/drift_detector.py
from scipy.stats import ks_2samp
import numpy as np
from typing import Dict, List
from loguru import logger


class DescriptorDriftDetector:
    """KS-test based drift detection for molecular descriptors.

    При добавлении новых молекул в базу — проверить, не изменилось ли
    распределение дескрипторов. Дрейф = модель может деградировать.
    """

    def __init__(self, alpha: float = 0.05, feature_names: List[str] = None):
        self.alpha = alpha
        self.feature_names = feature_names
        self.reference_X_: np.ndarray = None

    def fit(self, X_reference: np.ndarray) -> "DescriptorDriftDetector":
        self.reference_X_ = X_reference
        return self

    def detect(self, X_new: np.ndarray) -> Dict[str, object]:
        """Run KS test on each feature. Returns dict with drifted features."""
        drifted = []
        p_values = {}

        for i in range(X_new.shape[1]):
            stat, p = ks_2samp(self.reference_X_[:, i], X_new[:, i])
            name = self.feature_names[i] if self.feature_names else f"feat_{i}"
            p_values[name] = float(p)
            if p < self.alpha:
                drifted.append((name, float(stat), float(p)))

        result = {
            "n_features_drifted": len(drifted),
            "fraction_drifted": len(drifted) / X_new.shape[1],
            "drifted_features": drifted,
            "p_values": p_values,
            "drift_detected": len(drifted) > 0,
        }

        if result["drift_detected"]:
            logger.warning(
                f"Data drift detected in {len(drifted)} features: "
                f"{[d[0] for d in drifted[:5]]}"
            )
        return result
```

---

### 5.3 🔴 Dashboard скрипт для мониторинга метрик

```python
# scripts/generate_report.py
"""Генерация HTML-отчёта с метриками всех моделей."""
import json
from pathlib import Path
import pandas as pd

METRICS_DIR = Path("results/metrics")

def load_all_metrics() -> pd.DataFrame:
    rows = []
    for f in METRICS_DIR.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)
        data["model"] = f.stem
        rows.append(data)
    return pd.DataFrame(rows)

def generate_html_report(output: str = "results/report.html"):
    df = load_all_metrics()
    html = df.to_html(index=False, float_format="{:.3f}".format)
    Path(output).write_text(f"<html><body><h1>TB Drug Discovery Metrics</h1>{html}</body></html>")
    print(f"Report saved to {output}")

if __name__ == "__main__":
    generate_html_report()
```

---

## 6. Расширение пайплайна

### 6.1 🔴 GuacaMol бенчмарки для генеративных моделей

**Зачем:** Стандартизированная оценка генеративных моделей для публикации.

```bash
pip install guacamol
```

```python
# scripts/benchmark_generation.py
from guacamol.assess_distribution_learning import assess_distribution_learning
from guacamol.distribution_matching_generator import DistributionMatchingGenerator

class VAEGuacaMolWrapper(DistributionMatchingGenerator):
    def __init__(self, vae, tokenizer):
        self.vae = vae
        self.tokenizer = tokenizer

    def generate(self, number_samples: int):
        tokens = self.vae.generate(num_samples=number_samples)
        return [self.tokenizer.decode(t.tolist()) for t in tokens]

# Запуск бенчмарка:
generator = VAEGuacaMolWrapper(vae, tokenizer)
results = assess_distribution_learning(
    generator,
    chembl_training_file="data/raw/guacamol_v1_train.smiles",
)
print(results)
```

---

### 6.2 🔴 REINVENT-style RL fine-tuning

**Зачем:** После обучения VAE/Diffusion — fine-tune RL для максимизации предсказанной активности.

```python
# src/generation/rl_finetuner.py
import torch
import torch.nn as nn
from typing import Callable, List

class REINFORCEFinetuner:
    """REINFORCE-based fine-tuning для генеративных моделей.

    Оптимизирует composite reward: activity + QED + SA + novelty.
    """

    def __init__(
        self,
        generator,    # VAE или Diffusion model
        tokenizer,
        reward_fn: Callable[[List[str]], List[float]],
        baseline_decay: float = 0.99,
    ):
        self.generator = generator
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.baseline = 0.0
        self.baseline_decay = baseline_decay

    def step(self, n_samples: int = 64, learning_rate: float = 1e-5):
        """One RL optimization step."""
        # Sample molecules
        tokens = self.generator.generate(num_samples=n_samples)
        smiles = [self.tokenizer.decode(t.tolist()) for t in tokens]

        # Score
        rewards = torch.tensor(self.reward_fn(smiles), dtype=torch.float32)

        # Update baseline (moving average)
        self.baseline = (
            self.baseline_decay * self.baseline +
            (1 - self.baseline_decay) * rewards.mean().item()
        )

        # Advantage
        advantage = rewards - self.baseline

        # Policy gradient loss (simplified)
        # В production: использовать log-probabilities из модели
        loss = -advantage.mean()
        return {"loss": loss.item(), "mean_reward": rewards.mean().item(),
                "baseline": self.baseline}
```

---

### 6.3 🔴 Добавить таргет rpoB

```python
# src/data/chembl_loader.py уже содержит:
TARGETS = {
    "InhA": "CHEMBL1849",
    "rpoB": "CHEMBL1790",   # уже есть!
    "KatG": "CHEMBL1916",
}

# Нужно только добавить multi-target тренировку:
# scripts/train_multitarget_qsar.py
targets = ["InhA", "rpoB"]
models = {}
for target in targets:
    loader = ChEMBLLoader(target_id=ChEMBLLoader.TARGETS[target])
    df = loader.load_from_csv(f"data/raw/chembl_{target.lower()}.csv")
    # ... обучение
    models[target] = model

# Ensemble предсказание:
def predict_multitarget(smiles_list, models):
    results = {}
    for target, model in models.items():
        results[target] = model.predict(compute_features(smiles_list))
    return results
```

---

## 7. Документация и коллаборация

### 7.1 🔴 CONTRIBUTING.md

```markdown
# Contributing to TB Drug Discovery Pipeline

## Adding a new model
1. Create `src/models/your_model.py`
2. Implement interface: `fit(X, y)`, `predict(X)`, `evaluate(X, y)`
3. Add tests in `tests/test_your_model.py`
4. Register in `config/default_config.yaml`

## Adding a new target
1. Add target ID to `ChEMBLLoader.TARGETS`
2. Add config section to `config/default_config.yaml`
3. Create `data/raw/chembl_{target}.csv`

## Code style
- Black (line-length=100)
- isort (profile=black)
- Type hints required
- Docstrings for public methods (Google style)
```

---

### 7.2 🔴 Автогенерация API-документации

```bash
# Sphinx для Python docs:
pip install sphinx sphinx-autodoc-typehints sphinx-rtd-theme

# Инициализация:
cd docs && sphinx-quickstart

# docs/conf.py — добавить:
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google/NumPy docstrings
    "sphinx_autodoc_typehints",
]

# Генерация:
cd docs && make html
# Открыть: docs/_build/html/index.html
```

---

## Сводная таблица приоритетов

| # | Задача | Приоритет | Файл | Усилие | Влияние |
|---|--------|-----------|------|--------|---------|
| 1 | Пересчитать метрики со scaffold split | 🔴 HIGH | `scripts/train_qsar.py` | 2ч | Правильные метрики |
| 2 | KL Annealing в VAE | 🔴 HIGH | `src/generation/vae.py` | 1ч | Качество генерации |
| 3 | Fix AttentiveFP attention loop | 🔴 HIGH | `src/gnn/models.py:443` | 1ч | Скорость обучения |
| 4 | PR-AUC в метрики | 🟡 HIGH | `src/evaluation/metrics.py` | 30мин | Корректная оценка |
| 5 | SHAP интерпретация | 🟡 MEDIUM | `src/models/qsar_model.py` | 2ч | Коллаборация |
| 6 | MLflow tracking | 🟡 MEDIUM | `scripts/train_*.py` | 3ч | Воспроизводимость |
| 7 | Applicability Domain | 🟡 MEDIUM | `src/models/` | 2ч | Достоверность |
| 8 | Conformal prediction | 🟡 MEDIUM | `src/evaluation/` | 2ч | Uncertainty |
| 9 | DVC data versioning | 🟡 MEDIUM | `.dvc/` | 2ч | Воспроизводимость |
| 10 | GuacaMol бенчмарк | 🟢 LOW | `scripts/` | 3ч | Публикация |
| 11 | RL fine-tuning | 🟢 LOW | `src/generation/` | 8ч | Novel molecules |
| 12 | External validation | 🟡 HIGH | `scripts/` | 4ч | Публикация |
| 13 | Dockerfile | 🟡 MEDIUM | `.` | 1ч | Деплой |
| 14 | Makefile | 🟢 LOW | `.` | 30мин | Удобство |
| 15 | conda env.yml | 🟡 MEDIUM | `.` | 30мин | Воспроизводимость |

---

## Быстрый старт: неделя 1

```bash
# День 1: Scaffold split + пересчёт метрик
python scripts/train_qsar.py --split scaffold
# Сохранить новые метрики, сравнить с random split

# День 2: KL Annealing в VAE
# Отредактировать vae.py + train_vae.py (см. раздел 1.2)
# Запустить короткое обучение (10 эпох) и проверить KL loss

# День 3: Тесты
pip install hypothesis
pytest tests/test_property_based.py tests/test_integration.py -v

# День 4: Generation metrics
# Запустить evaluate_generation() на VAE/Diffusion результатах
from src.evaluation.generation_metrics import evaluate_generation

# День 5: MLflow
pip install mlflow
# Обернуть train_qsar.py (см. раздел 4.1)
mlflow ui
```
