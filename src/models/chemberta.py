"""
ChemBERTa molecular representations using HuggingFace Transformers.

ChemBERTa is a BERT-style language model pre-trained on SMILES strings.
Its [CLS] token embedding captures rich chemical semantics and often
outperforms fixed fingerprints on low-data regimes — which is common in
TB drug discovery where labelled data is scarce.

Model card: seyonec/ChemBERTa-zinc-base-v1 (384-dim embeddings, 6-layer BERT)

Requirements (optional):
    pip install transformers torch

If torch / transformers are not installed, importing this module raises an
informative ImportError only when the ChemBERTa classes are actually used.

Example
-------
>>> from src.models.chemberta import ChemBERTaFeaturizer, ChemBERTaQSAR
>>>
>>> featurizer = ChemBERTaFeaturizer(batch_size=64, device='cuda')
>>> embeddings = featurizer.encode(smiles_list)   # (n, 384)
>>>
>>> qsar = ChemBERTaQSAR()
>>> qsar.fit(train_smiles, y_train)
>>> probs = qsar.predict_proba(test_smiles)
>>>
>>> # Compare ChemBERTa vs Morgan FP vs ECFP
>>> results = compare_representations(smiles, labels, cv=5)
>>> print(results)
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports: transformers / torch are optional heavy dependencies
_TRANSFORMERS_AVAILABLE: Optional[bool] = None
_TORCH_AVAILABLE: Optional[bool] = None


def _check_transformers() -> None:
    """Raise a clear ImportError if transformers or torch are not installed."""
    global _TRANSFORMERS_AVAILABLE, _TORCH_AVAILABLE

    if _TRANSFORMERS_AVAILABLE is None:
        try:
            import transformers  # noqa: F401
            _TRANSFORMERS_AVAILABLE = True
        except ImportError:
            _TRANSFORMERS_AVAILABLE = False

    if _TORCH_AVAILABLE is None:
        try:
            import torch  # noqa: F401
            _TORCH_AVAILABLE = True
        except ImportError:
            _TORCH_AVAILABLE = False

    missing = []
    if not _TRANSFORMERS_AVAILABLE:
        missing.append("transformers")
    if not _TORCH_AVAILABLE:
        missing.append("torch")

    if missing:
        raise ImportError(
            f"ChemBERTa requires: {', '.join(missing)}.\n"
            f"Install with: pip install {' '.join(missing)}\n"
            "If GPU support is needed: pip install torch --index-url https://download.pytorch.org/whl/cu118"
        )


# ---------------------------------------------------------------------------
# ChemBERTa Featurizer
# ---------------------------------------------------------------------------

class ChemBERTaFeaturizer:
    """Converts SMILES strings to fixed-size embeddings using ChemBERTa.

    The [CLS] token (or mean-pooled token) representation is extracted from
    the pre-trained ChemBERTa encoder.  No fine-tuning is performed; the
    pre-trained weights are used as-is (transfer learning).

    The model is loaded once and cached in ``self._model`` / ``self._tokenizer``
    on the first call to encode().

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
        Common choices:
          - 'seyonec/ChemBERTa-zinc-base-v1'  (default, 384-dim)
          - 'seyonec/ChemBERTa-zinc-base-v2'
          - 'seyonec/PubChem10M_SMILES_BPE_450k'
    device : str or None
        'cuda', 'cpu', or None (auto-detect GPU if available).
    batch_size : int
        Number of SMILES processed per forward pass.  Larger = faster on
        GPU but higher memory usage.
    pooling : {'cls', 'mean'}
        'cls': use the [CLS] token embedding.
        'mean': average all token embeddings (sometimes better for short SMILES).

    Attributes
    ----------
    embedding_dim : int
        Dimensionality of the output embeddings.  Set after first encode() call.

    Examples
    --------
    >>> feat = ChemBERTaFeaturizer(device='cpu', batch_size=16)
    >>> X = feat.encode(['CCO', 'c1ccccc1', 'CC(=O)Oc1ccccc1C(=O)O'])
    >>> X.shape
    (3, 384)

    sklearn-compatible usage (for use inside a Pipeline):
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> pipe = Pipeline([('embed', feat), ('clf', RandomForestClassifier())])
    >>> pipe.fit(train_smiles, y_train)
    """

    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        device: Optional[str] = None,
        batch_size: int = 32,
        pooling: Literal["cls", "mean"] = "cls",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.pooling = pooling

        # Resolve device
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self._model = None
        self._tokenizer = None
        self.embedding_dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Model loading (cached)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load tokenizer and model from HuggingFace (first call only)."""
        if self._model is not None:
            return

        _check_transformers()

        from transformers import AutoModel, AutoTokenizer  # type: ignore
        import torch

        logger.info("Loading ChemBERTa model: %s → %s", self.model_name, self.device)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)

        # Determine embedding dimension from model config
        self.embedding_dim = self._model.config.hidden_size
        logger.info(
            "ChemBERTa loaded: %d-dim embeddings, device=%s", self.embedding_dim, self.device
        )

    # ------------------------------------------------------------------
    # Core encoding
    # ------------------------------------------------------------------

    def encode(self, smiles_list: List[str]) -> np.ndarray:
        """Encode a list of SMILES strings into embedding vectors.

        Invalid SMILES (those that cause tokeniser / model errors) are
        replaced with zero vectors of the correct dimension.

        Parameters
        ----------
        smiles_list : list of str
            SMILES strings to encode.

        Returns
        -------
        np.ndarray of shape (n, embedding_dim)
            Float32 embedding matrix.  Row i corresponds to smiles_list[i].

        Examples
        --------
        >>> embeddings = featurizer.encode(['CCO', 'invalid!!!smiles', 'c1ccccc1'])
        >>> embeddings.shape
        (3, 384)
        """
        self._load_model()
        _check_transformers()
        import torch

        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        n = len(smiles_list)
        all_embeddings: List[np.ndarray] = []

        batch_iter = range(0, n, self.batch_size)
        if use_tqdm and n > self.batch_size:
            batch_iter = tqdm(
                batch_iter,
                desc="ChemBERTa encoding",
                total=(n + self.batch_size - 1) // self.batch_size,
                unit="batch",
            )

        with torch.no_grad():
            for start in batch_iter:
                batch = smiles_list[start : start + self.batch_size]
                batch_embeddings = self._encode_batch(batch)
                all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings).astype(np.float32)

    def encode_with_attention(
        self, smiles_list: List[str]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Encode SMILES and return both embeddings and attention weights.

        Attention weights can be visualised to understand which atoms /
        substructures the model focuses on.

        Parameters
        ----------
        smiles_list : list of str

        Returns
        -------
        embeddings : np.ndarray of shape (n, embedding_dim)
        attention_weights : list of np.ndarray
            One entry per molecule.  Each entry has shape
            (n_layers, n_heads, seq_len, seq_len).

        Examples
        --------
        >>> embeddings, attn = featurizer.encode_with_attention(['CCO'])
        >>> # attn[0].shape → (6, 12, seq_len, seq_len)
        """
        self._load_model()
        _check_transformers()
        import torch

        embeddings_list: List[np.ndarray] = []
        attention_list: List[np.ndarray] = []

        with torch.no_grad():
            for smiles in smiles_list:
                try:
                    inputs = self._tokenizer(
                        smiles,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=False,
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    outputs = self._model(
                        **inputs, output_attentions=True, return_dict=True
                    )

                    emb = self._pool(outputs.last_hidden_state)
                    embeddings_list.append(emb.cpu().numpy()[0])

                    # Stack attention from all layers: (n_layers, n_heads, seq, seq)
                    attn_stacked = np.stack(
                        [a.cpu().numpy()[0] for a in outputs.attentions]
                    )
                    attention_list.append(attn_stacked)

                except Exception as exc:
                    logger.warning("Attention encoding failed for %r: %s", smiles, exc)
                    dim = self.embedding_dim or 384
                    embeddings_list.append(np.zeros(dim, dtype=np.float32))
                    attention_list.append(np.array([]))

        embeddings = np.array(embeddings_list, dtype=np.float32)
        return embeddings, attention_list

    # ------------------------------------------------------------------
    # sklearn compatibility (fit/transform)
    # ------------------------------------------------------------------

    def fit(self, smiles_list: List[str], y=None) -> "ChemBERTaFeaturizer":
        """No-op fit method for sklearn Pipeline compatibility.

        ChemBERTa is a pre-trained model; no fitting is performed.

        Parameters
        ----------
        smiles_list : list of str
        y : ignored

        Returns
        -------
        self
        """
        self._load_model()  # Trigger model loading so embedding_dim is set
        return self

    def transform(self, smiles_list: List[str], y=None) -> np.ndarray:
        """Transform SMILES to embeddings (calls encode internally).

        Parameters
        ----------
        smiles_list : list of str

        Returns
        -------
        np.ndarray of shape (n, embedding_dim)
        """
        return self.encode(smiles_list)

    def fit_transform(self, smiles_list: List[str], y=None) -> np.ndarray:
        """Fit (no-op) and transform in one call."""
        self.fit(smiles_list, y)
        return self.transform(smiles_list)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode_batch(self, batch: List[str]) -> np.ndarray:
        """Encode a single batch; returns zero vector for invalid SMILES."""
        import torch

        dim = self.embedding_dim or 384
        batch_embeddings = []

        for smiles in batch:
            try:
                inputs = self._tokenizer(
                    smiles,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=False,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self._model(**inputs, return_dict=True)
                emb = self._pool(outputs.last_hidden_state)
                batch_embeddings.append(emb.cpu().numpy()[0])
            except Exception as exc:
                logger.debug("Encoding failed for %r: %s", smiles, exc)
                batch_embeddings.append(np.zeros(dim, dtype=np.float32))

        return np.array(batch_embeddings, dtype=np.float32)

    def _pool(self, last_hidden_state):
        """Apply the configured pooling to the last hidden state tensor."""
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]  # [CLS] token
        elif self.pooling == "mean":
            return last_hidden_state.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling!r}")

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ChemBERTaFeaturizer(model_name={self.model_name!r}, "
            f"device={self.device!r}, batch_size={self.batch_size}, "
            f"pooling={self.pooling!r})"
        )


# ---------------------------------------------------------------------------
# ChemBERTa QSAR model
# ---------------------------------------------------------------------------

class ChemBERTaQSAR:
    """End-to-end QSAR model using ChemBERTa embeddings.

    Internally builds:  ChemBERTaFeaturizer → RobustScaler → RandomForestClassifier

    This sklearn Pipeline accepts raw SMILES strings at fit/predict time.

    Parameters
    ----------
    model_name : str
        HuggingFace model name passed to ChemBERTaFeaturizer.
    device : str or None
        Compute device for the transformer.
    batch_size : int
        Batch size for ChemBERTa forward passes.
    pooling : {'cls', 'mean'}
        Pooling strategy.
    n_estimators : int
        Number of RF trees.
    random_seed : int
        Random seed for reproducibility.
    task : {'classification', 'regression'}

    Examples
    --------
    >>> model = ChemBERTaQSAR(device='cuda', n_estimators=200)
    >>> model.fit(train_smiles, y_train)
    >>> y_pred = model.predict(test_smiles)
    >>> probs = model.predict_proba(test_smiles)
    """

    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        device: Optional[str] = None,
        batch_size: int = 32,
        pooling: Literal["cls", "mean"] = "cls",
        n_estimators: int = 200,
        random_seed: int = 42,
        task: Literal["classification", "regression"] = "classification",
    ) -> None:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler

        self.task = task
        self.model_name = model_name
        self.n_estimators = n_estimators
        self.random_seed = random_seed

        featurizer = ChemBERTaFeaturizer(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            pooling=pooling,
        )

        if task == "classification":
            estimator = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_seed,
                class_weight="balanced",
                n_jobs=-1,
            )
        else:
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_seed,
                n_jobs=-1,
            )

        self.pipeline = Pipeline(
            steps=[
                ("chemberta", featurizer),
                ("scaler", RobustScaler()),
                ("model", estimator),
            ]
        )

    def fit(self, smiles_list: List[str], y: np.ndarray) -> "ChemBERTaQSAR":
        """Fit the full pipeline on SMILES strings.

        Parameters
        ----------
        smiles_list : list of str
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        logger.info("Fitting ChemBERTaQSAR on %d molecules.", len(smiles_list))
        self.pipeline.fit(smiles_list, y)
        self.classes_ = getattr(self.pipeline.named_steps["model"], "classes_", None)
        return self

    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """Predict labels / values.

        Parameters
        ----------
        smiles_list : list of str

        Returns
        -------
        np.ndarray
        """
        return self.pipeline.predict(smiles_list)

    def predict_proba(self, smiles_list: List[str]) -> np.ndarray:
        """Predict class probabilities (classification only).

        Parameters
        ----------
        smiles_list : list of str

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
        """
        if self.task != "classification":
            raise AttributeError("predict_proba is only available for classification.")
        return self.pipeline.predict_proba(smiles_list)

    def score(self, smiles_list: List[str], y: np.ndarray) -> float:
        """Return pipeline score (accuracy or R²)."""
        return self.pipeline.score(smiles_list, y)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ChemBERTaQSAR(model_name={self.model_name!r}, "
            f"task={self.task!r}, n_estimators={self.n_estimators})"
        )


# ---------------------------------------------------------------------------
# Representation comparison utility
# ---------------------------------------------------------------------------

def compare_representations(
    smiles_list: List[str],
    y: np.ndarray,
    cv: int = 5,
    random_seed: int = 42,
    scoring: str = "roc_auc",
    n_estimators: int = 100,
    chemberta_model: str = "seyonec/ChemBERTa-zinc-base-v1",
    chemberta_device: Optional[str] = None,
) -> "pd.DataFrame":
    """Compare Morgan FP, ECFP4, and ChemBERTa embeddings on a classification task.

    Uses stratified k-fold cross-validation to estimate generalisation
    performance of each representation with a Random Forest classifier.

    Parameters
    ----------
    smiles_list : list of str
        SMILES strings for all molecules.
    y : array-like
        Binary or multi-class labels.
    cv : int
        Number of cross-validation folds.
    random_seed : int
        Random seed for CV splitting and RF.
    scoring : str
        sklearn scoring string, e.g. 'roc_auc', 'accuracy', 'f1'.
    n_estimators : int
        Number of trees in each RF.
    chemberta_model : str
        HuggingFace model name for ChemBERTa.
    chemberta_device : str or None
        Device for ChemBERTa forward passes.

    Returns
    -------
    pd.DataFrame
        Columns: representation, mean_score, std_score, cv_scores.

    Raises
    ------
    ImportError if pandas is not installed.

    Example
    -------
    >>> results = compare_representations(smiles, labels, cv=5, scoring='roc_auc')
    >>> print(results[['representation', 'mean_score', 'std_score']].to_string())
    representation    mean_score  std_score
    Morgan FP (2048)      0.823      0.031
    ECFP4 (1024)          0.817      0.028
    ChemBERTa CLS         0.851      0.024
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for compare_representations.") from exc

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler

    y = np.asarray(y)
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_seed)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_seed,
        class_weight="balanced",
        n_jobs=-1,
    )

    results = []

    # --- Morgan FP (2048 bits, radius 2) ---
    logger.info("Evaluating Morgan FP (2048 bits)...")
    morgan_X = _compute_morgan_fp(smiles_list, n_bits=2048, radius=2)
    morgan_pipe = Pipeline([("scaler", RobustScaler()), ("rf", rf)])
    morgan_scores = cross_val_score(morgan_pipe, morgan_X, y, cv=cv_splitter, scoring=scoring, n_jobs=-1)
    results.append({
        "representation": "Morgan FP (2048)",
        "mean_score": float(np.mean(morgan_scores)),
        "std_score": float(np.std(morgan_scores)),
        "cv_scores": morgan_scores.tolist(),
    })

    # --- ECFP4 (1024 bits, radius 2, use_features=True) ---
    logger.info("Evaluating ECFP4 (1024 bits)...")
    ecfp_X = _compute_morgan_fp(smiles_list, n_bits=1024, radius=2, use_features=True)
    ecfp_pipe = Pipeline([("scaler", RobustScaler()), ("rf", rf)])
    ecfp_scores = cross_val_score(ecfp_pipe, ecfp_X, y, cv=cv_splitter, scoring=scoring, n_jobs=-1)
    results.append({
        "representation": "ECFP4 (1024)",
        "mean_score": float(np.mean(ecfp_scores)),
        "std_score": float(np.std(ecfp_scores)),
        "cv_scores": ecfp_scores.tolist(),
    })

    # --- ChemBERTa ---
    logger.info("Evaluating ChemBERTa (%s)...", chemberta_model)
    try:
        _check_transformers()
        featurizer = ChemBERTaFeaturizer(
            model_name=chemberta_model,
            device=chemberta_device,
            batch_size=32,
        )
        chemberta_X = featurizer.encode(smiles_list)
        chemberta_pipe = Pipeline([("scaler", RobustScaler()), ("rf", rf)])
        chemberta_scores = cross_val_score(
            chemberta_pipe, chemberta_X, y, cv=cv_splitter, scoring=scoring, n_jobs=-1
        )
        results.append({
            "representation": "ChemBERTa CLS",
            "mean_score": float(np.mean(chemberta_scores)),
            "std_score": float(np.std(chemberta_scores)),
            "cv_scores": chemberta_scores.tolist(),
        })
    except ImportError as exc:
        logger.warning("Skipping ChemBERTa: %s", exc)
        results.append({
            "representation": "ChemBERTa CLS",
            "mean_score": float("nan"),
            "std_score": float("nan"),
            "cv_scores": [],
        })

    df = pd.DataFrame(results).sort_values("mean_score", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Internal utility: Morgan / ECFP fingerprints via RDKit
# ---------------------------------------------------------------------------

def _compute_morgan_fp(
    smiles_list: List[str],
    n_bits: int = 2048,
    radius: int = 2,
    use_features: bool = False,
) -> np.ndarray:
    """Compute Morgan / ECFP fingerprints for all SMILES.

    Invalid SMILES produce zero vectors.

    Parameters
    ----------
    smiles_list : list of str
    n_bits : int
        Number of bits in the bit vector.
    radius : int
        Morgan radius (2 = ECFP4, 3 = ECFP6).
    use_features : bool
        If True, use pharmacophoric features (FCFP instead of ECFP).

    Returns
    -------
    np.ndarray of shape (n, n_bits), dtype float32
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for Morgan fingerprint computation. "
            "Install with: conda install -c conda-forge rdkit"
        ) from exc

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.float32))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=n_bits, useFeatures=use_features
        )
        fps.append(np.array(fp, dtype=np.float32))

    return np.array(fps, dtype=np.float32)
