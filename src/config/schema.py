"""
Pydantic v2 configuration schema for the TB drug discovery pipeline.

Provides validated, typed configuration objects that can be loaded from YAML
files.  All validation errors are collected and re-raised together so the
user sees every problem at once rather than fixing one error at a time.

Example usage:
    >>> from src.config.schema import PipelineConfig, validate_config
    >>> cfg = validate_config('configs/default.yaml')
    >>> print(cfg.qsar.n_estimators)
    200

    Or construct programmatically:
    >>> cfg = PipelineConfig(
    ...     data=DataConfig(raw_dir='data/raw', chembl_target='CHEMBL1873'),
    ...     qsar=QSARConfig(n_estimators=500),
    ... )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ConfigValidationError(Exception):
    """Raised when configuration fails validation.

    Carries a list of violation strings so callers can display all problems
    at once.

    Parameters
    ----------
    violations : list of str
        Human-readable descriptions of each individual failure.
    """

    def __init__(self, violations: List[str]) -> None:
        self.violations = violations
        bullet_list = "\n".join(f"  - {v}" for v in violations)
        super().__init__(
            f"Configuration validation failed with {len(violations)} error(s):\n{bullet_list}"
        )


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class DataConfig(BaseModel):
    """Settings that control raw / processed data locations and ChEMBL fetching.

    Parameters
    ----------
    raw_dir : str or Path
        Directory where raw data files are stored or downloaded.
    processed_dir : str or Path
        Directory where pre-processed datasets are written.
    chembl_target : str
        ChEMBL target identifier, e.g. 'CHEMBL1873'.
    min_compounds : int
        Minimum number of compounds required before training proceeds.
        Set this to avoid training on trivially small datasets.
    """

    raw_dir: Path = Field(default=Path("data/raw"), description="Raw data directory")
    processed_dir: Path = Field(
        default=Path("data/processed"), description="Processed data directory"
    )
    chembl_target: str = Field(
        default="CHEMBL1873",
        description="ChEMBL target ID, e.g. CHEMBL1873 for InhA (M. tuberculosis)",
    )
    min_compounds: int = Field(
        default=50,
        ge=1,
        description="Minimum number of compounds required to proceed with training",
    )

    @field_validator("chembl_target")
    @classmethod
    def chembl_target_format(cls, v: str) -> str:
        v = v.strip().upper()
        if not v.startswith("CHEMBL"):
            raise ValueError(
                f"chembl_target must start with 'CHEMBL', got {v!r}. "
                "Example: 'CHEMBL1873'"
            )
        return v

    @field_validator("raw_dir", "processed_dir", mode="before")
    @classmethod
    def coerce_path(cls, v: Any) -> Path:
        return Path(v)


class QSARConfig(BaseModel):
    """Hyperparameters for the QSAR Random Forest model.

    Parameters
    ----------
    n_estimators : int
        Number of decision trees.  More trees = better calibration but slower.
    max_depth : int or None
        Maximum tree depth.  None means trees grow until leaves are pure.
    activity_threshold : float
        pIC50 cut-off used to binarise continuous activity values for
        classification.  Compounds with pIC50 >= threshold are 'active'.
    test_size : float
        Fraction of data held out as the final test set (0 < test_size < 1).
    val_size : float
        Fraction of data held out for validation / early stopping.
    n_folds : int
        Number of cross-validation folds.
    """

    n_estimators: int = Field(default=200, description="Number of RF trees")
    max_depth: Optional[int] = Field(
        default=None, ge=1, description="Max tree depth (None = unlimited)"
    )
    activity_threshold: float = Field(
        default=6.3,
        gt=0.0,
        description="pIC50 cut-off for active/inactive labelling",
    )
    test_size: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Proportion of data reserved for the held-out test set",
    )
    val_size: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Proportion of data reserved for validation",
    )
    n_folds: int = Field(default=5, ge=2, description="Number of CV folds")
    random_seed: int = Field(default=42, description="Global random seed")
    class_weight: Literal["balanced", "balanced_subsample", "none"] = Field(
        default="balanced",
        description="Class weight strategy for handling imbalanced activity data",
    )

    @field_validator("n_estimators")
    @classmethod
    def positive_estimators(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"n_estimators must be > 0, got {v}")
        return v

    @model_validator(mode="after")
    def split_sizes_valid(self) -> "QSARConfig":
        total = self.test_size + self.val_size
        if total >= 1.0:
            raise ValueError(
                f"test_size ({self.test_size}) + val_size ({self.val_size}) = {total:.3f} "
                "must be strictly less than 1.0 — otherwise no training data remains."
            )
        return self


class SplittingConfig(BaseModel):
    """Controls train / val / test dataset splitting strategy.

    Parameters
    ----------
    method : {'scaffold', 'random'}
        'scaffold' uses Bemis-Murcko scaffold splits (recommended — avoids
        optimistic bias from training/test structural similarity).
        'random' is faster but less realistic.
    frac_train : float
        Fraction of data for training.
    frac_val : float
        Fraction of data for validation.
    frac_test : float
        Fraction of data for testing.  Must satisfy frac_train + frac_val + frac_test == 1.0.
    """

    method: Literal["scaffold", "random"] = Field(
        default="scaffold",
        description="Splitting strategy: 'scaffold' (recommended) or 'random'",
    )
    frac_train: float = Field(default=0.8, gt=0.0, lt=1.0)
    frac_val: float = Field(default=0.1, gt=0.0, lt=1.0)
    frac_test: float = Field(default=0.1, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def fractions_sum_to_one(self) -> "SplittingConfig":
        total = self.frac_train + self.frac_val + self.frac_test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"frac_train + frac_val + frac_test must equal 1.0, "
                f"got {self.frac_train} + {self.frac_val} + {self.frac_test} = {total:.6f}"
            )
        return self


class VAEConfig(BaseModel):
    """Hyperparameters for the SMILES Variational Autoencoder.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent space.
    hidden_dim : int
        Number of units in GRU hidden layers.
    embedding_dim : int
        Character embedding size.
    encoder_layers : int
        Number of stacked GRU layers in the encoder.
    decoder_layers : int
        Number of stacked GRU layers in the decoder.
    dropout : float
        Dropout rate applied to GRU outputs.
    learning_rate : float
        Initial learning rate for Adam optimiser.
    batch_size : int
        Training mini-batch size.
    max_epochs : int
        Maximum number of training epochs.
    kl_weight : float
        Initial KL-divergence loss weight (beta-VAE style).
    kl_annealing_epochs : int
        Number of epochs over which kl_weight is linearly ramped from 0 to
        its final value (KL annealing prevents posterior collapse).
    max_smiles_len : int
        Maximum length of SMILES strings (longer sequences are truncated).
    """

    latent_dim: int = Field(default=256, ge=8, description="VAE latent space dimension")
    hidden_dim: int = Field(default=512, ge=16, description="GRU hidden dimension")
    embedding_dim: int = Field(default=128, ge=8, description="Character embedding size")
    encoder_layers: int = Field(default=2, ge=1, le=6, description="Encoder GRU layers")
    decoder_layers: int = Field(default=2, ge=1, le=6, description="Decoder GRU layers")
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0, description="Dropout rate")
    learning_rate: float = Field(default=1e-3, gt=0.0, description="Adam learning rate")
    batch_size: int = Field(default=256, ge=1, description="Training batch size")
    max_epochs: int = Field(default=100, ge=1, description="Maximum training epochs")
    kl_weight: float = Field(
        default=1.0, ge=0.0, description="KL divergence loss weight (beta)"
    )
    kl_annealing_epochs: int = Field(
        default=10,
        ge=0,
        description="Epochs to linearly ramp KL weight from 0 to kl_weight",
    )
    max_smiles_len: int = Field(
        default=120, ge=10, description="Maximum SMILES sequence length"
    )
    teacher_forcing_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability of using ground-truth token during decoder training",
    )

    @field_validator("learning_rate")
    @classmethod
    def lr_sensible(cls, v: float) -> float:
        if v > 1.0:
            raise ValueError(
                f"learning_rate {v} seems unreasonably large for Adam. "
                "Typical values: 1e-4 to 1e-2."
            )
        return v


class PipelineConfig(BaseModel):
    """Top-level configuration object for the entire TB drug discovery pipeline.

    Contains nested sub-configs for each component.  Load from a YAML file
    with ``PipelineConfig.from_yaml(path)`` or validate an existing file with
    ``validate_config(path)``.

    Parameters
    ----------
    data : DataConfig
    qsar : QSARConfig
    splitting : SplittingConfig
    vae : VAEConfig
    experiment_name : str
        Human-readable name for MLflow / W&B experiment tracking.
    output_dir : Path
        Directory where results, plots, and saved models are written.
    log_level : str
        Python logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
    """

    data: DataConfig = Field(default_factory=DataConfig)
    qsar: QSARConfig = Field(default_factory=QSARConfig)
    splitting: SplittingConfig = Field(default_factory=SplittingConfig)
    vae: VAEConfig = Field(default_factory=VAEConfig)
    experiment_name: str = Field(
        default="tb_drug_discovery",
        description="Name used for experiment tracking (MLflow / W&B)",
    )
    output_dir: Path = Field(
        default=Path("outputs"), description="Directory for results and saved models"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Python logging level"
    )

    @field_validator("output_dir", mode="before")
    @classmethod
    def coerce_output_path(cls, v: Any) -> Path:
        return Path(v)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PipelineConfig":
        """Load and validate configuration from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to a YAML configuration file.

        Returns
        -------
        PipelineConfig

        Raises
        ------
        FileNotFoundError if the file does not exist.
        ConfigValidationError if any field fails validation.

        Example
        -------
        >>> cfg = PipelineConfig.from_yaml('configs/experiment_01.yaml')
        """
        try:
            import yaml  # PyYAML
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load config from YAML. "
                "Install it with: pip install pyyaml"
            ) from exc

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = yaml.safe_load(fh) or {}

        try:
            return cls.model_validate(raw)
        except Exception as exc:  # pydantic ValidationError
            violations = _extract_violations(exc)
            raise ConfigValidationError(violations) from exc

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Serialise the configuration to a YAML file.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required. pip install pyyaml") from exc

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to plain dict, converting Path objects to strings
        data = _config_to_dict(self.model_dump())
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)
        logger.info("Config written to %s", path)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_config(config_path: Union[str, Path]) -> PipelineConfig:
    """Load a YAML config file and validate it, raising on ANY violation.

    Unlike PipelineConfig.from_yaml which re-raises a pydantic error,
    this function collects ALL validation errors across nested models and
    presents them together in a single ConfigValidationError.

    Parameters
    ----------
    config_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    PipelineConfig
        Fully validated configuration object.

    Raises
    ------
    FileNotFoundError
    ConfigValidationError — lists every individual field violation.

    Example
    -------
    >>> try:
    ...     cfg = validate_config('configs/broken.yaml')
    ... except ConfigValidationError as e:
    ...     for v in e.violations:
    ...         print(v)
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required. pip install pyyaml") from exc

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    violations: List[str] = []

    try:
        cfg = PipelineConfig.model_validate(raw)
    except Exception as exc:
        violations = _extract_violations(exc)

    if violations:
        raise ConfigValidationError(violations)

    logger.info("Config validated successfully: %s", config_path)
    return cfg  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------

def _extract_violations(exc: Exception) -> List[str]:
    """Extract human-readable violation strings from a pydantic ValidationError."""
    violations: List[str] = []
    try:
        # pydantic v2 ValidationError
        for error in exc.errors():  # type: ignore[attr-defined]
            loc = " -> ".join(str(p) for p in error.get("loc", []))
            msg = error.get("msg", str(error))
            violations.append(f"{loc}: {msg}" if loc else msg)
    except AttributeError:
        violations.append(str(exc))
    return violations


def _config_to_dict(d: Any) -> Any:
    """Recursively convert Path objects in a dict to strings for YAML output."""
    if isinstance(d, dict):
        return {k: _config_to_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_config_to_dict(v) for v in d]
    if isinstance(d, Path):
        return str(d)
    return d
