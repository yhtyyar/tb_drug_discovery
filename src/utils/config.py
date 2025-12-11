"""Configuration management for the TB Drug Discovery pipeline.

This module provides configuration loading and management utilities.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class Config:
    """Configuration manager for the TB Drug Discovery pipeline.
    
    Handles loading configuration from YAML files and provides
    default values for all pipeline parameters.
    
    Attributes:
        config: Dictionary containing all configuration parameters.
        
    Example:
        >>> config = Config()
        >>> config.get("random_seed")
        42
        >>> config.get("qsar.n_estimators")
        100
    """
    
    # Default configuration values
    DEFAULTS: Dict[str, Any] = {
        "random_seed": 42,
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "chembl_target": "CHEMBL1849",  # InhA target ID
            "min_compounds": 500,
        },
        "qsar": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "test_size": 0.15,
            "val_size": 0.15,
            "n_folds": 5,
            "activity_threshold": 6.0,  # pIC50 threshold for active/inactive
        },
        "descriptors": {
            "lipinski": True,
            "topological": True,
            "extended": True,
        },
        "paths": {
            "models": "models",
            "results": "results",
            "figures": "results/figures",
            "metrics": "results/metrics",
        },
    }
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize configuration.
        
        Args:
            config_path: Optional path to YAML configuration file.
                If not provided, uses default values.
        """
        self.config = self.DEFAULTS.copy()
        
        if config_path and Path(config_path).exists():
            self._load_yaml(config_path)
            
        # Set random seeds for reproducibility
        self._set_random_seeds()
    
    def _load_yaml(self, path: str) -> None:
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file.
        """
        with open(path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f)
            
        if user_config:
            self._deep_update(self.config, user_config)
    
    def _deep_update(self, base: Dict, update: Dict) -> None:
        """Recursively update nested dictionary.
        
        Args:
            base: Base dictionary to update.
            update: Dictionary with updates.
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        import random
        import numpy as np
        
        seed = self.config["random_seed"]
        random.seed(seed)
        np.random.seed(seed)
        
        # Try to set PyTorch seed if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'qsar.n_estimators').
            default: Default value if key not found.
            
        Returns:
            Configuration value or default.
            
        Example:
            >>> config.get("qsar.n_estimators")
            100
        """
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation).
            value: Value to set.
        """
        keys = key.split(".")
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
    
    def save(self, path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            path: Path to save configuration file.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config({self.config})"
