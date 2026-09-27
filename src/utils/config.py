"""
YAML configuration loader with validation.
"""
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import copy


class Config:
    """Configuration object with attribute-style access."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to plain dict."""
        result = {}
        for key, value in self._data.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        self._data[key] = value
        if isinstance(value, dict):
            setattr(self, key, Config(value))
        else:
            setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def update(self, other: Dict[str, Any]):
        """Update config with another dict."""
        def deep_update(d1: dict, d2: dict):
            for k, v in d2.items():
                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                    deep_update(d1[k], v)
                else:
                    d1[k] = v

        deep_update(self._data, other)
        # Rebuild attributes
        for key in list(self.__dict__.keys()):
            if key != "_data":
                delattr(self, key)
        for key, value in self._data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        return f"Config({self._data})"


def load_config(path: str) -> Config:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        Config object
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    return Config(data)


def merge_configs(base: Config, override: Config) -> Config:
    """Merge two configs, with override taking precedence."""
    merged = copy.deepcopy(base.to_dict())
    merged.update(override.to_dict())
    return Config(merged)


def save_config(config: Config, path: str):
    """Save config to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


# Default configuration template
DEFAULT_CONFIG = {
    "seed": 42,
    "device": "auto",  # auto, cuda, cpu

    "dataset": {
        "name": "facebook",  # facebook, twitter, gplus, ogbl-collab, ogbl-ppa, ogbl-ddi
        "root": "./data",
        "val_ratio": 0.05,
        "test_ratio": 0.10,
        "neg_sampling_ratio": 1.0,
    },

    "features": {
        "type": "structural",  # structural, identity, rwpe, laplacian_pe, learnable
        "structural": ["degree", "pagerank"],
        "rwpe_length": 16,
        "laplacian_k": 16,
        "learnable_dim": 64,
    },

    "model": {
        "type": "gcn",  # gcn, gat, gatv2, gin, graphconv, lightgcn, sage
        "hidden_channels": 128,
        "out_channels": 64,
        "num_layers": 2,
        "dropout": 0.3,
        "use_residual": False,
        "use_layer_norm": False,
        # GAT-specific
        "heads": 4,
        "attn_dropout": 0.0,
        # LightGCN-specific
        # SAGE-specific
        "aggr": "mean",
    },

    "predictor": {
        "type": "dot",  # dot, cosine, bilinear, mlp, hadamard_mlp
        "hidden_channels": 128,
        "num_layers": 2,
        "dropout": 0.3,
    },

    "training": {
        "epochs": 100,
        "batch_size": -1,  # -1 for full-batch
        "lr": 0.005,
        "weight_decay": 0.0,
        "optimizer": "adam",  # adam, adamw, sgd
        "loss": "bce",  # bce, bpr, focal
        "grad_clip": 1.0,
        "log_interval": 10,
    },

    "early_stopping": {
        "enabled": True,
        "patience": 10,
        "min_delta": 1e-4,
        "mode": "max",
        "restore_best_weights": True,
    },

    "lr_scheduler": {
        "enabled": True,
        "mode": "max",
        "factor": 0.5,
        "patience": 5,
        "min_lr": 1e-6,
    },

    "evaluation": {
        "metrics": ["auc", "ap", "precision@10", "recall@10", "ndcg@10", "hit@10", "mrr"],
        "ks": [5, 10, 20, 50],
        "run_baselines": True,
        "num_seeds": 5,
    },

    "logging": {
        "level": "INFO",
        "use_wandb": False,
        "wandb_project": "gnn-friend-rec",
        "wandb_entity": None,
        "log_dir": "./logs",
    },

    "checkpoint": {
        "save_dir": "./checkpoints",
        "save_best_only": True,
    },
}


def get_default_config() -> Config:
    """Get default configuration."""
    return Config(DEFAULT_CONFIG)


def validate_config(config: Config) -> bool:
    """Validate configuration."""
    # Check required fields
    required = ["dataset", "model", "training"]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required config section: {field}")

    # Validate dataset
    valid_datasets = ["facebook", "twitter", "gplus", "ogbl-collab", "ogbl-ppa", "ogbl-ddi"]
    if config.dataset.name not in valid_datasets:
        print(f"Warning: Unknown dataset {config.dataset.name}. Valid: {valid_datasets}")

    # Validate model
    valid_models = ["gcn", "gat", "gatv2", "gin", "graphconv", "lightgcn", "sage"]
    if config.model.type not in valid_models:
        raise ValueError(f"Unknown model type: {config.model.type}. Valid: {valid_models}")

    # Validate predictor
    valid_predictors = ["dot", "cosine", "bilinear", "mlp", "hadamard_mlp"]
    if config.predictor.type not in valid_predictors:
        raise ValueError(f"Unknown predictor type: {config.predictor.type}. Valid: {valid_predictors}")

    return True