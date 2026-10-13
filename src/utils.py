"""Shared utilities: seeding, config I/O, logging, environment reporting.

Everything that touches reproducibility lives here so every experiment
records seeds, library versions and the resolved configuration the same way.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import platform
import random
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

TRACKED_PACKAGES = [
    "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "cma",
    "pennylane", "pennylane-lightning", "qiskit", "torch",
    "matplotlib", "seaborn", "pyyaml", "pytest",
]


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------
def set_global_seed(seed: int) -> np.random.Generator:
    """Seed Python, NumPy and (if installed) PyTorch. Returns a NumPy Generator.

    Search engines should use the returned Generator (not the global NumPy
    state) so that seeds stay independent across methods.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: Optional[os.PathLike] = None,
                overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load YAML config; `overrides` is a nested dict merged on top."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def config_hash(cfg: Dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def resolve_path(p: os.PathLike) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    root = resolve_path(cfg["project"]["results_dir"])
    for sub in ("figures", "tables", "raw"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    resolve_path(cfg["project"]["models_dir"]).mkdir(parents=True, exist_ok=True)
    resolve_path(cfg["data"]["raw_dir"]).mkdir(parents=True, exist_ok=True)
    resolve_path(cfg["data"]["processed_dir"]).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# JSON / logging
# --------------------------------------------------------------------------
class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: os.PathLike) -> Path:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=_NumpyEncoder)
    return path


def save_run_config(cfg: Dict[str, Any], run_name: str, seed: int,
                    extra: Optional[Dict[str, Any]] = None) -> Path:
    """Persist the resolved config + environment next to the run's results."""
    record = {
        "run_name": run_name,
        "seed": seed,
        "config_hash": config_hash(cfg),
        "config": cfg,
        "environment": environment_report(),
        "extra": extra or {},
    }
    return save_json(record, Path(cfg["project"]["results_dir"]) / "raw"
                     / f"{run_name}_seed{seed}_config.json")


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                                         "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
def environment_report() -> Dict[str, Any]:
    pkgs = {}
    for name in TRACKED_PACKAGES:
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": pkgs,
    }
