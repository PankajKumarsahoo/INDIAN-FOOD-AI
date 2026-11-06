"""Config loading. Every path in config.yaml is resolved against the project root."""
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"

try:  # optional: load .env (GOOGLE_API_KEY etc.)
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def load_config(path=None):
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["paths"] = {k: (PROJECT_ROOT / v) for k, v in cfg["paths"].items()}

    nw = cfg["classifier"].get("num_workers", "auto")
    if nw == "auto":
        nw = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)
    cfg["classifier"]["num_workers"] = int(nw)
    return cfg


def ensure_dirs(cfg):
    for key, p in cfg["paths"].items():
        if key == "nutrition_db":
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)
