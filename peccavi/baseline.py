"""
peccavi/baseline.py
Baseline loading and metrics helpers for the PECCAVI app and evaluation.
"""

from __future__ import annotations
from pathlib import Path
import json
import yaml

CONFIG_PATH = Path("configs") / "experiments.yaml"
BASELINE_RESULTS_PATH = Path("results") / "benchmark_results.json"


def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_baseline_results(path: Path = BASELINE_RESULTS_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle) or {}
    except Exception:
        return {}


def get_baseline_metrics() -> dict:
    results = load_baseline_results()
    baseline_metrics: dict = {}
    for model_name, entry in results.items():
        if not isinstance(entry, dict) or "error" in entry:
            continue
        baseline_metrics[model_name] = {
            "robustness": float(entry.get("robustness", 0.0)),
            "resilience": float(entry.get("resilience", 0.0)),
            "fpr": float(entry.get("fpr", 0.0)),
            "auc": float(entry.get("auc_roc", entry.get("auc", 0.0))),
            "readability": float(entry.get("readability", 0.0)),
        }
    return baseline_metrics


def get_baseline_names() -> list[str]:
    cfg = load_config()
    return [b.get("name") for b in cfg.get("baseline_models", []) if b.get("name")]
