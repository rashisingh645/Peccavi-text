"""
eval/confidence_intervals.py
Computes mean ± 95% CI across seeds for key metrics.

Groups result files by experiment name (prefix before _s<seed>.json) and
prints a table with mean and 95% confidence interval for each metric.

Usage:
    python eval/confidence_intervals.py results/peccavi_s*.json results/kgw_s*.json
    python eval/confidence_intervals.py results/*.json
"""

from __future__ import annotations
import json
import math
import sys
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

METRICS = [
    ("auc_roc",           "AUC-ROC"),
    ("tpr_at_1fpr",       "TPR@1%FPR"),
    ("ppl_ratio",         "PPL ratio"),
    ("avg_readability",   "Readability"),
    ("avg_retention_rate","Retention"),
    ("false_positive_rate","FPR@z4"),
]

# t* values for 95% CI: index = n-2 (n=2→12.71, n=3→4.303, n=4→3.182)
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}


def _inner(data: dict) -> dict:
    keys = list(data.keys())
    if len(keys) == 1 and isinstance(data[keys[0]], dict):
        inner = data[keys[0]]
        if "watermark_mode" in inner or "auc_roc" in inner:
            return inner
    return data


def _ci95(values: List[float]) -> tuple:
    n = len(values)
    if n == 1:
        return values[0], float("nan")
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    t = T95.get(n, 1.96)
    return mean, t * se


def _experiment_key(stem: str) -> str:
    """Strip trailing _s<digits> to group seeds together."""
    return re.sub(r"_s\d+$", "", stem)


def main():
    import glob as _glob
    raw_paths = sys.argv[1:]
    if not raw_paths:
        print("Usage: python eval/confidence_intervals.py <results.json> [...]")
        sys.exit(1)

    # Expand glob patterns (needed on Windows where the shell doesn't expand them)
    paths = []
    for p in raw_paths:
        expanded = _glob.glob(p)
        if expanded:
            paths.extend(expanded)
        else:
            paths.append(p)  # keep as-is; will fail with a clear error below

    groups: Dict[str, List[dict]] = defaultdict(list)
    for path in paths:
        p = Path(path)
        try:
            with open(p) as f:
                data = json.load(f)
            inner = _inner(data)
            key = _experiment_key(p.stem)
            groups[key].append(inner)
        except Exception as e:
            print(f"  SKIP {path}: {e}")

    col_w = 18
    print(f"\n{'Experiment':<30}" + "".join(f"  {label:<{col_w}}" for _, label in METRICS))
    print("-" * (30 + len(METRICS) * (col_w + 2)))

    for exp, records in sorted(groups.items()):
        row = f"{exp:<30}"
        for key, _ in METRICS:
            vals = [r[key] for r in records if isinstance(r.get(key), (int, float)) and r[key] == r[key]]
            if not vals:
                row += f"  {'—':<{col_w}}"
                continue
            mean, ci = _ci95(vals)
            if math.isnan(ci):
                cell = f"{mean:.3f}"
            else:
                cell = f"{mean:.3f}±{ci:.3f}"
            row += f"  {cell:<{col_w}}"
        print(row)


if __name__ == "__main__":
    main()
