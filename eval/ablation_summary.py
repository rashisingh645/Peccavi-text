"""
eval/ablation_summary.py
Multi-seed ablation averaging — Table 2 in paper.

Reads result JSONs for each ablation variant across all available seeds,
computes mean ± std for all headline metrics, prints a formatted table,
and saves a machine-readable JSON for import into LaTeX.

Usage:
    python eval/ablation_summary.py
    python eval/ablation_summary.py --seeds 7 42 123
    python eval/ablation_summary.py --output results/ablation_summary.json
"""

from __future__ import annotations
import argparse
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple

# ── metrics to extract and display ──────────────────────────────────────────

METRICS = [
    "auc_roc",
    "tpr_at_1fpr",
    "ppl_ratio",
    "avg_gpt4_quality",
    "avg_readability",
    "effective_score_final",
    "theta_final",
]

COL_LABELS = {
    "auc_roc":               "AUC-ROC",
    "tpr_at_1fpr":           "TPR@1%FPR",
    "ppl_ratio":             "PPL ratio",
    "avg_gpt4_quality":      "GPT-4 Q",
    "avg_readability":       "Readability",
    "effective_score_final": "S_eff",
    "theta_final":           "theta_final",
}

# ── variants to summarise ────────────────────────────────────────────────────
# Each entry: (display name, list of path templates with {seed} placeholder)
# Multiple templates = try each in order until one exists (handles naming variants).

VARIANTS: List[Tuple[str, List[str]]] = [
    ("PECCAVI (full)",         ["results/peccavi_s{seed}.json"]),
    ("PECCAVI (attack-aware)", ["results/peccavi_attack_aware_s{seed}.json"]),
    ("PECCAVI (high-nu)",      ["results/peccavi_high_nu_s{seed}.json"]),
    ("ablation: fixed theta",  ["results/ablation_fixed_s{seed}.json"]),
    ("ablation: no quality",   ["results/ablation_noq_s{seed}.json"]),
    ("ablation: no watermark", ["results/ablation_nowm_s{seed}.json"]),
    ("KGW (delta=2.0)",        ["results/kgw_s{seed}.json"]),
    ("KGW-Strong (delta=8.0)", ["results/kgw_strong_s{seed}.json"]),
    ("SIR",                    ["results/sir_s{seed}.json"]),
    ("DiPMark",                ["results/dipmark_s{seed}.json"]),
    ("SynthID-Text",           ["results/synthid_s{seed}.json"]),
]


def load_metrics(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    key = next(iter(data))
    record = data[key]
    return {m: record.get(m) for m in METRICS}


def mean_std(values: List) -> Tuple[Optional[float], Optional[float]]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def fmt(mean: Optional[float], std: Optional[float], decimals: int = 3) -> str:
    if mean is None:
        return "—"
    if std is not None and std > 0:
        return f"{mean:.{decimals}f}±{std:.{decimals}f}"
    return f"{mean:.{decimals}f}"


def main():
    parser = argparse.ArgumentParser(description="Multi-seed ablation table")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 123])
    parser.add_argument("--output", default="results/ablation_summary.json")
    args = parser.parse_args()

    rows = []

    for variant_name, templates in VARIANTS:
        metric_values: Dict[str, List] = {m: [] for m in METRICS}
        found_seeds = []

        for seed in args.seeds:
            for tmpl in templates:
                path = tmpl.format(seed=seed)
                record = load_metrics(path)
                if record is not None:
                    found_seeds.append(seed)
                    for m in METRICS:
                        metric_values[m].append(record[m])
                    break  # found this seed, move to next

        if not found_seeds:
            continue  # variant has no results yet — skip silently

        row: Dict = {"variant": variant_name, "seeds_found": found_seeds, "n": len(found_seeds)}
        for m in METRICS:
            mean, std = mean_std(metric_values[m])
            row[f"{m}_mean"] = round(mean, 4) if mean is not None else None
            row[f"{m}_std"]  = round(std,  4) if std  is not None else None
        rows.append(row)

    # ── print table ──────────────────────────────────────────────────────────
    primary_metrics = ["auc_roc", "tpr_at_1fpr", "ppl_ratio", "avg_gpt4_quality"]
    col_w = 16

    header = f"{'Variant':<32}" + "".join(f"{COL_LABELS[m]:>{col_w}}" for m in primary_metrics) + f"{'seeds':>8}"
    separator = "-" * len(header)

    print(f"\n{'ABLATION SUMMARY':^{len(header)}}")
    print(separator)
    print(header)
    print(separator)

    for row in rows:
        name = row["variant"]
        seeds_str = str(row["seeds_found"])
        cols = []
        for m in primary_metrics:
            mean = row.get(f"{m}_mean")
            std  = row.get(f"{m}_std")
            decimals = 3 if m in ("auc_roc", "tpr_at_1fpr", "ppl_ratio") else 2
            cols.append(fmt(mean, std, decimals))
        line = f"{name:<32}" + "".join(f"{c:>{col_w}}" for c in cols) + f"  {seeds_str}"
        print(line)

    print(separator)
    print(f"\nFull metrics (theta_final, S_eff, readability) saved to {args.output}\n")

    # ── also print full metrics per variant ─────────────────────────────────
    for row in rows:
        n = row["n"]
        print(f"  {row['variant']}  (n={n} seeds: {row['seeds_found']})")
        for m in METRICS:
            mean = row.get(f"{m}_mean")
            std  = row.get(f"{m}_std")
            decimals = 3 if m != "avg_gpt4_quality" else 2
            print(f"    {COL_LABELS[m]:<18} {fmt(mean, std, decimals)}")
        print()

    # ── save JSON ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    summary_out = {row["variant"]: row for row in rows}
    with open(args.output, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
