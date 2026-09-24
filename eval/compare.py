"""
eval/compare.py
Loads multiple benchmark result JSON files and prints a comparison table.
Usage:
    python eval/compare.py results/peccavi.json results/kgw.json results/ablation_fixed_theta.json
"""

from __future__ import annotations
import json
import sys
import os
import math
from typing import Dict, List, Optional


METRICS = [
    ("auc_roc",               "AUC-ROC",          "{:.4f}", "≥0.90"),
    ("tpr_at_1fpr",           "TPR@1%FPR",        "{:.4f}", "≥0.80"),
    ("false_positive_rate",   "FPR@z=4",          "{:.4f}", "≤0.05"),
    ("effective_score_final", "S_eff",             "{:.4f}", ">0.50"),
    ("ppl_ratio",             "PPL ratio",         "{:.4f}", "≤1.10"),
    ("theta_final",           "θ_final",           "{:.4f}", "—"),
    ("improvement_pct",       "Improvement %",     "{:.1f}",  "—"),
    ("avg_readability",       "Readability",       "{:.2f}",  "≥3.0"),
]


def load_results(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def flatten(results: Dict) -> Dict[str, Dict]:
    """Handle both single-model and multi-model result files."""
    flat = {}
    for key, val in results.items():
        if isinstance(val, dict) and "auc_roc" in val:
            flat[key] = val
        elif isinstance(val, dict) and not val.get("error"):
            flat[key] = val
    return flat


def print_table(all_results: Dict[str, Dict]):
    models = list(all_results.keys())
    col_w = max(20, max(len(m) for m in models) + 2)

    header = f"{'Metric':<22}" + "".join(f"{m:^{col_w}}" for m in models)
    print("\n" + "═" * len(header))
    print("  PECCAVI COMPARISON TABLE")
    print("═" * len(header))
    print(header)
    print("─" * len(header))

    for key, label, fmt, target in METRICS:
        row = f"{label + ' (' + target + ')':<22}"
        for model in models:
            val = all_results[model].get(key)
            if val is None:
                row += f"{'N/A':^{col_w}}"
            else:
                row += f"{fmt.format(val):^{col_w}}"
        print(row)

    print("═" * len(header))


def latex_table(all_results: Dict[str, Dict]) -> str:
    models = list(all_results.keys())
    col_spec = "l" + "r" * len(models)
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        "Metric & " + " & ".join(models) + " \\\\",
        "\\midrule",
    ]
    for key, label, fmt, target in METRICS:
        vals = []
        for model in models:
            val = all_results[model].get(key)
            vals.append(fmt.format(val) if val is not None else "—")
        lines.append(f"{label} & " + " & ".join(vals) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{PECCAVI vs baselines and ablations}",
        "\\label{tab:results}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def _std(vals: List[float]) -> Optional[float]:
    if len(vals) < 2:
        return None
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return math.sqrt(variance)


def aggregate_seeds(paths: List[str]) -> Dict[str, Dict]:
    """
    Given result files named <name>_seed42.json, <name>_seed123.json, etc.,
    group by base name and return mean ± std for each metric.
    Entries without a _seed suffix are treated as single-run results.
    """
    import re
    groups: Dict[str, List[Dict]] = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        raw = load_results(path)
        flat = flatten(raw)
        base = re.sub(r"_seed\d+$", "", os.path.splitext(os.path.basename(path))[0])
        groups.setdefault(base, [])
        for model, vals in flat.items():
            groups[base].append(vals)

    aggregated: Dict[str, Dict] = {}
    for base, runs in groups.items():
        if len(runs) == 1:
            aggregated[base] = runs[0]
            continue
        merged: Dict = {}
        for key, _, _, _ in METRICS:
            values = [r[key] for r in runs if r.get(key) is not None]
            if values:
                merged[key] = sum(values) / len(values)
                std = _std(values)
                merged[f"{key}_std"] = std
        # carry through non-metric fields from first run
        for k, v in runs[0].items():
            if k not in merged:
                merged[k] = v
        aggregated[base] = merged
    return aggregated


def print_table_with_std(all_results: Dict[str, Dict]):
    """Print table with mean ± std columns when multi-seed data is present."""
    models = list(all_results.keys())
    col_w = max(24, max(len(m) for m in models) + 2)

    header = f"{'Metric':<22}" + "".join(f"{m:^{col_w}}" for m in models)
    print("\n" + "═" * len(header))
    print("  PECCAVI COMPARISON TABLE  (mean ± std across seeds)")
    print("═" * len(header))
    print(header)
    print("─" * len(header))

    for key, label, fmt, target in METRICS:
        row = f"{label + ' (' + target + ')':<22}"
        for model in models:
            val = all_results[model].get(key)
            std = all_results[model].get(f"{key}_std")
            if val is None:
                row += f"{'N/A':^{col_w}}"
            elif std is not None:
                cell = f"{fmt.format(val)}±{fmt.format(std)}"
                row += f"{cell:^{col_w}}"
            else:
                row += f"{fmt.format(val):^{col_w}}"
        print(row)
    print("═" * len(header))


def latex_table_with_std(all_results: Dict[str, Dict]) -> str:
    models = list(all_results.keys())
    col_spec = "l" + "r" * len(models)
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        "Metric & " + " & ".join(f"\\textbf{{{m}}}" for m in models) + " \\\\",
        "\\midrule",
    ]
    for key, label, fmt, target in METRICS:
        vals = []
        for model in models:
            val = all_results[model].get(key)
            std = all_results[model].get(f"{key}_std")
            if val is None:
                vals.append("—")
            elif std is not None:
                vals.append(f"{fmt.format(val)} $\\pm$ {fmt.format(std)}")
            else:
                vals.append(fmt.format(val))
        lines.append(f"{label} & " + " & ".join(vals) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{PECCAVI vs. KGW, SIR baselines and ablations (mean $\\pm$ std, 3 seeds)}",
        "\\label{tab:results}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def main(paths: List[str]):
    # Detect if any paths look like multi-seed files (_seed42, _seed123, etc.)
    import re
    multi_seed = any(re.search(r"_seed\d+", p) for p in paths)

    if multi_seed:
        all_results = aggregate_seeds(paths)
        print_table_with_std(all_results)
        print("\n  LaTeX table (mean ± std):\n")
        print(latex_table_with_std(all_results))
    else:
        all_results = {}
        for path in paths:
            if not os.path.exists(path):
                print(f"  WARNING: {path} not found — skipping")
                continue
            raw = load_results(path)
            flat = flatten(raw)
            all_results.update(flat)

        if not all_results:
            print("No results found.")
            return

        print_table(all_results)
        print("\n  LaTeX table:\n")
        print(latex_table(all_results))

    out = "./results/comparison_table.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval/compare.py <result1.json> [result2.json ...]")
        sys.exit(1)
    main(sys.argv[1:])
