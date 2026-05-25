"""
eval/plot_theta_entropy.py
Figure 2: θ vs prompt entropy scatter plot.

Shows that PECCAVI's adaptive policy assigns higher watermark strength to
high-entropy (creative/open-ended) prompts and lower strength to low-entropy
(factual/constrained) prompts. Points are coloured by training generation to
show when the entropy–theta correlation emerges as the weight vector w is learned.

Usage:
    python eval/plot_theta_entropy.py
    python eval/plot_theta_entropy.py --inputs results/peccavi_s7.json results/peccavi_s42.json results/peccavi_s123.json
    python eval/plot_theta_entropy.py --output figures/theta_entropy.pdf
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from scipy.stats import linregress, pearsonr
except ImportError as e:
    sys.exit(f"Missing dependency: {e}. Run: pip install matplotlib scipy")


DEFAULT_INPUTS = [
    "results/peccavi_s7.json",
    "results/peccavi_s42.json",
    "results/peccavi_s123.json",
]
SEED_COLORS = ["#1976D2", "#E64A19", "#388E3C"]  # blue, orange, green
QUARTILE_COLOR = "#37474F"


def load_theta_by_prompt(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    key = next(iter(data))
    points = data[key].get("theta_by_prompt", [])
    return points


def quartile_means(entropies: np.ndarray, thetas: np.ndarray):
    q25, q50, q75 = np.percentile(entropies, [25, 50, 75])
    bins = [
        (entropies <= q25, "Q1"),
        ((entropies > q25) & (entropies <= q50), "Q2"),
        ((entropies > q50) & (entropies <= q75), "Q3"),
        (entropies > q75, "Q4"),
    ]
    centres, means, stds = [], [], []
    for mask, _ in bins:
        if mask.sum() > 0:
            centres.append(float(np.median(entropies[mask])))
            means.append(float(np.mean(thetas[mask])))
            stds.append(float(np.std(thetas[mask])))
    return np.array(centres), np.array(means), np.array(stds)


def main():
    parser = argparse.ArgumentParser(description="Plot θ vs prompt entropy (Figure 2)")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS,
                        help="PECCAVI result JSON files (one per seed)")
    parser.add_argument("--output", default="figures/theta_entropy.pdf",
                        help="Output path (.pdf and .png will both be saved)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    all_entropy, all_theta = [], []
    plotted_any = False

    for i, path in enumerate(args.inputs):
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        points = load_theta_by_prompt(path)
        if not points:
            print(f"[skip] no theta_by_prompt data in {path} (adaptive_theta may be False)")
            continue

        entropies = np.array([p["entropy"] for p in points])
        thetas = np.array([p["theta_context"] for p in points])
        n = len(entropies)

        seed_label = os.path.basename(path).replace(".json", "").split("_s")[-1]
        color = SEED_COLORS[i % len(SEED_COLORS)]

        # Colour scatter by generation index so early (untrained) vs late (learned) is visible
        gen_indices = np.arange(n)
        scatter = ax.scatter(
            entropies, thetas,
            c=gen_indices, cmap="Blues", vmin=0, vmax=n,
            alpha=0.55, s=28, edgecolors=color, linewidths=0.4,
            label=f"seed {seed_label} (n={n})", zorder=2,
        )

        all_entropy.extend(entropies.tolist())
        all_theta.extend(thetas.tolist())
        plotted_any = True

    if not plotted_any:
        sys.exit("No data found. Run PECCAVI training with adaptive_theta=True first.")

    x = np.array(all_entropy)
    y = np.array(all_theta)

    # Linear regression line
    slope, intercept, r, p_val, _ = linregress(x, y)
    r_pearson, p_pearson = pearsonr(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(
        x_line, slope * x_line + intercept,
        color="black", linewidth=1.8, linestyle="--", zorder=4,
        label=f"Linear fit  r={r_pearson:.2f}, p={p_pearson:.3f}",
    )

    # Quartile means overlay
    qx, qy, qstd = quartile_means(x, y)
    ax.errorbar(
        qx, qy, yerr=qstd,
        fmt="D", color=QUARTILE_COLOR, markersize=7, capsize=4,
        linewidth=1.5, zorder=5, label="Quartile mean ± std",
    )

    ax.set_xlabel("Prompt entropy (normalised token-level entropy)", fontsize=12)
    ax.set_ylabel("θ_context (watermark strength assigned to prompt)", fontsize=12)
    ax.set_title("PECCAVI: Content-Adaptive Watermark Strength\n"
                 "Higher-entropy prompts receive stronger watermarks", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25, linestyle=":")

    # Annotation: Q1 vs Q4 spread
    q25_thresh = np.percentile(x, 25)
    q75_thresh = np.percentile(x, 75)
    q1_mean = float(np.mean(y[x <= q25_thresh]))
    q4_mean = float(np.mean(y[x >= q75_thresh]))
    spread = q4_mean - q1_mean
    ax.annotate(
        f"Q4 – Q1 spread: {spread:+.3f}",
        xy=(0.97, 0.05), xycoords="axes fraction",
        ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.8),
    )

    plt.tight_layout()
    pdf_path = args.output if args.output.endswith(".pdf") else args.output + ".pdf"
    png_path = pdf_path.replace(".pdf", ".png")
    plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"\nStats (all seeds pooled, n={len(x)}):")
    print(f"  Pearson r = {r_pearson:.3f}  (p = {p_pearson:.4f})")
    print(f"  Q1 mean theta = {q1_mean:.3f}  |  Q4 mean theta = {q4_mean:.3f}  |  spread = {spread:+.3f}")
    print(f"  Linear slope = {slope:.4f}  (theta per unit entropy)")


if __name__ == "__main__":
    main()
