"""
scripts/run_ablations.py
Runs all PECCAVI ablations and the KGW baseline in sequence.
Each experiment saves its results to ./results/<name>.json.
Run from repo root:
    python scripts/run_ablations.py
"""

from __future__ import annotations
import subprocess
import sys
import os

CONFIG_FILE = "configs/experiments.yaml"

EXPERIMENTS = [
    {
        "name": "peccavi",
        "profile": "peccavi",
        "mode": "train",
        "output": "results/peccavi.json",
        "seed": 42,
    },
    {
        "name": "peccavi_distortion_free",
        "profile": "peccavi_distortion_free",
        "mode": "train",
        "output": "results/peccavi_df.json",
        "seed": 42,
    },
    {
        "name": "kgw_baseline",
        "profile": "kgw_baseline",
        "mode": "kgw",
        "output": "results/kgw_baseline.json",
        "seed": 42,
    },
    {
        "name": "sir_baseline",
        "profile": "sir_baseline",
        "mode": "sir",
        "output": "results/sir_baseline.json",
        "seed": 42,
    },
    {
        "name": "dipmark_baseline",
        "profile": "dipmark_baseline",
        "mode": "dipmark",
        "output": "results/dipmark_baseline.json",
        "seed": 42,
    },
    {
        "name": "synthid_baseline",
        "profile": "synthid_baseline",
        "mode": "synthid",
        "output": "results/synthid_baseline.json",
        "seed": 42,
    },
    {
        "name": "ablation_fixed_theta",
        "profile": "ablation_fixed_theta",
        "mode": "train",   # reads alpha/nu/watermark_mode from the ablation profile
        "output": "results/ablation_fixed_theta.json",
        "seed": 42,
    },
    {
        "name": "ablation_no_quality",
        "profile": "ablation_no_quality",
        "mode": "train",
        "output": "results/ablation_no_quality.json",
        "seed": 42,
    },
    {
        "name": "ablation_no_watermark",
        "profile": "ablation_no_watermark",
        "mode": "train",
        "output": "results/ablation_no_watermark.json",
        "seed": 42,
    },
]

# Seeds for multi-seed runs (set MULTI_SEED=1 to enable)
SEEDS = [42, 123, 7]


def run_experiment(exp: dict, seed: int = None):
    s = seed if seed is not None else exp["seed"]
    cmd = [
        sys.executable, "main.py",
        "--mode", exp["mode"],
        "--output", exp["output"],
        "--config-file", CONFIG_FILE,
        "--profile", exp["profile"],
        "--seed", str(s),
    ]
    print(f"\n{'='*60}")
    print(f"  Running: {exp['name']} (seed={s})")
    print(f"  Config:  {CONFIG_FILE} [profile={exp['profile']}]")
    print(f"  Output:  {exp['output']}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  WARNING: {exp['name']} exited with code {result.returncode}")


def main():
    os.makedirs("results", exist_ok=True)
    multi_seed = os.getenv("MULTI_SEED", "0") == "1"

    if multi_seed:
        print("Running all experiments with 3 seeds...")
        for exp in EXPERIMENTS:
            for i, seed in enumerate(SEEDS):
                out = exp["output"].replace(".json", f"_seed{seed}.json")
                run_experiment({**exp, "output": out}, seed=seed)
    else:
        print("Running all experiments with seed=42...")
        for exp in EXPERIMENTS:
            run_experiment(exp)

    # Auto-generate comparison table
    result_files = [exp["output"] for exp in EXPERIMENTS if os.path.exists(exp["output"])]
    if result_files:
        print(f"\n{'='*60}")
        print("  Generating comparison table...")
        subprocess.run([sys.executable, "eval/compare.py"] + result_files)

    # Generate Pareto curve data and figure (Option 3: unified framework)
    print(f"\n{'='*60}")
    print("  Running Pareto sweep (KGW/SIR delta sweep for Figure 1)...")
    print("  Sweeps delta in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] — takes ~30–60 min.")
    sweep_result = subprocess.run(
        [sys.executable, "scripts/pareto_sweep.py"], check=False
    )
    if sweep_result.returncode == 0:
        print("  Rendering Pareto curve figure...")
        subprocess.run([sys.executable, "eval/plot_pareto.py"], check=False)
    else:
        print("  WARNING: Pareto sweep failed — run manually: python scripts/pareto_sweep.py")


if __name__ == "__main__":
    main()
