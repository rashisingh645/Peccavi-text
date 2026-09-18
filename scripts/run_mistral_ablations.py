"""
scripts/run_mistral_ablations.py
Runs all PECCAVI experiments on Mistral-7B-Instruct-v0.3.
Mirrors run_ablations.py but uses mistral_* configs.

Run from repo root:
    python scripts/run_mistral_ablations.py
    MULTI_SEED=1 python scripts/run_mistral_ablations.py
"""

from __future__ import annotations
import subprocess
import sys
import os

CONFIG_FILE = "configs/experiments_mistral.yaml"

EXPERIMENTS = [
    {
        "name": "peccavi_mistral",
        "profile": "peccavi",
        "mode": "train",
        "output": "results/peccavi_mistral_s42.json",
        "seed": 42,
    },
    {
        "name": "kgw_mistral",
        "profile": "kgw_baseline",
        "mode": "kgw",
        "output": "results/kgw_mistral_s42.json",
        "seed": 42,
    },
    {
        "name": "sir_mistral",
        "profile": "sir_baseline",
        "mode": "sir",
        "output": "results/sir_mistral_s42.json",
        "seed": 42,
    },
    {
        "name": "ablation_fixed_mistral",
        "profile": "ablation_fixed_theta",
        "mode": "train",
        "output": "results/ablation_fixed_mistral_s42.json",
        "seed": 42,
    },
    {
        "name": "ablation_noq_mistral",
        "profile": "ablation_no_quality",
        "mode": "train",
        "output": "results/ablation_noq_mistral_s42.json",
        "seed": 42,
    },
    {
        "name": "ablation_nowm_mistral",
        "profile": "ablation_no_watermark",
        "mode": "train",
        "output": "results/ablation_nowm_mistral_s42.json",
        "seed": 42,
    },
]

SEEDS = [42, 123, 7]


def run_experiment(exp: dict, seed: int = None):
    s = seed if seed is not None else exp["seed"]
    out = exp["output"].replace("_s42.json", f"_s{s}.json")
    cmd = [
        sys.executable, "main.py",
        "--mode", exp["mode"],
        "--output", out,
        "--config-file", CONFIG_FILE,
        "--profile", exp["profile"],
        "--seed", str(s),
    ]
    print(f"\n{'='*60}")
    print(f"  Running: {exp['name']} (seed={s})")
    print(f"  Config:  {CONFIG_FILE} [profile={exp['profile']}]")
    print(f"  Output:  {out}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  WARNING: {exp['name']} exited with code {result.returncode}")


def main():
    os.makedirs("results", exist_ok=True)
    multi_seed = os.getenv("MULTI_SEED", "0") == "1"
    single_seed = os.getenv("SEED")  # SEED=123 python scripts/run_mistral_ablations.py

    if single_seed is not None:
        s = int(single_seed)
        print(f"Running Mistral experiments with seed={s}...")
        for exp in EXPERIMENTS:
            run_experiment(exp, seed=s)
    elif multi_seed:
        print("Running Mistral experiments with 3 seeds...")
        for exp in EXPERIMENTS:
            for seed in SEEDS:
                run_experiment(exp, seed=seed)
    else:
        print("Running Mistral experiments with seed=42...")
        for exp in EXPERIMENTS:
            run_experiment(exp)

    result_files = [
        exp["output"].replace("_s42.json", f"_s{SEEDS[0]}.json")
        for exp in EXPERIMENTS
        if os.path.exists(exp["output"].replace("_s42.json", f"_s{SEEDS[0]}.json"))
    ]
    if result_files:
        print(f"\n{'='*60}")
        print("  Generating comparison table...")
        subprocess.run([sys.executable, "eval/compare.py"] + result_files)


if __name__ == "__main__":
    main()
