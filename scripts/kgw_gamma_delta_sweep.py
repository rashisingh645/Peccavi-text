"""
scripts/kgw_gamma_delta_sweep.py
Sweeps KGW's gamma x delta grid at a fixed generation length, matching the original
KGW paper's own ablation (Kirchenbauer et al. 2023 test gamma in {0.1, 0.25, 0.5} and
delta in {1.0, 2.0, 5.0}, finding gamma=0.1 Pareto-optimal) — our earlier kgw_matched_long
run used a single fixed (gamma=0.5, delta=2.0) point, not KGW's own best-found setting.

Loads the backbone once and reuses it across all 9 runs (skips 9x reload overhead).
Each combo is wrapped in try/except so one failure doesn't kill the whole sweep, and
results are written after every combo (not just at the end) so a crash partway through
still leaves usable partial results.

Usage:
    python scripts/kgw_gamma_delta_sweep.py
    python scripts/kgw_gamma_delta_sweep.py --max-tokens 250 --n-eval 100 --n-attack 100 --seed 42
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time

# `python scripts/kgw_gamma_delta_sweep.py` doesn't put the repo root on sys.path
# (only the script's own directory), so `from main import ...` below can't find
# main.py unless we add it explicitly — same fix backbone/model.py already uses.
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GAMMAS = [0.1, 0.25, 0.5]
DELTAS = [1.0, 2.0, 5.0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="configs/experiments.yaml")
    p.add_argument("--profile", default="kgw_matched_long", help="Profile to source the model/backbone config from")
    p.add_argument("--max-tokens", type=int, default=250)
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--n-attack", type=int, default=100)
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--gammas", default=None, help="Comma-separated gamma values, overrides the default 0.1,0.25,0.5 grid")
    p.add_argument("--deltas", default=None, help="Comma-separated delta values, overrides the default 1.0,2.0,5.0 grid")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gammas = [float(x) for x in args.gammas.split(",")] if args.gammas else GAMMAS
    deltas = [float(x) for x in args.deltas.split(",")] if args.deltas else DELTAS

    from main import init_backbone, load_profile
    from eval.watermark import run_peccavi

    logger.info(f"Initialising backbone from profile '{args.profile}'...")
    backbone = init_backbone(load_profile(args.config_file, args.profile))

    rows = []
    total = len(gammas) * len(deltas)
    done = 0
    for gamma in gammas:
        for delta in deltas:
            done += 1
            tag = f"g{gamma}_d{delta}"
            out_path = os.path.join(args.out_dir, f"kgw_sweep_{tag}.json")
            logger.info(f"[{done}/{total}] gamma={gamma} delta={delta} -> {out_path}")
            t0 = time.time()
            try:
                summary = run_peccavi(
                    backbone,
                    generations=args.generations,
                    n_eval_samples=args.n_eval,
                    n_attack_samples=args.n_attack,
                    max_tokens=args.max_tokens,
                    verbose=False,
                    watermark_mode="kgw",
                    kgw_delta=delta,
                    kgw_gamma=gamma,
                    seed=args.seed,
                )
                elapsed = time.time() - t0
                report = {k: v for k, v in summary.items() if k != "detailed_records"}
                report["gamma"] = gamma
                report["delta"] = delta
                report["elapsed_sec"] = round(elapsed, 1)
                with open(out_path, "w") as f:
                    json.dump({"kgw_baseline": report}, f, indent=2, default=str)

                aa = summary.get("attack_auc", {})
                asurv = summary.get("attack_survival", {})
                rows.append({
                    "gamma": gamma, "delta": delta,
                    "auc_roc": summary.get("auc_roc"),
                    "tpr_at_1fpr": summary.get("tpr_at_1fpr"),
                    "attack_auc_syntactic": aa.get("syntactic"),
                    "attack_survival_syntactic_z4": asurv.get("syntactic"),
                    "elapsed_sec": round(elapsed, 1),
                })
                logger.info(
                    f"  done in {elapsed:.0f}s | AUC={summary.get('auc_roc')} "
                    f"| syntactic attack AUC={aa.get('syntactic')} "
                    f"| syntactic z4 survival={asurv.get('syntactic')}"
                )
            except Exception as e:
                elapsed = time.time() - t0
                logger.warning(f"  FAILED after {elapsed:.0f}s: {e}")
                rows.append({"gamma": gamma, "delta": delta, "error": str(e), "elapsed_sec": round(elapsed, 1)})

    summary_path = os.path.join(args.out_dir, "kgw_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)

    print("\n" + "=" * 78)
    print("  KGW gamma x delta SWEEP SUMMARY")
    print("=" * 78)
    header = f"{'gamma':>6} {'delta':>6} {'AUC':>8} {'TPR@1%FPR':>10} {'synAttAUC':>10} {'synZ4':>7} {'sec':>6}"
    print(header)
    for r in rows:
        if "error" in r:
            print(f"{r['gamma']:>6} {r['delta']:>6}  ERROR: {r['error'][:50]}")
        else:
            print(f"{r['gamma']:>6} {r['delta']:>6} {r['auc_roc']:>8} {r['tpr_at_1fpr']:>10} "
                  f"{r['attack_auc_syntactic']:>10} {r['attack_survival_syntactic_z4']:>7} {r['elapsed_sec']:>6}")
    print(f"\nSaved -> {summary_path}")


if __name__ == "__main__":
    main()
