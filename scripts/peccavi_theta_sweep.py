"""
scripts/peccavi_theta_sweep.py
Sweeps PECCAVI's watermark strength theta at FIXED values (no REINFORCE training),
mirroring scripts/kgw_gamma_delta_sweep.py's gamma x delta grid -- this is the
theta-matched-to-KGW's-delta ablation: at which fixed theta does PECCAVI's tournament
resampling win or lose against a given KGW delta, rather than only comparing at
PECCAVI's single REINFORCE-learned theta_final against KGW's own tuned optimum.

Uses watermark_mode="peccavi" with generations=0, which skips the training loop
entirely -- run_peccavi() constructs Auctor(theta=theta_init) once and never mutates
`.theta` again when generations==0 (that mutation only happens inside the per-generation
training loop), so the eval phase generates and detects every sample at that one fixed
theta. See eval/watermark.py's run_peccavi() for the exact mechanism.

Loads the backbone once and reuses it across all sweep points; each point is wrapped in
try/except so one failure doesn't kill the whole sweep, and results are written after
every point (not just at the end) so a crash partway through still leaves usable partial
results.

Usage:
    python scripts/peccavi_theta_sweep.py
    python scripts/peccavi_theta_sweep.py --thetas 0.5,1.0,2.0,3.0,4.0,5.0 --max-tokens 250
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

THETAS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="configs/experiments.yaml")
    p.add_argument("--profile", default="kgw_matched_long",
                    help="Profile to source the model/backbone config from (only 'model' is used)")
    p.add_argument("--max-tokens", type=int, default=250)
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--n-attack", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--thetas", default=None,
                    help="Comma-separated theta values, overrides the default 0.5,1.0,2.0,3.0,4.0,5.0 grid")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    thetas = [float(x) for x in args.thetas.split(",")] if args.thetas else THETAS

    from main import init_backbone, load_profile
    from eval.watermark import run_peccavi

    logger.info(f"Initialising backbone from profile '{args.profile}'...")
    backbone = init_backbone(load_profile(args.config_file, args.profile))

    rows = []
    total = len(thetas)
    for i, theta in enumerate(thetas, 1):
        tag = f"theta{theta}"
        out_path = os.path.join(args.out_dir, f"peccavi_theta_sweep_{tag}.json")
        logger.info(f"[{i}/{total}] theta={theta} -> {out_path}")
        t0 = time.time()
        try:
            summary = run_peccavi(
                backbone,
                generations=0,
                n_eval_samples=args.n_eval,
                n_attack_samples=args.n_attack,
                max_tokens=args.max_tokens,
                verbose=False,
                watermark_mode="peccavi",
                theta_init=theta,
                seed=args.seed,
            )
            elapsed = time.time() - t0
            report = {k: v for k, v in summary.items() if k != "detailed_records"}
            report["theta"] = theta
            report["elapsed_sec"] = round(elapsed, 1)
            with open(out_path, "w") as f:
                json.dump({"peccavi_theta_sweep": report}, f, indent=2, default=str)

            aa = summary.get("attack_auc", {})
            asurv = summary.get("attack_survival", {})
            rows.append({
                "theta": theta,
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
            rows.append({"theta": theta, "error": str(e), "elapsed_sec": round(elapsed, 1)})

    summary_path = os.path.join(args.out_dir, "peccavi_theta_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PECCAVI fixed-theta SWEEP SUMMARY")
    print("=" * 70)
    header = f"{'theta':>6} {'AUC':>8} {'TPR@1%FPR':>10} {'synAttAUC':>10} {'synZ4':>7} {'sec':>6}"
    print(header)
    for r in rows:
        if "error" in r:
            print(f"{r['theta']:>6}  ERROR: {r['error'][:50]}")
        else:
            print(f"{r['theta']:>6} {r['auc_roc']:>8} {r['tpr_at_1fpr']:>10} "
                  f"{r['attack_auc_syntactic']:>10} {r['attack_survival_syntactic_z4']:>7} {r['elapsed_sec']:>6}")
    print(f"\nSaved -> {summary_path}")


if __name__ == "__main__":
    main()
