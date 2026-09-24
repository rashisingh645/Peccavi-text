"""
eval/benchmarks.py
Runs the PECCAVI benchmark suite and produces a unified report.
Prints a summary table and saves results to JSON.
"""

from __future__ import annotations
from backbone.model import LLaMABackbone
from eval.watermark import run_peccavi
from typing import Dict
import json
import math
import os
import logging
import yaml

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and NaN."""
    def default(self, o):
        try:
            import numpy as np
            if isinstance(o, np.bool_):
                return bool(o)
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                f = float(o)
                return None if math.isnan(f) else f
        except ImportError:
            pass
        if isinstance(o, float) and math.isnan(o):
            return None
        return super().default(o)

THETA_CHECKPOINT = "./results/theta_checkpoint.json"
DETAILED_OUTPUT = "./results/detailed_results.json"
CONFIG_PATH = "configs/experiments.yaml"
CONFIG_PROFILE = "peccavi"


def _load_theta(checkpoint_path: str = THETA_CHECKPOINT) -> float:
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            theta = json.load(f).get("theta", 2.0)
        logger.info(f"Loaded θ={theta:.4f} from checkpoint {checkpoint_path}")
        return theta
    logger.info("No θ checkpoint found — starting from theta_init=2.0")
    return 2.0


def _load_config() -> dict:
    """Returns the "peccavi" profile's content from the consolidated config file
    (policy_learning/agents), matching what used to be the top level of peccavi.yaml."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("profiles", {}).get(CONFIG_PROFILE, {})
    return {}


def _peccavi_summary(pec_out: Dict) -> Dict:
    robustness = pec_out.get("avg_retention_rate", pec_out["effective_score_final"]) * 100
    resilience = pec_out["effective_score_final"] * 100
    return {
        "watermark_mode": pec_out.get("watermark_mode", "peccavi"),
        "seed": pec_out.get("seed", 42),
        "theta_final": pec_out["theta_final"],
        "effective_score_final": pec_out["effective_score_final"],
        "improvement_pct": pec_out["effective_score_improvement_pct"],
        "auc_roc": pec_out["auc_roc"],
        "tpr_at_1fpr": pec_out.get("tpr_at_1fpr"),
        "false_positive_rate": pec_out["false_positive_rate"],
        "avg_ppl_baseline": pec_out.get("avg_ppl_baseline"),
        "avg_ppl_watermarked": pec_out.get("avg_ppl_watermarked"),
        "ppl_ratio": pec_out.get("ppl_ratio"),
        "attack_survival": pec_out.get("attack_survival", {}),
        "avg_readability": pec_out["avg_readability"],
        "pass_retention": pec_out["meets_85pct_retention"],
        "pass_auc": pec_out["meets_90pct_auc"],
        "pass_readability": pec_out.get("meets_readability_30", pec_out["meets_readability_45"]),
        "robustness": round(robustness, 2),
        "resilience": round(resilience, 2),
        "fpr": round(pec_out["false_positive_rate"] * 100, 2),
        "readability": pec_out["avg_readability"],
    }


def run_benchmarks(
    backbone: LLaMABackbone = None,
    output_path: str = "./results/benchmark_results.json",
    verbose: bool = True,
    baseline_config: Dict = None,
) -> Dict:
    """
    Run PECCAVI benchmarks. If baseline_config provided, run multiple baseline models.
    n_eval_samples and n_paraphrases are read from configs/peccavi.yaml.
    """
    cfg = _load_config()
    pl_cfg = cfg.get("policy_learning", {})
    n_eval_samples = pl_cfg.get("n_eval_samples", 100)
    n_paraphrases = cfg.get("agents", {}).get("scriba_n_variants", 10)

    report: Dict = {}
    all_detailed: Dict = {}

    print("\n" + "═" * 60)
    print("  BENCHMARK: PECCAVI - Watermarking & Content Authenticity")
    print(f"  Eval samples: {n_eval_samples} | Paraphrases: {n_paraphrases}")
    print("═" * 60)

    if baseline_config:
        baselines = baseline_config.get("baseline_models", [])
        for baseline in baselines:
            model_name = baseline.get("name")
            model_id = baseline.get("backbone")
            backend = baseline.get("backend", "transformers")
            api_key = baseline.get("api_key")

            print(f"\n  Running baseline: {model_name}...")
            try:
                model_backbone = LLaMABackbone(
                    model_name=model_id,
                    backend=backend,
                    api_key=api_key,
                )
                theta_init = _load_theta()
                pec_out = run_peccavi(
                    model_backbone,
                    generations=5,
                    n_paraphrases=n_paraphrases,
                    n_eval_samples=n_eval_samples,
                    verbose=verbose,
                    theta_init=theta_init,
                )
                report[model_name] = _peccavi_summary(pec_out)
                all_detailed[model_name] = pec_out.get("detailed_records", [])
            except Exception as e:
                logger.error(f"Error running baseline {model_name}: {e}")
                report[model_name] = {"error": str(e)}
    else:
        if backbone is None:
            raise ValueError("Either backbone or baseline_config must be provided")
        theta_init = _load_theta()
        wm_cfg = cfg.get("watermarking", {})
        pl_cfg2 = cfg.get("policy_learning", {})
        pec_out = run_peccavi(
            backbone,
            generations=5,
            n_paraphrases=n_paraphrases,
            n_eval_samples=n_eval_samples,
            verbose=verbose,
            theta_init=theta_init,
            watermark_mode=wm_cfg.get("watermark_mode", "peccavi"),
            lam=pl_cfg2.get("lambda_wm", 0.6),
            nu=pl_cfg2.get("nu_quality", 0.4),
            alpha=pl_cfg2.get("alpha", 0.05),
        )
        report["peccavi"] = _peccavi_summary(pec_out)
        all_detailed["peccavi"] = pec_out.get("detailed_records", [])

    _print_summary(report)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, cls=_NumpyEncoder)
    print(f"\n  Summary saved → {output_path}")

    with open(DETAILED_OUTPUT, "w") as f:
        json.dump(all_detailed, f, indent=2, cls=_NumpyEncoder)
    print(f"  Detailed records saved → {DETAILED_OUTPUT}")

    return report


def _print_summary(report: Dict):
    print("\n" + "═" * 60)
    print("  PECCAVI BENCHMARK SUMMARY")
    print("═" * 60)

    for model_name, results in report.items():
        if isinstance(results, dict) and "error" in results:
            print(f"\n  {model_name}: ERROR - {results['error']}")
            continue

        print(f"\n  Model: {model_name}")
        print(f"  θ_final           : {results['theta_final']}")
        print(f"  Effective Score   : {results['effective_score_final']:.4f}  "
              f"({'PASS' if results['pass_retention'] else 'FAIL'} ≥0.85)")
        print(f"  Improvement       : {results['improvement_pct']:.1f}%")
        print(f"  AUC-ROC           : {results['auc_roc']:.4f}  "
              f"({'PASS' if results['pass_auc'] else 'FAIL'} ≥0.90)")
        print(f"  False Positive    : {results['false_positive_rate']:.4f}")
        print(f"  Avg Readability   : {results['avg_readability']:.2f}/5  "
              f"({'PASS' if results['pass_readability'] else 'FAIL'} ≥3.0)")
        if results.get("tpr_at_1fpr") is not None:
            print(f"  TPR @ 1% FPR      : {results['tpr_at_1fpr']:.4f}")
        if results.get("ppl_ratio") is not None:
            try:
                print(f"  PPL Ratio (wm/bl) : {results['ppl_ratio']:.4f}  (1.0 = no quality cost)")
            except (ValueError, TypeError):
                pass
        if results.get("attack_survival"):
            surv = results["attack_survival"]
            parts = "  |  ".join(f"{k}: {v:.2f}" for k, v in surv.items())
            print(f"  Attack survival   : {parts}")
