"""
main.py
Master entry point for PECCAVI watermarking via a shared LLaMA backbone.

Usage:
    python main.py --mode eval
    python main.py --mode train
    python main.py --mode kgw
    python main.py --mode infer --prompt "Your prompt here"
    python main.py --mode eval --seed 123 --output results/run_seed123.json
"""

from __future__ import annotations
import argparse
import logging
import sys
import yaml
import os
import json
import random
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("aiisc.log"),
    ],
)
logger = logging.getLogger("peccavi.main")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed set to {seed}")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def init_backbone(config_path: str = "configs/peccavi.yaml"):
    from backbone.model import LLaMABackbone
    cfg = load_config(config_path)
    model_cfg = cfg.get("model", {})
    logger.info(f"Initialising backbone: {model_cfg.get('backbone')}")
    return LLaMABackbone(
        model_name=model_cfg.get("backbone", "meta-llama/Llama-2-7b-chat-hf"),
        backend=model_cfg.get("backend", "transformers"),
        device=model_cfg.get("device", "auto"),
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        api_key=model_cfg.get("api_key"),
    )


THETA_CHECKPOINT = "./results/theta_checkpoint.json"


def _sanitize(d: dict) -> dict:
    """Recursively convert numpy scalars / NaN to JSON-safe Python primitives."""
    import math
    try:
        import numpy as _np
        _NB, _NI, _NF = _np.bool_, _np.integer, _np.floating
    except ImportError:
        _NB = _NI = _NF = type(None)

    def _fix(v):
        if isinstance(v, dict):
            return {k: _fix(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_fix(item) for item in v]
        if isinstance(v, _NB):
            return bool(v)
        if isinstance(v, _NI):
            return int(v)
        if isinstance(v, _NF):
            f = float(v)
            return None if math.isnan(f) else f
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    result = _fix(d)
    # alias expected by eval/compare.py METRICS
    if "improvement_pct" not in result and "effective_score_improvement_pct" in result:
        result["improvement_pct"] = result["effective_score_improvement_pct"]
    return result


def _load_theta() -> float:
    if os.path.exists(THETA_CHECKPOINT):
        with open(THETA_CHECKPOINT) as f:
            theta = json.load(f).get("theta", 2.0)
        logger.info(f"Loaded θ={theta:.4f} from checkpoint")
        return theta
    return 2.0


def mode_eval(backbone, args):
    from eval.benchmarks import run_benchmarks
    logger.info("Starting PECCAVI benchmark evaluation...")
    output_path = getattr(args, "output", "./results/benchmark_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    use_baselines = getattr(args, "baselines", False)
    if use_baselines:
        cfg = load_config(os.path.join(args.config_dir, "peccavi.yaml"))
        results = run_benchmarks(baseline_config=cfg, output_path=output_path, verbose=True)
    else:
        results = run_benchmarks(backbone, output_path=output_path, verbose=True)

    logger.info("Evaluation complete.")
    return results


def mode_kgw(backbone, args):
    """Run KGW baseline evaluation."""
    from eval.watermark import run_peccavi
    cfg_file = getattr(args, "config_file", None)
    cfg = load_config(cfg_file if cfg_file else os.path.join(args.config_dir, "kgw_baseline.yaml"))
    wm_cfg = cfg.get("watermarking", {})
    pl_cfg = cfg.get("policy_learning", {})
    seed = getattr(args, "seed", 42)

    logger.info("Running KGW baseline evaluation...")
    summary = run_peccavi(
        backbone,
        generations=pl_cfg.get("generations", 5),
        n_paraphrases=cfg.get("agents", {}).get("scriba_n_variants", 10),
        n_eval_samples=pl_cfg.get("n_eval_samples", 100),
        verbose=True,
        theta_init=_load_theta(),
        watermark_mode="kgw",
        kgw_delta=wm_cfg.get("delta", 2.0),
        kgw_gamma=wm_cfg.get("gamma", 0.5),
        seed=seed,
    )

    output_path = getattr(args, "output", "./results/kgw_baseline.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        report = {"kgw_baseline": _sanitize({k: v for k, v in summary.items() if k != "detailed_records"})}
        json.dump(report, f, indent=2)

    logger.info(
        f"KGW done. AUC-ROC={summary['auc_roc']:.4f} | "
        f"FPR={summary['false_positive_rate']:.4f} | "
        f"S_eff={summary['effective_score_final']:.4f}"
    )


def mode_sir(backbone, args):
    """Run SIR (entropy-aware) baseline evaluation."""
    from eval.watermark import run_peccavi
    cfg = load_config(os.path.join(args.config_dir, "sir_baseline.yaml"))
    wm_cfg = cfg.get("watermarking", {})
    pl_cfg = cfg.get("policy_learning", {})
    seed = getattr(args, "seed", 42)

    logger.info("Running SIR (entropy-aware) baseline evaluation...")
    summary = run_peccavi(
        backbone,
        generations=pl_cfg.get("generations", 5),
        n_paraphrases=cfg.get("agents", {}).get("scriba_n_variants", 10),
        n_eval_samples=pl_cfg.get("n_eval_samples", 100),
        verbose=True,
        theta_init=_load_theta(),
        watermark_mode="sir",
        sir_delta=wm_cfg.get("delta", 2.0),
        sir_gamma=wm_cfg.get("gamma", 0.5),
        sir_entropy_threshold=wm_cfg.get("entropy_threshold", 1.0),
        seed=seed,
    )

    output_path = getattr(args, "output", "./results/sir_baseline.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        report = {"sir_baseline": _sanitize({k: v for k, v in summary.items() if k != "detailed_records"})}
        json.dump(report, f, indent=2)

    logger.info(
        f"SIR done. AUC-ROC={summary['auc_roc']:.4f} | "
        f"FPR={summary['false_positive_rate']:.4f} | "
        f"S_eff={summary['effective_score_final']:.4f} | "
        f"PPL_ratio={summary.get('ppl_ratio', 'N/A')}"
    )


def mode_dipmark(backbone, args):
    """Run DiPMark (Zhao et al., 2024) baseline evaluation."""
    from eval.watermark import run_peccavi
    cfg_file = getattr(args, "config_file", None)
    cfg = load_config(cfg_file if cfg_file else os.path.join(args.config_dir, "dipmark_baseline.yaml"))
    wm_cfg = cfg.get("watermarking", {})
    pl_cfg = cfg.get("policy_learning", {})
    seed = getattr(args, "seed", 42)

    logger.info("Running DiPMark baseline evaluation...")
    summary = run_peccavi(
        backbone,
        generations=pl_cfg.get("generations", 5),
        n_paraphrases=cfg.get("agents", {}).get("scriba_n_variants", 10),
        n_eval_samples=pl_cfg.get("n_eval_samples", 100),
        verbose=True,
        theta_init=wm_cfg.get("theta_init", 2.0),
        watermark_mode="dipmark",
        dipmark_delta=wm_cfg.get("delta", 2.0),
        dipmark_gamma=wm_cfg.get("gamma", 0.5),
        dipmark_window=wm_cfg.get("window", 5),
        seed=seed,
    )

    output_path = getattr(args, "output", "./results/dipmark_baseline.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        report = {"dipmark_baseline": _sanitize({k: v for k, v in summary.items() if k != "detailed_records"})}
        json.dump(report, f, indent=2)

    logger.info(
        f"DiPMark done. AUC-ROC={summary['auc_roc']:.4f} | "
        f"FPR={summary['false_positive_rate']:.4f} | "
        f"S_eff={summary['effective_score_final']:.4f} | "
        f"PPL_ratio={summary.get('ppl_ratio', 'N/A')}"
    )


def mode_synthid(backbone, args):
    """Run SynthID-Text (Dathathri et al., 2024) baseline evaluation."""
    from eval.watermark import run_peccavi
    cfg_file = getattr(args, "config_file", None)
    cfg = load_config(cfg_file if cfg_file else os.path.join(args.config_dir, "synthid_baseline.yaml"))
    wm_cfg = cfg.get("watermarking", {})
    pl_cfg = cfg.get("policy_learning", {})
    seed = getattr(args, "seed", 42)

    logger.info("Running SynthID-Text baseline evaluation...")
    summary = run_peccavi(
        backbone,
        generations=pl_cfg.get("generations", 5),
        n_paraphrases=cfg.get("agents", {}).get("scriba_n_variants", 10),
        n_eval_samples=pl_cfg.get("n_eval_samples", 100),
        verbose=True,
        theta_init=wm_cfg.get("theta_init", 2.0),
        watermark_mode="synthid",
        synthid_tournament_k=wm_cfg.get("tournament_k", 8),
        seed=seed,
    )

    output_path = getattr(args, "output", "./results/synthid_baseline.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        report = {"synthid_baseline": _sanitize({k: v for k, v in summary.items() if k != "detailed_records"})}
        json.dump(report, f, indent=2)

    logger.info(
        f"SynthID done. AUC-ROC={summary['auc_roc']:.4f} | "
        f"FPR={summary['false_positive_rate']:.4f} | "
        f"S_eff={summary['effective_score_final']:.4f} | "
        f"PPL_ratio={summary.get('ppl_ratio', 'N/A')}"
    )


def mode_train(backbone, args):
    _train_peccavi(backbone, args)


def _train_peccavi(backbone, args=None):
    from eval.watermark import run_peccavi
    cfg_dir = getattr(args, "config_dir", "configs") if args else "configs"
    cfg_file = getattr(args, "config_file", None) if args else None
    cfg = load_config(cfg_file if cfg_file else os.path.join(cfg_dir, "peccavi.yaml"))
    pl_cfg = cfg.get("policy_learning", {})
    wm_cfg = cfg.get("watermarking", {})
    seed = getattr(args, "seed", 42) if args else 42

    logger.info("Training PECCAVI watermarking policy...")
    summary = run_peccavi(
        backbone,
        generations=pl_cfg.get("generations", 10),
        n_paraphrases=cfg.get("agents", {}).get("scriba_n_variants", 5),
        n_eval_samples=pl_cfg.get("n_eval_samples", 100),
        verbose=True,
        theta_init=wm_cfg.get("theta_init", 2.0),
        watermark_mode=wm_cfg.get("watermark_mode", "peccavi"),
        lam=pl_cfg.get("lambda_wm", 0.6),
        nu=pl_cfg.get("nu_quality", 0.4),
        mu_ppl=pl_cfg.get("mu_ppl", 0.0),
        rho_survival=pl_cfg.get("rho_survival", 0.0),
        alpha=pl_cfg.get("alpha", 0.05),
        seed=seed,
        checkpoint_path=THETA_CHECKPOINT,
        adaptive_theta=wm_cfg.get("adaptive_theta", False),
        theta_min=wm_cfg.get("theta_min", 0.5),
        theta_max=wm_cfg.get("theta_max", 8.0),
    )

    theta_final = summary["theta_final"]
    w_final = summary.get("w_final")
    os.makedirs(os.path.dirname(THETA_CHECKPOINT), exist_ok=True)
    ckpt = {"theta": theta_final}
    if w_final is not None:
        ckpt["w"] = w_final
    with open(THETA_CHECKPOINT, "w") as f:
        json.dump(ckpt, f)

    # Save results JSON (needed by compare.py and run_ablations.py)
    output_path = getattr(args, "output", "./results/peccavi_train.json") if args else "./results/peccavi_train.json"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        report_data = _sanitize({k: v for k, v in summary.items() if k != "detailed_records"})
        json.dump({"peccavi": report_data}, f, indent=2)

    w_log = f" | w={[round(x,3) for x in w_final]}" if w_final else ""
    logger.info(
        f"PECCAVI training done. "
        f"θ_base={theta_final}{w_log} | "
        f"S_eff={summary['effective_score_final']:.4f} | "
        f"Improvement={summary['effective_score_improvement_pct']:.1f}% | "
        f"θ saved → {THETA_CHECKPOINT} | results → {output_path}"
    )


def mode_infer(backbone, args):
    prompt = args.prompt
    print("\n" + "═" * 60)
    print(f"  PROMPT: {prompt}")
    print("═" * 60)

    std_out = backbone.generate(prompt, max_new_tokens=200)
    print("\n  [Backbone Response]")
    print(f"  {std_out['text']}\n")

    from peccavi.auctor import Auctor
    from peccavi.custos import Custos
    auctor = Auctor(backbone, theta=_load_theta())
    custos = Custos(backbone)
    wm_text = auctor.generate(prompt, max_tokens=200)
    detection = custos.detect(wm_text)
    print("  [PECCAVI Watermarked Response]")
    print(f"  {wm_text}")
    print(f"\n  Z-score         : {detection['z_score']} "
          f"({'detected' if detection['is_watermarked'] else 'not detected'})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PECCAVI - Watermarking and Content Authenticity",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--mode", required=True,
        choices=["eval", "train", "kgw", "sir", "dipmark", "synthid", "infer"],
        help=(
            "eval    - run full PECCAVI benchmarks\n"
            "train   - run policy learning over simulated generations\n"
            "kgw     - run KGW baseline evaluation\n"
            "sir     - run SIR (entropy-aware) baseline evaluation\n"
            "dipmark - run DiPMark (Zhao et al., 2024) baseline evaluation\n"
            "synthid - run SynthID-Text (Dathathri et al., 2024) baseline evaluation\n"
            "infer   - single-prompt watermarking demo"
        ),
    )
    p.add_argument("--prompt", type=str,
                   default="Explain the importance of AI safety in modern systems.")
    p.add_argument("--output", type=str, default="./results/benchmark_results.json")
    p.add_argument("--baselines", action="store_true",
                   help="Run against all baseline models defined in config")
    p.add_argument("--config-dir", type=str, default="configs")
    p.add_argument("--config-file", type=str, default=None,
                   help="Explicit path to a config YAML (overrides --config-dir default)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    print(" PECCAVI-TEXT")
    logger.info(f"Mode: {args.mode.upper()}")

    set_seed(args.seed)

    if args.config_file:
        backbone = init_backbone(config_path=args.config_file)
    else:
        config_map = {
            "kgw": "kgw_baseline.yaml",
            "sir": "sir_baseline.yaml",
            "dipmark": "dipmark_baseline.yaml",
            "synthid": "synthid_baseline.yaml",
        }
        config_filename = config_map.get(args.mode, "peccavi.yaml")
        backbone = init_backbone(
            config_path=os.path.join(args.config_dir, config_filename)
        )

    if args.mode == "eval":
        mode_eval(backbone, args)
    elif args.mode == "train":
        mode_train(backbone, args)
    elif args.mode == "kgw":
        mode_kgw(backbone, args)
    elif args.mode == "sir":
        mode_sir(backbone, args)
    elif args.mode == "dipmark":
        mode_dipmark(backbone, args)
    elif args.mode == "synthid":
        mode_synthid(backbone, args)
    elif args.mode == "infer":
        mode_infer(backbone, args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
