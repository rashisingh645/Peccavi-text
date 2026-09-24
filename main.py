"""
main.py
Master entry point for PECCAVI watermarking via a shared LLaMA backbone.

Config is a single consolidated file with named profiles inside
(configs/experiments.yaml for LLaMA, configs/experiments_mistral.yaml for Mistral),
selected via --profile <name>. Each CLI --mode has a default profile
(MODE_DEFAULT_PROFILE) used when --profile is omitted.

Usage:
    python main.py --mode eval
    python main.py --mode train --profile peccavi_attack_aware
    python main.py --mode kgw --profile kgw_strong
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

try:
    # Windows consoles often default to a non-UTF-8 codepage (e.g. cp1252), which can't
    # encode the θ/μ/ν/→ characters used throughout this codebase's log messages —
    # without this, logging.StreamHandler silently drops those lines (UnicodeEncodeError
    # inside logging's own error handling, which swallows it and moves on).
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("aiisc.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("peccavi.main")

DEFAULT_CONFIG_FILE = "configs/experiments.yaml"
MODE_DEFAULT_PROFILE = {
    "eval": "peccavi",
    "train": "peccavi",
    "kgw": "kgw_baseline",
    "sir": "sir_baseline",
    "dipmark": "dipmark_baseline",
    "synthid": "synthid_baseline",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed set to {seed}")


def load_config_file(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_profile(path: str, profile: str) -> dict:
    """Loads one named profile's config block out of a consolidated experiments file."""
    data = load_config_file(path)
    profiles = data.get("profiles", data)  # tolerate a bare (non-profile) file too
    if profile not in profiles:
        raise KeyError(
            f"Profile '{profile}' not found in {path}. "
            f"Available profiles: {sorted(profiles.keys())}"
        )
    return profiles[profile]


def init_backbone(cfg: dict):
    from backbone.model import LLaMABackbone
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


def mode_eval(backbone, cfg, args):
    from eval.benchmarks import run_benchmarks
    logger.info("Starting PECCAVI benchmark evaluation...")
    output_path = getattr(args, "output", "./results/benchmark_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    use_baselines = getattr(args, "baselines", False)
    if use_baselines:
        # baseline_models lives at the top level of the consolidated file, not inside
        # any single profile — load the whole file, not load_profile()'s sub-block.
        full_cfg = load_config_file(getattr(args, "config_file", None) or DEFAULT_CONFIG_FILE)
        results = run_benchmarks(baseline_config=full_cfg, output_path=output_path, verbose=True)
    else:
        results = run_benchmarks(backbone, output_path=output_path, verbose=True)

    logger.info("Evaluation complete.")
    return results


def mode_kgw(backbone, cfg, args):
    """Run KGW baseline evaluation."""
    from eval.watermark import run_peccavi
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
        # Fixed default, not _load_theta(): KGW has no policy (magister=None), so this
        # value only ever populates the reported "theta_final" field, never actual
        # generation behaviour (that's kgw_delta below). Reading the shared PECCAVI
        # training checkpoint here would leak an unrelated learned theta into KGW's
        # report and make it look like KGW "learned" something it didn't.
        theta_init=wm_cfg.get("theta_init", 2.0),
        watermark_mode="kgw",
        kgw_delta=wm_cfg.get("delta", 2.0),
        kgw_gamma=wm_cfg.get("gamma", 0.5),
        n_attack_samples=pl_cfg.get("n_attack_samples", 100),
        max_tokens=pl_cfg.get("max_tokens", 100),
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


def mode_sir(backbone, cfg, args):
    """Run SIR (Liu, Pan, Hu, Meng, Wen, ICLR 2024) baseline evaluation."""
    from eval.watermark import run_peccavi
    wm_cfg = cfg.get("watermarking", {})
    pl_cfg = cfg.get("policy_learning", {})
    seed = getattr(args, "seed", 42)

    logger.info("Running SIR baseline evaluation...")
    summary = run_peccavi(
        backbone,
        generations=pl_cfg.get("generations", 5),
        n_paraphrases=cfg.get("agents", {}).get("scriba_n_variants", 10),
        n_eval_samples=pl_cfg.get("n_eval_samples", 100),
        verbose=True,
        # Fixed default, not _load_theta() — same reasoning as mode_kgw above; SIR also
        # has magister=None, so this only feeds the (otherwise-meaningless) theta_final
        # report field, and must not leak PECCAVI's trained checkpoint into it.
        theta_init=wm_cfg.get("theta_init", 2.0),
        watermark_mode="sir",
        sir_delta=wm_cfg.get("delta", 2.0),
        sir_embedding_model=wm_cfg.get(
            "embedding_model", "perceptiveshawty/compositional-bert-large-uncased"
        ),
        sir_checkpoint_path=wm_cfg.get("checkpoint_path", "results/sir_transform_model.pt"),
        sir_proj_dim=wm_cfg.get("proj_dim", 1000),
        n_attack_samples=pl_cfg.get("n_attack_samples", 100),
        max_tokens=pl_cfg.get("max_tokens", 100),
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


def mode_dipmark(backbone, cfg, args):
    """Run DiPmark (Wu, Hu, Guo, Zhang, Huang, ICML 2024) baseline evaluation."""
    from eval.watermark import run_peccavi
    wm_cfg = cfg.get("watermarking", {})
    pl_cfg = cfg.get("policy_learning", {})
    seed = getattr(args, "seed", 42)

    logger.info("Running DiPmark baseline evaluation...")
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
        n_attack_samples=pl_cfg.get("n_attack_samples", 100),
        max_tokens=pl_cfg.get("max_tokens", 100),
        seed=seed,
    )

    output_path = getattr(args, "output", "./results/dipmark_baseline.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        report = {"dipmark_baseline": _sanitize({k: v for k, v in summary.items() if k != "detailed_records"})}
        json.dump(report, f, indent=2)

    logger.info(
        f"DiPmark done. AUC-ROC={summary['auc_roc']:.4f} | "
        f"FPR={summary['false_positive_rate']:.4f} | "
        f"S_eff={summary['effective_score_final']:.4f} | "
        f"PPL_ratio={summary.get('ppl_ratio', 'N/A')}"
    )


def mode_synthid(backbone, cfg, args):
    """Run SynthID-Text (Dathathri et al., Nature 2024) baseline evaluation."""
    from eval.watermark import run_peccavi
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
        synthid_tournament_k=wm_cfg.get("tournament_k", 16),
        synthid_score_function=wm_cfg.get("score_function", "bayesian"),
        synthid_g_distribution=wm_cfg.get("g_distribution", "bernoulli"),
        n_attack_samples=pl_cfg.get("n_attack_samples", 100),
        max_tokens=pl_cfg.get("max_tokens", 100),
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


def mode_train(backbone, cfg, args):
    from eval.watermark import run_peccavi
    wm_cfg = cfg.get("watermarking", {})
    pl_cfg = cfg.get("policy_learning", {})
    seed = getattr(args, "seed", 42)
    wm_mode = wm_cfg.get("watermark_mode", "peccavi")
    # peccavi_df learns alpha, not theta — checkpointing it to the same file as the
    # theta-based "peccavi" mode would silently overwrite one with the other's value.
    checkpoint_path = "./results/alpha_checkpoint.json" if wm_mode == "peccavi_df" else THETA_CHECKPOINT

    logger.info("Training PECCAVI watermarking policy...")
    summary = run_peccavi(
        backbone,
        generations=pl_cfg.get("generations", 10),
        n_paraphrases=cfg.get("agents", {}).get("scriba_n_variants", 5),
        n_eval_samples=pl_cfg.get("n_eval_samples", 100),
        verbose=True,
        theta_init=wm_cfg.get("theta_init", 2.0),
        tournament_k=wm_cfg.get("tournament_k", 16),
        watermark_mode=wm_mode,
        df_alpha_init=wm_cfg.get("alpha_init", 0.3),
        df_alpha_min=wm_cfg.get("alpha_min", 0.05),
        df_alpha_max=wm_cfg.get("alpha_max", 0.49),
        df_window=wm_cfg.get("window", 5),
        lam=pl_cfg.get("lambda_wm", 0.6),
        nu=pl_cfg.get("nu_quality", 0.4),
        mu_ppl=pl_cfg.get("mu_ppl", 0.0),
        rho_survival=pl_cfg.get("rho_survival", 0.0),
        alpha=pl_cfg.get("alpha", 0.05),
        seed=seed,
        checkpoint_path=checkpoint_path,
        adaptive_theta=wm_cfg.get("adaptive_theta", False),
        theta_min=wm_cfg.get("theta_min", 0.5),
        theta_max=wm_cfg.get("theta_max", 5.0),
        n_attack_samples=pl_cfg.get("n_attack_samples", 100),
        attack_mix=pl_cfg.get("attack_mix", False),
        max_tokens=pl_cfg.get("max_tokens", 100),
    )
    # run_peccavi() already writes checkpoint_path itself (before eval, so a crash during
    # eval doesn't lose the learned theta/alpha) — no need to duplicate that write here.

    # Save results JSON (needed by compare.py and run_ablations.py)
    output_path = getattr(args, "output", None) or f"./results/{wm_mode}_train.json"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        report_data = _sanitize({k: v for k, v in summary.items() if k != "detailed_records"})
        json.dump({wm_mode: report_data}, f, indent=2)

    theta_final = summary["theta_final"]
    w_final = summary.get("w_final")
    w_log = f" | w={[round(x, 3) for x in w_final]}" if w_final else ""
    logger.info(
        f"PECCAVI training done. "
        f"θ_base={theta_final}{w_log} | "
        f"S_eff={summary['effective_score_final']:.4f} | "
        f"Improvement={summary['effective_score_improvement_pct']:.1f}% | "
        f"checkpoint saved → {checkpoint_path} | results → {output_path}"
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
            "sir     - run SIR (Liu et al., ICLR 2024) baseline evaluation\n"
            "dipmark - run DiPmark (Wu et al., ICML 2024) baseline evaluation\n"
            "synthid - run SynthID-Text (Dathathri et al., Nature 2024) baseline evaluation\n"
            "infer   - single-prompt watermarking demo"
        ),
    )
    p.add_argument("--profile", type=str, default=None,
                   help="Named profile inside --config-file's `profiles:` block "
                        "(default depends on --mode, e.g. 'peccavi' for train/eval, "
                        "'kgw_baseline' for kgw)")
    p.add_argument("--prompt", type=str,
                   default="Explain the importance of AI safety in modern systems.")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--baselines", action="store_true",
                   help="Run against all baseline models defined in config")
    p.add_argument("--config-file", type=str, default=None,
                   help=f"Path to the consolidated experiments YAML (default: {DEFAULT_CONFIG_FILE})")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    print(" PECCAVI-TEXT")
    logger.info(f"Mode: {args.mode.upper()}")

    set_seed(args.seed)

    config_file = args.config_file or DEFAULT_CONFIG_FILE
    # "infer" isn't in MODE_DEFAULT_PROFILE, so it falls back to "peccavi" here too —
    # it just needs a plain backbone, same as every other mode.
    profile_name = args.profile or MODE_DEFAULT_PROFILE.get(args.mode, "peccavi")
    cfg = load_profile(config_file, profile_name)
    backbone = init_backbone(cfg)

    if args.mode == "eval":
        mode_eval(backbone, cfg, args)
    elif args.mode == "train":
        mode_train(backbone, cfg, args)
    elif args.mode == "kgw":
        mode_kgw(backbone, cfg, args)
    elif args.mode == "sir":
        mode_sir(backbone, cfg, args)
    elif args.mode == "dipmark":
        mode_dipmark(backbone, cfg, args)
    elif args.mode == "synthid":
        mode_synthid(backbone, cfg, args)
    elif args.mode == "infer":
        mode_infer(backbone, args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
