"""
eval/watermark.py
PECCAVI full evaluation loop.
Supports watermark_mode: "peccavi" | "kgw" | "none"
"""

from __future__ import annotations
import json
import os
import random
import numpy as np
import torch
from backbone.model import LLaMABackbone
from peccavi.praeco import Praeco
from peccavi.auctor import Auctor
from peccavi.auctor_kgw import KGWAuctor
from peccavi.auctor_sir import SIRAuctor
from peccavi.auctor_dipmark import DiPMarkAuctor
from peccavi.auctor_synthid import SynthIDAuctor
from peccavi.scriba import Scriba
from peccavi.custos import Custos
from peccavi.magister import Magister
from peccavi.featurizer import PromptFeaturizer
from eval.quality import quality_score, flesch_quality_score, perplexity
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_peccavi(
    backbone: LLaMABackbone,
    generations: int = 10,
    n_paraphrases: int = 5,
    n_eval_samples: int = 100,
    verbose: bool = True,
    theta_init: float = 2.0,
    watermark_mode: str = "peccavi",   # "peccavi" | "kgw" | "sir" | "none"
    kgw_delta: float = 2.0,
    kgw_gamma: float = 0.5,
    sir_delta: float = 2.0,
    sir_gamma: float = 0.5,
    sir_entropy_threshold: float = 1.0,
    dipmark_delta: float = 2.0,
    dipmark_gamma: float = 0.5,
    dipmark_window: int = 5,
    synthid_tournament_k: int = 8,
    lam: float = 0.6,
    nu: float = 0.4,
    mu_ppl: float = 0.0,
    rho_survival: float = 0.0,
    alpha: float = 0.05,
    seed: int = 42,
    checkpoint_path: str = None,
    adaptive_theta: bool = False,
    theta_min: float = 0.5,
    theta_max: float = 8.0,
) -> Dict:
    _set_seed(seed)

    praeco = Praeco()
    scriba = Scriba(backbone, n_variants=n_paraphrases)
    custos = Custos(backbone)

    if watermark_mode == "kgw":
        generator = KGWAuctor(backbone, delta=kgw_delta, gamma=kgw_gamma)
        magister = None
    elif watermark_mode == "sir":
        generator = SIRAuctor(
            backbone, delta=sir_delta, gamma=sir_gamma,
            entropy_threshold=sir_entropy_threshold,
        )
        magister = None
    elif watermark_mode == "dipmark":
        generator = DiPMarkAuctor(backbone, delta=dipmark_delta, gamma=dipmark_gamma,
                                  window=dipmark_window)
        magister = None
    elif watermark_mode == "synthid":
        generator = SynthIDAuctor(backbone, theta=theta_init,
                                  tournament_k=synthid_tournament_k)
        magister = None
    elif watermark_mode == "none":
        generator = None
        magister = None
    else:
        generator = Auctor(backbone, theta=theta_init)
        magister = Magister(
            backbone, theta_init=theta_init, alpha=alpha, lam=lam, nu=nu,
            mu_ppl=mu_ppl, rho_survival=rho_survival,
            adaptive=adaptive_theta, theta_min=theta_min, theta_max=theta_max,
        )

    featurizer = PromptFeaturizer() if (adaptive_theta and watermark_mode == "peccavi") else None

    history = []
    detailed_records: List[Dict] = []
    theta_by_prompt: List[Dict] = []   # tracks (entropy, theta_context) for paper Figure 2

    for gen in range(1, generations + 1):
        try:
            prompt = praeco.next_prompt()

            if watermark_mode == "peccavi":
                features = featurizer.extract(prompt) if featurizer else None
                context_theta = magister.compute_theta(features)
                generator.theta = context_theta
                wm_text = generator.generate(prompt, max_tokens=100)
            elif watermark_mode in ("kgw", "sir", "dipmark", "synthid"):
                wm_text = generator.generate(prompt, max_tokens=100)
            else:
                wm_text = backbone.generate(prompt, max_new_tokens=100)["text"]

            if watermark_mode in ("kgw", "sir", "dipmark"):
                original_score = (generator.z_score(wm_text) + 10) / 20  # normalise z to [0,1] approx
            else:
                original_score = custos.watermark_score(wm_text)

            paraphrases = scriba.paraphrase(wm_text)
            s_eff = custos.effective_score(paraphrases)
            z_eff = custos.effective_z_score(paraphrases)

            z_threshold = 4.0
            if watermark_mode in ("kgw", "sir", "dipmark"):
                para_z = [generator.z_score(p) for p in paraphrases]
            else:
                para_z = [custos.z_score(p) for p in paraphrases]
            retention = sum(1 for z in para_z if z >= z_threshold) / max(len(para_z), 1)

            q_score = quality_score(wm_text, prompt=prompt)
            readability = flesch_quality_score(wm_text)

            _feats = features if watermark_mode == "peccavi" else None
            new_theta = (
                magister.update(wm_text, original_score, reference_text=prompt, prompt_features=_feats)
                if magister else theta_init
            )
        except Exception as _gen_exc:
            logger.warning(f"Gen {gen} failed and will be skipped: {_gen_exc}")
            continue

        # Track (entropy, θ_context) for Figure 2: θ vs prompt entropy scatter plot
        if featurizer is not None and _feats is not None:
            theta_by_prompt.append({
                "entropy": round(float(_feats[0]), 4),
                "theta_context": round(context_theta, 4),
            })

        record = {
            "generation": gen,
            "theta": round(new_theta, 4),
            "theta_context": round(context_theta if watermark_mode == "peccavi" else new_theta, 4),
            "original_score": round(original_score, 4),
            "effective_score": round(s_eff, 4),
            "effective_z_score": round(z_eff, 4),
            "retention_rate": round(retention, 4),
            "readability": readability,
            "gpt4_quality": q_score,
        }
        history.append(record)

        detailed_records.append({
            "phase": "training",
            "generation": gen,
            "dataset": praeco.get_source(prompt),
            "prompt": prompt,
            "watermarked_text": wm_text,
            "paraphrases": paraphrases,
            "scores": {
                "s_orig": round(original_score, 4),
                "s_eff": round(s_eff, 4),
                "z_eff": round(z_eff, 4),
                "retention_rate": round(retention, 4),
                "theta": round(new_theta, 4),
                "theta_context": round(context_theta if watermark_mode == "peccavi" else new_theta, 4),
                "readability": readability,
                "gpt4_quality": q_score,
            },
        })

        if verbose:
            logger.info(
                f"Gen {gen:>3} | θ={new_theta:.4f} | "
                f"S_orig={original_score:.4f} | S_eff={s_eff:.4f}"
            )

    # Save theta checkpoint immediately after training — before any eval that could crash
    if checkpoint_path and watermark_mode == "peccavi" and magister is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        ckpt = {"theta": round(magister.theta, 6)}
        if adaptive_theta:
            ckpt["w"] = magister.w.tolist()
        with open(checkpoint_path, "w") as _ckpt:
            json.dump(ckpt, _ckpt)
        logger.info(f"θ checkpoint saved → {checkpoint_path} (θ={magister.theta:.4f})")

    # AUC-ROC and False Positive Rate
    logger.info("Computing AUC-ROC and false positive rate")
    eval_prompts = praeco.batch_prompts(n_eval_samples)

    baseline_texts = [
        backbone.generate(p, max_new_tokens=100)["text"]
        for p in eval_prompts
    ]

    if watermark_mode == "none":
        wm_texts_eval = [
            backbone.generate(p, max_new_tokens=100)["text"]
            for p in eval_prompts
        ]
    else:
        # For PECCAVI with adaptive theta, set context-specific theta per eval prompt
        wm_texts_eval = []
        for p in eval_prompts:
            if featurizer is not None and magister is not None:
                generator.theta = magister.compute_theta(featurizer.extract(p))
            wm_texts_eval.append(generator.generate(p, max_tokens=100))

    use_generator_z = watermark_mode in ("kgw", "sir", "dipmark")
    if use_generator_z:
        z_scores_all = (
            [generator.z_score(t) for t in baseline_texts]
            + [generator.z_score(t) for t in wm_texts_eval]
        )
        baseline_z = [generator.z_score(t) for t in baseline_texts]
    else:
        z_scores_all = [custos.z_score(t) for t in baseline_texts + wm_texts_eval]
        baseline_z = [custos.z_score(t) for t in baseline_texts]

    labels = [0] * n_eval_samples + [1] * n_eval_samples
    auc = roc_auc_score(labels, z_scores_all)

    # TPR @ 1% FPR — standard detection metric for watermarking papers
    fpr_curve, tpr_curve, _ = roc_curve(labels, z_scores_all)
    tpr_at_1fpr = float(np.interp(0.01, fpr_curve, tpr_curve))

    z_threshold = 4.0
    fp = sum(1 for z in baseline_z if z >= z_threshold)
    fpr = fp / len(baseline_texts)

    # Perplexity ratio: PPL(watermarked) / PPL(baseline) — should be close to 1.0
    logger.info("Computing perplexity ratio (GPT-2)...")
    sample_size = min(50, n_eval_samples)
    ppl_baseline = [perplexity(t) for t in baseline_texts[:sample_size]]
    ppl_wm = [perplexity(t) for t in wm_texts_eval[:sample_size]]
    valid = [(b, w) for b, w in zip(ppl_baseline, ppl_wm)
             if not (b != b or w != w)]  # drop NaN pairs
    if valid:
        avg_ppl_baseline = round(float(np.mean([b for b, _ in valid])), 4)
        avg_ppl_wm = round(float(np.mean([w for _, w in valid])), 4)
        ppl_ratio = round(avg_ppl_wm / max(avg_ppl_baseline, 1e-6), 4)
    else:
        avg_ppl_baseline = avg_ppl_wm = ppl_ratio = float("nan")

    # Per-attack survival at multiple detection thresholds — enables survival-vs-threshold curve.
    # Saving raw z-scores lets us recompute at any threshold without re-running.
    logger.info("Computing per-attack survival rates...")
    THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    attack_names = ["lexical", "syntactic", "semantic", "lm_paraphrase", "gpt4_paraphrase"]
    attack_survival: Dict[str, float] = {}
    attack_z_scores: Dict[str, List[float]] = {}
    attack_survival_by_threshold: Dict[str, Dict[str, float]] = {}
    sample_attack = min(30, n_eval_samples)
    _attack_techniques = [
        scriba.lexical_attack,
        scriba.syntactic_attack,
        scriba.semantic_attack,
        lambda t: scriba.lm_paraphrase(t, "Rephrase the following:\n\n{text}"),
        scriba.gpt4_paraphrase,
    ]
    for attack_idx, attack_name in enumerate(attack_names):
        z_list = []
        for wm_text in wm_texts_eval[:sample_attack]:
            attacked = _attack_techniques[attack_idx](wm_text)
            z_val = (generator.z_score(attacked) if use_generator_z
                     else custos.z_score(attacked))
            z_list.append(z_val)
        attack_z_scores[attack_name] = [round(z, 4) for z in z_list]
        survival = sum(1 for z in z_list if z >= z_threshold) / max(len(z_list), 1)
        attack_survival[attack_name] = round(survival, 4)
        attack_survival_by_threshold[attack_name] = {
            f"z{t:.1f}": round(sum(1 for z in z_list if z >= t) / max(len(z_list), 1), 4)
            for t in THRESHOLDS
        }

    logger.info("Building detailed eval records...")
    EVAL_DETAIL_LIMIT = 50
    for i, prompt in enumerate(eval_prompts):
        wm_text = wm_texts_eval[i]
        paraphrases = scriba.paraphrase(wm_text) if i < EVAL_DETAIL_LIMIT else []
        wm_text = wm_texts_eval[i]
        s_orig_i = custos.watermark_score(wm_text)
        s_eff_i = custos.effective_score(paraphrases)
        z_i = generator.z_score(wm_text) if use_generator_z else custos.z_score(wm_text)
        detailed_records.append({
            "phase": "eval",
            "dataset": praeco.get_source(prompt),
            "prompt": prompt,
            "baseline_text": baseline_texts[i],
            "watermarked_text": wm_text,
            "paraphrases": paraphrases,
            "scores": {
                "s_orig": round(s_orig_i, 4),
                "z_score": round(z_i, 4),
                "s_eff": round(s_eff_i, 4),
                "is_watermarked": z_i >= z_threshold,
            },
        })

    first_eff = history[0]["effective_score"] if history else 0.0
    last_eff = history[-1]["effective_score"] if history else 0.0
    improvement = (last_eff - first_eff) / max(first_eff, 1e-6) * 100 if history else 0.0

    avg_readability = round(sum(r["readability"] for r in history) / len(history), 2) if history else 0.0
    avg_gpt4_quality = round(sum(r["gpt4_quality"] for r in history) / len(history), 2) if history else 0.0
    avg_retention = round(sum(r["retention_rate"] for r in history) / len(history), 4) if history else 0.0

    # Mean θ per prompt-entropy quartile — primary evidence for content-adaptive claim.
    # Spread = Q4_creative - Q1_factual; ≥ 1.0 confirms meaningful adaptation.
    theta_by_quartile: Dict = {}
    if theta_by_prompt:
        _ents = np.array([d["entropy"] for d in theta_by_prompt])
        _thts = np.array([d["theta_context"] for d in theta_by_prompt])
        q25, q50, q75 = np.percentile(_ents, [25, 50, 75])
        _q1 = _thts[_ents <= q25]
        _q4 = _thts[_ents >= q75]
        _q2 = _thts[(_ents > q25) & (_ents < q50)]
        _q3 = _thts[(_ents >= q50) & (_ents < q75)]
        def _safe_mean(arr):
            return round(float(np.mean(arr)), 4) if len(arr) > 0 else None
        _spread = round(float(np.mean(_q4) - np.mean(_q1)), 4) if (len(_q1) > 0 and len(_q4) > 0) else None
        theta_by_quartile = {
            "Q1_factual":  _safe_mean(_q1),
            "Q2":          _safe_mean(_q2),
            "Q3":          _safe_mean(_q3),
            "Q4_creative": _safe_mean(_q4),
            "spread":      _spread,
        }
        logger.info(
            f"θ-by-quartile | Q1={theta_by_quartile['Q1_factual']:.3f} "
            f"Q4={theta_by_quartile['Q4_creative']:.3f} "
            f"spread={theta_by_quartile['spread']:.3f}"
        )

    summary = {
        "watermark_mode": watermark_mode,
        "seed": seed,
        "theta_final": history[-1]["theta"] if history else theta_init,
        "effective_score_final": last_eff,
        "effective_score_improvement_pct": round(improvement, 2),
        "avg_retention_rate": avg_retention,
        "meets_85pct_retention": avg_retention >= 0.85,
        "auc_roc": round(auc, 4),
        "tpr_at_1fpr": round(tpr_at_1fpr, 4),
        "false_positive_rate": round(fpr, 4),
        "meets_90pct_auc": auc >= 0.90,
        "avg_ppl_baseline": avg_ppl_baseline,
        "avg_ppl_watermarked": avg_ppl_wm,
        "ppl_ratio": ppl_ratio,
        "attack_survival": attack_survival,
        "attack_z_scores": attack_z_scores,
        "attack_survival_by_threshold": attack_survival_by_threshold,
        "gpt4_survival": attack_survival.get("gpt4_paraphrase"),
        "adaptive_theta": adaptive_theta,
        "w_final": magister.w.tolist() if (magister and adaptive_theta) else None,
        "w_feature_names": ["token_entropy", "length_norm", "vocab_diversity", "avg_token_len_norm"],
        "theta_by_prompt": theta_by_prompt,
        "theta_by_quartile": theta_by_quartile,
        "avg_readability": avg_readability,
        "avg_gpt4_quality": avg_gpt4_quality,
        "meets_readability_45": avg_readability >= 4.5,
        "meets_readability_30": avg_readability >= 3.0,
        "history": history,
        "detailed_records": detailed_records,
    }
    return summary
