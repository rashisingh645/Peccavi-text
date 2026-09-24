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
from peccavi.auctor_dipmark import DiPMarkAuctor, per_token_green_flags
from peccavi.auctor_synthid import SynthIDAuctor
from peccavi.scriba import Scriba
from peccavi.custos import Custos
from peccavi.magister import Magister
from peccavi.featurizer import PromptFeaturizer
from peccavi.constants import THETA_MIN, THETA_MAX, Z_DETECTION_THRESHOLD
from eval.quality import flesch_quality_score, perplexity, ORACLE_MODEL_NAME
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


def _norm_z(z: float) -> float:
    """Normalise a z-score to [0,1] assuming z typically falls in [-10,10] —
    clamped since strongly-watermarked long texts can push z outside that range."""
    return min(max((z + 10) / 20, 0.0), 1.0)


def run_peccavi(
    backbone: LLaMABackbone,
    generations: int = 10,
    n_paraphrases: int = 5,
    n_eval_samples: int = 100,
    verbose: bool = True,
    theta_init: float = 2.0,
    tournament_k: int = 16,            # PECCAVI's own Auctor top-k truncation (peccavi mode only)
    watermark_mode: str = "peccavi",   # "peccavi" | "peccavi_df" | "kgw" | "sir" | "none"
    df_alpha_init: float = 0.3,        # peccavi_df: starting DiPmark alpha (distinct from `alpha`, the REINFORCE learning rate below)
    df_alpha_min: float = 0.05,
    df_alpha_max: float = 0.49,
    df_window: int = 5,
    kgw_delta: float = 2.0,
    kgw_gamma: float = 0.5,
    sir_delta: float = 2.0,
    sir_embedding_model: str = "perceptiveshawty/compositional-bert-large-uncased",
    sir_checkpoint_path: str = "results/sir_transform_model.pt",
    sir_proj_dim: int = 1000,
    dipmark_delta: float = 2.0,
    dipmark_gamma: float = 0.5,
    dipmark_window: int = 5,
    synthid_tournament_k: int = 16,
    synthid_score_function: str = "bayesian",
    synthid_g_distribution: str = "bernoulli",
    lam: float = 0.6,
    nu: float = 0.4,
    mu_ppl: float = 0.0,
    rho_survival: float = 0.0,
    alpha: float = 0.05,
    seed: int = 42,
    checkpoint_path: str = None,
    adaptive_theta: bool = False,
    theta_min: float = THETA_MIN,
    theta_max: float = THETA_MAX,
    n_attack_samples: int = 100,
    attack_mix: bool = False,
    max_tokens: int = 100,
) -> Dict:
    _set_seed(seed)

    praeco = Praeco()
    scriba = Scriba(backbone, n_variants=n_paraphrases)
    custos = Custos(backbone)

    train_attack_pool = None
    if attack_mix and rho_survival > 0.0:
        # lexical_attack deliberately excluded: it's a context-blind WordNet synonym
        # substitution that often produces mangled, barely-coherent text. Empirically
        # (mix40 run: theta collapsed below theta_init, AUC fell to near-chance) it
        # injects too much reward noise into REINFORCE to be useful as a training-time
        # attack, even though it's a legitimate eval-time attack to test survival against.
        train_attack_pool = [
            lambda t: scriba.lm_paraphrase(t, "Rephrase the following:\n\n{text}"),
        ]

    if watermark_mode == "kgw":
        generator = KGWAuctor(backbone, delta=kgw_delta, gamma=kgw_gamma)
        magister = None
    elif watermark_mode == "sir":
        generator = SIRAuctor(
            backbone, delta=sir_delta, embedding_model=sir_embedding_model,
            checkpoint_path=sir_checkpoint_path, proj_dim=sir_proj_dim,
        )
        magister = None
    elif watermark_mode == "dipmark":
        generator = DiPMarkAuctor(backbone, delta=dipmark_delta, gamma=dipmark_gamma,
                                  window=dipmark_window)
        magister = None
    elif watermark_mode == "synthid":
        generator = SynthIDAuctor(backbone, theta=theta_init,
                                  tournament_k=synthid_tournament_k,
                                  score_function=synthid_score_function,
                                  g_distribution=synthid_g_distribution)
        magister = None
    elif watermark_mode == "peccavi_df":
        # Distortion-free PECCAVI: DiPmark's provably distribution-preserving reweight
        # (peccavi/auctor_dipmark.py) as the generator, with its strength parameter alpha
        # made content-adaptive and REINFORCE-learned instead of fixed — Magister's `.theta`
        # field holds alpha here (see Magister.update()'s docstring); the bounds passed as
        # theta_min/theta_max are alpha's bounds, not theta's.
        generator = DiPMarkAuctor(backbone, gamma=df_alpha_init, window=df_window)
        magister = Magister(
            backbone, theta_init=df_alpha_init, alpha=alpha, lam=lam, nu=nu,
            mu_ppl=mu_ppl, rho_survival=rho_survival,
            adaptive=adaptive_theta, theta_min=df_alpha_min, theta_max=df_alpha_max,
            df_window=df_window, extra_attacks=train_attack_pool,
        )
    elif watermark_mode == "none":
        generator = None
        magister = None
    else:
        generator = Auctor(backbone, theta=theta_init, tournament_k=tournament_k)
        magister = Magister(
            backbone, theta_init=theta_init, alpha=alpha, lam=lam, nu=nu,
            mu_ppl=mu_ppl, rho_survival=rho_survival,
            adaptive=adaptive_theta, theta_min=theta_min, theta_max=theta_max,
            extra_attacks=train_attack_pool,
        )

    featurizer = PromptFeaturizer(backbone) if (adaptive_theta and watermark_mode in ("peccavi", "peccavi_df")) else None

    history = []
    detailed_records: List[Dict] = []
    theta_by_prompt: List[Dict] = []   # tracks (entropy, theta_context) for paper Figure 2 — theta or alpha, depending on mode

    for gen in range(1, generations + 1):
        try:
            prompt = praeco.next_prompt()
            green_flags = None  # only populated in peccavi_df mode, consumed by magister.update() below

            if watermark_mode in ("peccavi", "peccavi_df"):
                features = featurizer.extract(prompt) if featurizer else None
                context_theta = magister.compute_theta(features)
                if watermark_mode == "peccavi":
                    generator.theta = context_theta
                else:
                    generator.alpha = context_theta
                wm_text = generator.generate(prompt, max_tokens=max_tokens)
            elif watermark_mode in ("kgw", "sir", "dipmark", "synthid"):
                wm_text = generator.generate(prompt, max_tokens=max_tokens)
            else:
                wm_text = backbone.generate(prompt, max_new_tokens=max_tokens)["text"]

            _uses_generator_z = watermark_mode in ("kgw", "sir", "dipmark", "synthid", "peccavi_df")
            if _uses_generator_z:
                original_score = _norm_z(generator.z_score(wm_text))
            else:
                original_score = custos.watermark_score(wm_text)

            paraphrases = scriba.paraphrase(wm_text)
            if _uses_generator_z:
                # Custos's SHA256 g-score measures a different notion of "green" than
                # these generators' own partitions (context-seeded permutation, tournament
                # g-values, etc.) — use each generator's own z_score (same one driving
                # detection) so S_eff reflects what was actually optimised/embedded.
                para_scores_norm = [_norm_z(generator.z_score(p)) for p in paraphrases]
                s_eff = sum(para_scores_norm) / max(len(para_scores_norm), 1)
            else:
                s_eff = custos.effective_score(paraphrases)
                z_eff = custos.effective_z_score(paraphrases)

            z_threshold = Z_DETECTION_THRESHOLD
            if _uses_generator_z:
                para_z = [generator.z_score(p) for p in paraphrases]
                z_eff = sum(para_z) / max(len(para_z), 1)
            else:
                para_z = [custos.z_score(p) for p in paraphrases]
            retention = sum(1 for z in para_z if z >= z_threshold) / max(len(para_z), 1)

            readability = flesch_quality_score(wm_text)

            _feats = features if watermark_mode in ("peccavi", "peccavi_df") else None
            if magister and watermark_mode == "peccavi_df":
                green_flags = per_token_green_flags(backbone, wm_text, generator.alpha,
                                                     generator.window, generator.secret_key)
                new_theta = magister.update(wm_text, s_eff, reference_text=prompt, prompt_features=_feats,
                                             green_flags=green_flags, current_param=generator.alpha)
            elif magister:
                new_theta = magister.update(wm_text, s_eff, reference_text=prompt, prompt_features=_feats)
            else:
                new_theta = theta_init
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
            "theta_context": round(context_theta if watermark_mode in ("peccavi", "peccavi_df") else new_theta, 4),
            "original_score": round(original_score, 4),
            "effective_score": round(s_eff, 4),
            "effective_z_score": round(z_eff, 4),
            "retention_rate": round(retention, 4),
            "readability": readability,
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
                "theta_context": round(context_theta if watermark_mode in ("peccavi", "peccavi_df") else new_theta, 4),
                "readability": readability,
            },
        })

        if verbose:
            logger.info(
                f"Gen {gen:>3} | θ={new_theta:.4f} | "
                f"S_orig={original_score:.4f} | S_eff={s_eff:.4f}"
            )

    # Save theta/alpha checkpoint immediately after training — before any eval that could crash
    if checkpoint_path and watermark_mode in ("peccavi", "peccavi_df") and magister is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        ckpt = {"theta": round(magister.theta, 6)}
        if adaptive_theta:
            ckpt["w"] = magister.w.tolist()
        with open(checkpoint_path, "w") as _ckpt:
            json.dump(ckpt, _ckpt)
        logger.info(f"Checkpoint saved → {checkpoint_path} (value={magister.theta:.4f})")

    # AUC-ROC and False Positive Rate
    logger.info("Computing AUC-ROC and false positive rate")
    eval_prompts = praeco.batch_prompts(n_eval_samples)

    baseline_texts = [
        backbone.generate(p, max_new_tokens=max_tokens)["text"]
        for p in eval_prompts
    ]

    if watermark_mode == "none":
        wm_texts_eval = [
            backbone.generate(p, max_new_tokens=max_tokens)["text"]
            for p in eval_prompts
        ]
    else:
        # For PECCAVI with adaptive theta/alpha, set context-specific value per eval prompt.
        # DiPMarkAuctor (peccavi_df) has no `.theta` attribute — setting one would silently
        # do nothing, since its generate()/z_score() read `.alpha`.
        wm_texts_eval = []
        for p in eval_prompts:
            if featurizer is not None and magister is not None:
                ctx_val = magister.compute_theta(featurizer.extract(p))
                if watermark_mode == "peccavi_df":
                    generator.alpha = ctx_val
                else:
                    generator.theta = ctx_val
            wm_texts_eval.append(generator.generate(p, max_tokens=max_tokens))

    use_generator_z = watermark_mode in ("kgw", "sir", "dipmark", "synthid", "peccavi_df")
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

    z_threshold = Z_DETECTION_THRESHOLD
    fp = sum(1 for z in baseline_z if z >= z_threshold)
    fpr = fp / len(baseline_texts)

    # Perplexity ratio: PPL(watermarked) / PPL(baseline) — should be close to 1.0
    logger.info(f"Computing perplexity ratio (oracle: {ORACLE_MODEL_NAME})...")
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
    # "semantic" was removed entirely — it was an unconditional duplicate of lm_paraphrase.
    logger.info("Computing per-attack survival rates...")
    THRESHOLDS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    attack_names = ["lexical", "syntactic", "lm_paraphrase"]
    _attack_techniques = [
        scriba.lexical_attack,
        scriba.syntactic_attack,
        lambda t: scriba.lm_paraphrase(t, "Rephrase the following:\n\n{text}"),
    ]

    attack_survival: Dict[str, float] = {}
    attack_z_scores: Dict[str, List[float]] = {}
    attack_survival_by_threshold: Dict[str, Dict[str, float]] = {}
    attack_auc: Dict[str, float] = {}
    attack_tpr_at_1fpr: Dict[str, float] = {}
    sample_attack = min(n_attack_samples, n_eval_samples)
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
        # Continuous robustness metric: how well attacked-watermarked text still separates
        # from the unwatermarked baseline z-scores, using every z instead of one hard cutoff.
        try:
            _lbl = [0] * len(baseline_z) + [1] * len(z_list)
            _scr = list(baseline_z) + list(z_list)
            attack_auc[attack_name] = round(float(roc_auc_score(_lbl, _scr)), 4)
            _fpr_c, _tpr_c, _ = roc_curve(_lbl, _scr)
            attack_tpr_at_1fpr[attack_name] = round(float(np.interp(0.01, _fpr_c, _tpr_c)), 4)
        except ValueError:
            attack_auc[attack_name] = None
            attack_tpr_at_1fpr[attack_name] = None

    logger.info("Building detailed eval records...")
    EVAL_DETAIL_LIMIT = 50
    for i, prompt in enumerate(eval_prompts):
        wm_text = wm_texts_eval[i]
        paraphrases = scriba.paraphrase(wm_text) if i < EVAL_DETAIL_LIMIT else []
        wm_text = wm_texts_eval[i]
        if use_generator_z:
            s_orig_i = _norm_z(generator.z_score(wm_text))
            s_eff_i = (
                sum(_norm_z(generator.z_score(p)) for p in paraphrases) / len(paraphrases)
                if paraphrases else 0.0
            )
        else:
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
        "theta_final": history[-1]["theta"] if history else (df_alpha_init if watermark_mode == "peccavi_df" else theta_init),
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
        "attack_auc": attack_auc,
        "attack_tpr_at_1fpr": attack_tpr_at_1fpr,
        "n_attack_samples": sample_attack,
        "adaptive_theta": adaptive_theta,
        "w_final": magister.w.tolist() if (magister and adaptive_theta) else None,
        "w_feature_names": ["token_entropy", "length_norm", "vocab_diversity", "avg_token_len_norm"],
        "theta_by_prompt": theta_by_prompt,
        "theta_by_quartile": theta_by_quartile,
        "avg_readability": avg_readability,
        "meets_readability_45": avg_readability >= 4.5,
        "meets_readability_30": avg_readability >= 3.0,
        "history": history,
        "detailed_records": detailed_records,
    }
    return summary
