"""
peccavi/magister.py
Agent: Magister – Policy Learning via REINFORCE.
Updates watermark parameter θ to maximise composite reward.
"""

from __future__ import annotations
import random
import torch
import numpy as np
from backbone.model import LLaMABackbone
from peccavi.auctor import Auctor, _watermark_score, _context_seed
from peccavi.auctor_dipmark import per_token_green_flags
from peccavi.featurizer import N_FEATURES, FEATURE_NAMES
from typing import List, Optional
import logging
from peccavi.constants import SECRET_KEY, THETA_MIN, THETA_MAX

logger = logging.getLogger(__name__)


def text_quality_score(text: str, backbone=None, reference_text: str = None) -> float:
    """
    `reference_text`, when provided, should be an unwatermarked baseline generation from
    the same prompt — not the prompt itself. BERTScore against the prompt measures topical
    relevance, not quality degradation caused by watermarking, which is what this feeds
    into the reward as (see Magister.update, which builds that baseline before calling in).
    """
    if backbone is None or not text.strip():
        return 0.5

    ppl_score = 0.5  # Default

    if backbone.backend == "transformers" and hasattr(backbone, 'model'):
        try:
            import torch, math
            enc = backbone.tokenizer(text, return_tensors="pt").to(backbone.model.device)
            with torch.no_grad():
                loss = backbone.model(**enc, labels=enc["input_ids"]).loss
            ppl = math.exp(loss.item())
            ppl_score = max(0.0, 1.0 - (ppl - 1) / 99)
        except Exception as e:
            logger.warning(f"Failed to compute perplexity: {e}")
            ppl_score = 0.5

    if reference_text:
        if not hasattr(text_quality_score, "_scorer"):
            try:
                from bert_score import BERTScorer
                text_quality_score._scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            except ImportError:
                logger.warning("bert-score not installed — quality reward uses perplexity only")
                text_quality_score._scorer = None
        if text_quality_score._scorer is None:
            return ppl_score
        try:
            P, R, F1 = text_quality_score._scorer.score([text], [reference_text])
            bert_score_val = F1.mean().item()
            return (ppl_score + bert_score_val) / 2
        except Exception as e:
            logger.warning(f"Failed to compute BERTScore: {e}")
            return ppl_score
    return ppl_score


def composite_reward(
    effective_wm_score: float, quality: float,
    lam: float = 0.6, nu: float = 0.4,
    mu_ppl: float = 0.0, ppl_ratio: float = 1.0,
    rho_survival: float = 0.0, survival_score: float = 0.0,
) -> float:
    """r = λ*S_eff + ν*Q - μ*max(0,PPL_ratio-1) + ρ*survival_after_attack
    rho_survival=0 (default) reproduces the original reward with no attack term."""
    ppl_penalty = mu_ppl * max(0.0, ppl_ratio - 1.0)
    return lam * effective_wm_score + nu * quality - ppl_penalty + rho_survival * survival_score


class Magister:
    def __init__(
        self,
        backbone: LLaMABackbone,
        theta_init: float = 2.0,
        alpha: float = 0.05,
        gamma: float = 0.99,
        lam: float = 0.6,
        nu: float = 0.4,
        mu_ppl: float = 0.0, # weight of perplexity penalty (μ) — disabled by default
        rho_survival: float = 0.0, # weight of attack survival term (ρ) — disabled by default
        secret_key: str = SECRET_KEY,
        adaptive: bool = False,
        theta_min: float = THETA_MIN,
        theta_max: float = THETA_MAX,
        df_window: int = 5,
        extra_attacks: Optional[list] = None,
    ):
        self.backbone = backbone
        self.extra_attacks = extra_attacks or []
        self.theta = theta_init
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.nu = nu
        self.mu_ppl = mu_ppl
        self.rho_survival = rho_survival
        self.secret_key = secret_key
        self.history: List[float] = []
        self.df_window = df_window  # DiPmark context window, used only by the distortion-free path

        self.adaptive = adaptive
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.w: np.ndarray = np.zeros(N_FEATURES, dtype=np.float32)

    def compute_theta(self, features: Optional[np.ndarray] = None) -> float:
        """
        θ(context) = clip(θ_base + w · features, θ_min, θ_max)
w is learbnt weight vector for prmpts and features is the feature vector extracted from the prompt. The dot product w · features gives a context-specific adjustment to the base θ.
        When adaptive=False or features=None, returns the base θ unchanged —
        making this a drop-in replacement for the fixed-θ policy.

        High-entropy prompts (creative writing) → higher θ (stronger watermark).
        Low-entropy prompts (factual Q&A, code) → lower θ (gentle watermark).
        """
        if not self.adaptive or features is None:
            return float(np.clip(self.theta, self.theta_min, self.theta_max))
        raw = self.theta + float(np.dot(self.w, features))
        return float(np.clip(raw, self.theta_min, self.theta_max))

    def feature_report(self) -> dict:
        """Returns the learned weight vector for logging and paper reporting."""
        return {
            "feature_names": FEATURE_NAMES,
            "w": self.w.tolist(),
            "theta_base": round(self.theta, 4),
            "interpretation": {
                name: round(float(wi), 4)
                for name, wi in zip(FEATURE_NAMES, self.w)
            },
        }

    def _lazy_load_marianmt(self) -> None:
        """Load MarianMT EN→FR→EN models lazily on first use, onto the same device
        as the main backbone when it's a GPU — these are small (~300M param) models,
        so they fit comfortably alongside a loaded 7B backbone, and running the
        back-translation attack on GPU instead of CPU is the difference between
        rho_survival being usable in training and being a multi-minute-per-step
        bottleneck (CPU was the deliberate original default, kept here as fallback
        when no GPU is available)."""
        if hasattr(self, "_marian_loaded"):
            return
        try:
            from transformers import MarianMTModel, MarianTokenizer
            self._marian_device = (
                self.backbone.model.device
                if hasattr(self.backbone, "model") and torch.cuda.is_available()
                else "cpu"
            )
            logger.info(f"Loading MarianMT EN→FR→EN for attack-aware training onto {self._marian_device}...")
            self._tok_en_fr = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
            self._mdl_en_fr = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr").to(self._marian_device)
            self._tok_fr_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
            self._mdl_fr_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en").to(self._marian_device)
            self._mdl_en_fr.eval()
            self._mdl_fr_en.eval()
            self._marian_loaded = True
            logger.info(f"MarianMT loaded on {self._marian_device} for back-translation attack")
        except Exception as e:
            logger.warning(f"MarianMT unavailable ({e}) — rho_survival term disabled")
            self._marian_loaded = False

    def _back_translate(self, text: str) -> str:
        """EN→FR→EN back-translation using MarianMT, on whichever device it was loaded onto."""
        if not getattr(self, "_marian_loaded", False) or not text.strip():
            return text
        try:
            import torch
            enc = self._tok_en_fr(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to(self._marian_device)
            with torch.no_grad():
                fr_ids = self._mdl_en_fr.generate(**enc, max_new_tokens=512)
            fr = self._tok_en_fr.decode(fr_ids[0], skip_special_tokens=True)
            enc2 = self._tok_fr_en(
                fr, return_tensors="pt", truncation=True, max_length=512, padding=True
            ).to(self._marian_device)
            with torch.no_grad():
                en_ids = self._mdl_fr_en.generate(**enc2, max_new_tokens=512)
            return self._tok_fr_en.decode(en_ids[0], skip_special_tokens=True)
        except Exception:
            return text

    def _survival_score(self, attacked_text: str, alpha_used: Optional[float] = None) -> float:
        """Watermark survival in [0,1] after back-translation attack.

        Computes a z-score on the attacked text using whichever scheme actually generated
        it — the exponential-tilt hash score by default, or (when `alpha_used` is given,
        i.e. peccavi_df mode) the DiPmark alpha-reweight's own green/red partition, via the
        same per_token_green_flags() helper DiPMarkAuctor.z_score() uses. Scoring
        DiPmark-generated text with the exponential-tilt hash formula would measure a
        statistic unrelated to what was actually optimised during generation. Either way,
        a sigmoid centred at z=2.0 is applied so the reward kicks in above the practical
        detection threshold. Returns 0 if the watermark is fully destroyed, approaching 1
        as the signal survives strongly.
        """
        import math
        if not hasattr(self.backbone, "tokenizer"):
            return 0.0

        if alpha_used is not None:
            flags = per_token_green_flags(self.backbone, attacked_text, alpha_used,
                                           self.df_window, self.secret_key)
            n = len(flags)
            if n == 0:
                return 0.0
            green = sum(flags)
            expected = n * (1.0 - alpha_used)
            variance = n * alpha_used * (1.0 - alpha_used)
            if variance <= 0:
                return 0.0
            z = (green - expected) / math.sqrt(variance)
        else:
            token_ids = self.backbone.tokenizer.encode(attacked_text)
            n = len(token_ids)
            if n == 0:
                return 0.0
            green = sum(
                1 for i, tid in enumerate(token_ids)
                if _watermark_score(tid, _context_seed(token_ids[:i], self.secret_key)) > 0.5
            )
            z = (green - n * 0.5) / max(math.sqrt(n * 0.25), 1e-6)

        return float(1.0 / (1.0 + math.exp(-(z - 2.0))))

    def _policy_gradient(self, token_ids: List[int]) -> float:
        """
        Approximate score function for the exponential-tilt policy
        p_w(x_t) ∝ p_LM(x_t) * exp(θ * g(x_t, r_t)):
            ∇_θ log p_w(x) ≈ Σ_t (g(x_t, r_t) - E[g])
        Centered at the null expectation E[g]=0.5 (g is hash-derived and ~Uniform(0,1)),
        not the raw Σ_t g(x_t, r_t) used previously. The uncentered sum is always
        positive for any watermarked sample (mean g > 0.5 whenever the policy embeds
        any signal at all) regardless of whether that sample was better or worse than
        the policy's current average — it doesn't discriminate signal from chance, it
        just grows with token count. Centering makes the gradient reflect how far above
        or below chance this specific sample landed, which is what a score-function
        estimator is supposed to measure.
        """
        total = 0.0
        for i, tid in enumerate(token_ids):
            r_t = _context_seed(token_ids[:i], self.secret_key)
            g = _watermark_score(tid, r_t)
            total += (g - 0.5)
        return total

    def _policy_gradient_df(self, green_flags: List[bool], alpha_used: float) -> float:
        """
        Distortion-free analogue of _policy_gradient, for use when the generator is the
        DiPmark-style alpha-reweight (peccavi_df mode) rather than the biased exponential
        tilt. There the tunable parameter is alpha, not theta, and the null expectation of
        landing in the boosted ("green") region under a uniformly random permutation is
        exactly (1 - alpha) — that's the same quantity used as the null in
        DiPMarkAuctor.z_score(). Centering there instead of at 0.5 keeps this the same
        "how far above/below chance did this sample land" score-function heuristic as
        _policy_gradient, just referenced against alpha-reweight's own null rate rather
        than the exponential-tilt scheme's fixed 0.5.

        This is a heuristic proxy, not an exact derivative of log p_alpha(x) (which is
        piecewise through F_alpha and not smooth in alpha at the interval boundaries) —
        the same honesty caveat that already applies to _policy_gradient's g-0.5 proxy
        for the exponential-tilt case.
        """
        if not green_flags:
            return 0.0
        null_rate = 1.0 - alpha_used
        return sum((1.0 if g else 0.0) - null_rate for g in green_flags)

    def update(
        self,
        generated_text: str,
        effective_wm_score: float,
        reference_text: str = None,
        prompt_features: Optional[np.ndarray] = None,
        green_flags: Optional[List[bool]] = None,
        current_param: Optional[float] = None,
    ) -> float:
        """
        One REINFORCE update step. Updates both the base θ and, when
        adaptive=True, the feature weight vector w.

        `reference_text` is the source prompt. When the quality term (nu>0) or the
        PPL penalty (mu_ppl>0) are active, an unwatermarked baseline generation from
        this same prompt is produced and used as the comparison text for both — comparing
        the watermarked output against the prompt itself measures "how different is this
        from its own instruction," not "how much did watermarking degrade quality
        relative to not watermarking," which is what these terms are meant to capture
        (and what the eval-time ppl_ratio metric actually compares against). This costs
        one extra backbone.generate() call per update whenever those terms are active.

        `green_flags`/`current_param`: pass these (peccavi_df mode only) to use the
        distortion-free policy-gradient proxy (_policy_gradient_df) instead of the
        exponential-tilt one — see that method's docstring. `green_flags` is the
        per-token green/red trace from DiPMarkAuctor's own reweight for this exact
        generation, and `current_param` is the alpha value used to produce it. The field
        this method returns and updates is still named `theta`/`self.theta` regardless of
        mode — it holds theta for the biased Auctor and alpha for the distortion-free one;
        the caller decides what to do with the returned scalar (assign to `.theta` or
        `.alpha` on the generator, respectively).

        Returns the updated base parameter. Use compute_theta(features) to get
        the context-specific value for the next prompt.
        """
        baseline_text = None
        if (self.nu > 0.0 or self.mu_ppl > 0.0) and reference_text and hasattr(self.backbone, "tokenizer"):
            try:
                raw = self.backbone.generate(reference_text, max_new_tokens=100)
                baseline_text = raw["text"] if isinstance(raw, dict) else raw
            except Exception as e:
                logger.warning(f"Failed to generate unwatermarked baseline for reward reference: {e}")

        quality = text_quality_score(generated_text, self.backbone, baseline_text)
        # Compute backbone PPL ratio for penalty term (1.0 = no cost; >1.0 penalised).
        # Compares against the unwatermarked baseline generation, not the prompt — ratio > 1
        # means watermarking made the text harder for LLaMA to predict than an unwatermarked
        # generation from the same prompt would be, indicating quality degradation.
        ppl_ratio = 1.0
        if self.mu_ppl > 0.0 and self.backbone.backend == "transformers" and hasattr(self.backbone, "model"):
            try:
                import math
                ref_text_for_ppl = baseline_text or reference_text or generated_text
                ref_enc = self.backbone.tokenizer(ref_text_for_ppl, return_tensors="pt").to(self.backbone.model.device)
                gen_enc = self.backbone.tokenizer(generated_text, return_tensors="pt").to(self.backbone.model.device)
                import torch
                with torch.no_grad():
                    ppl_ref = math.exp(self.backbone.model(**ref_enc, labels=ref_enc["input_ids"]).loss.item())
                    ppl_gen = math.exp(self.backbone.model(**gen_enc, labels=gen_enc["input_ids"]).loss.item())
                ppl_ratio = ppl_gen / max(ppl_ref, 1e-6)
            except Exception:
                pass
        survival_score = 0.0
        if self.rho_survival > 0.0:
            self._lazy_load_marianmt()
            if getattr(self, "_marian_loaded", False):
                attack_fn = (random.choice([self._back_translate] + self.extra_attacks)
                             if self.extra_attacks else self._back_translate)
                attacked = attack_fn(generated_text)
                survival_score = self._survival_score(attacked, alpha_used=current_param if green_flags is not None else None)

        reward = composite_reward(
            effective_wm_score, quality, self.lam, self.nu,
            self.mu_ppl, ppl_ratio, self.rho_survival, survival_score,
        )

        self.history.append(reward)
        if green_flags is not None:
            # Distortion-free path (peccavi_df): the caller already computed, during
            # generation-time inference, whether each token landed in the DiPmark
            # alpha-reweight's boosted region — reused here instead of re-deriving it.
            if current_param is None:
                raise ValueError("current_param (the alpha used to generate this text) is required when green_flags is provided")
            grad = self._policy_gradient_df(green_flags, current_param)
        else:
            if hasattr(self.backbone, "tokenizer"):
                token_ids = self.backbone.tokenizer.encode(generated_text)
            else:
                token_ids = generated_text.split()
            grad = self._policy_gradient(token_ids)

        baseline = float(np.mean(self.history[-20:])) if len(self.history) >= 5 else 0.5
        advantage = reward - baseline

        # Update base θ (same as non-adaptive policy)
        self.theta += self.alpha * grad * advantage
        self.theta = float(np.clip(self.theta, self.theta_min, self.theta_max))

        # Update feature weights w (only when adaptive and features provided)
        # ∇_w J ≈ grad * advantage * features  (REINFORCE for linear policy)
        if self.adaptive and prompt_features is not None:
            self.w += self.alpha * grad * advantage * prompt_features
            # Clip w to prevent unbounded growth; ±3 allows θ to swing ±3 units
            self.w = np.clip(self.w, -3.0, 3.0)

        logger.debug(
            f"θ_base={self.theta:.4f} | reward={reward:.4f} | "
            f"advantage={advantage:.4f}"
            + (f" | survival={survival_score:.3f}" if self.rho_survival > 0 else "")
            + (f" | w={self.w.tolist()}" if self.adaptive else "")
        )
        return self.theta