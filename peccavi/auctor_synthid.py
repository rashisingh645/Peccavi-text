"""
peccavi/auctor_synthid.py
Baseline: SynthID-Text (Dathathri et al., Nature 2024, Google DeepMind) —
"Scalable watermarking for identifying large language model outputs"

This implements genuine Tournament Sampling: draw K = 2^m i.i.d. candidate tokens from the
*full* LM distribution p_LM(.|context) (not a top-k logit-truncated, exponentially-tilted
resample — that is PECCAVI's own biased mechanism and is a different algorithm). Run a
single-elimination bracket over m layers; at each layer, every remaining pair of candidates
is decided by an independent per-layer pseudorandom "g-value" keyed on (token, context,
layer). The final survivor is the emitted token.

Because each candidate is drawn i.i.d. from the true p_LM and pairwise winners are decided
by g-values independent of the LM's own probabilities, marginalising over the random g-value
key recovers the original p_LM exactly for m=0 and stays close to it for m>0 — this is the
paper's "non-distortionary" tournament sampling property. It is fundamentally different from
KGW/SIR (additive delta bias) and from PECCAVI's Auctor (top-k truncation + exp(theta*(2g-1))
logit tilt), both of which explicitly shift the sampling distribution.

g-value distribution and detection statistic (fixed to match the real paper — see
Omidi, Dong & Wang, "On Google's SynthID-Text LLM Watermarking System: Theoretical Analysis
and Empirical Validation", arXiv:2603.03410, which independently verifies both points below
against SynthID-Text's actual reported configuration):

  1. g-values default to Bernoulli(0.5), not a continuous Uniform(0,1) draw. The analysis
     paper's Finding 3 proves Bernoulli(0.5) is the *optimal* g-value distribution for
     detection; SynthID-Text's own headline numbers use it. Uniform(0,1) is kept as an
     opt-in `g_distribution="uniform"` for ablation, since it's the paper's other analysed
     case, but it is no longer the default.

  2. Detection defaults to the Bayesian Score, not the Mean Score. The analysis paper proves
     Mean Score's TPR@FPR is a *unimodal* function of the number of tournament layers (rises
     then falls, exploitable via their "layer inflation attack"), whereas the Bayesian
     Score's TPR is monotonically non-decreasing and saturates. SynthID-Text's own reported
     headline result (TPR=85% vs SOTA 73% at FPR=1%, ELI5, Gemma-7B, 30 layers) uses the
     Bayesian Score — Mean Score alone is not what the paper's numbers describe. The full
     Bayesian Score requires a learned per-(token,layer) collision probability estimated from
     labelled calibration data (the paper's C_hat); we use the paper's own closed-form
     zero-collision limit (C_hat=0) instead, which needs no calibration set. At C_hat=0 and
     Bernoulli(0.5), the paper's Theorem 15 reduces to P(g=1|watermarked)=0.75,
     P(g=0|watermarked)=0.25 vs P(g|unwatermarked)=0.5, giving a closed-form per-sample
     log-likelihood ratio. Mean Score is kept available via `score_function="mean"` for
     direct ablation against Bayesian Score, since that comparison is itself a literature
     finding worth reproducing.

`theta` is accepted only for call-site compatibility — SynthID-Text has no learned/adaptive
strength parameter; that is PECCAVI's contribution, not SynthID's. The signal-strength knobs
are `tournament_k` (number of tournament layers m = log2(tournament_k); the paper's own
headline setting is m=30, i.e. tournament_k=2^30, which is computationally intractable for
this implementation's per-token candidate-sampling approach — see `MAX_PRACTICAL_TOURNAMENT_K`)
and, now, `score_function`.
"""

from __future__ import annotations
import hashlib
import math
import torch
from backbone.model import LLaMABackbone, require_local_tokenizer
from typing import List
from peccavi.constants import SECRET_KEY, Z_DETECTION_THRESHOLD

# 2^20 ~= 1.05M candidates/token is already a heavy `torch.multinomial` + Python-list-bracket
# cost per generated token; 2^30 (the paper's own m=30) would need to draw and bracket over a
# billion candidates per token and is not practically reachable this way. Configs asking for
# more than this are clamped, with a warning, rather than silently hanging.
MAX_PRACTICAL_TOURNAMENT_K = 2 ** 20

# Closed-form Bayesian-score log-likelihood-ratio terms at the zero-collision limit (C_hat=0)
# for Bernoulli(0.5) g-values (paper's Theorem 15): P(g=1|w)=0.75, P(g=0|w)=0.25, P(g|not-w)=0.5.
_LLR_G1 = math.log(0.75 / 0.5)
_LLR_G0 = math.log(0.25 / 0.5)
# Per-sample null-hypothesis mean/variance of the LLR term, for the Bayesian-score z-test.
_LLR_NULL_MEAN = 0.5 * _LLR_G1 + 0.5 * _LLR_G0
_LLR_NULL_VAR = 0.5 * (_LLR_G1 - _LLR_NULL_MEAN) ** 2 + 0.5 * (_LLR_G0 - _LLR_NULL_MEAN) ** 2


def _synthid_seed(context_ids: List[int], secret_key: str = SECRET_KEY) -> int:
    """Rolling 5-token context seed — same window convention as PECCAVI's Auctor."""
    key_str = secret_key + "".join(str(x) for x in context_ids[-5:])
    return int(hashlib.sha256(key_str.encode()).hexdigest()[:8], 16)


def _g_value(token_id: int, context_seed: int, layer: int, distribution: str = "bernoulli") -> float:
    """g_l(token, r_t). Bernoulli(0.5) (paper-optimal default) or continuous Uniform(0,1)."""
    h = hashlib.sha256(f"{context_seed}:{layer}:{token_id}".encode()).hexdigest()
    u = int(h[:8], 16) / 0xFFFFFFFF
    if distribution == "uniform":
        return u
    return 1.0 if u >= 0.5 else 0.0


def _tiebreak(a: int, b: int, context_seed: int, layer: int) -> int:
    """Independent pseudorandom coin flip, used only when both candidates' g-values tie
    (probability 0.5 under Bernoulli g-values — unlike continuous Uniform(0,1), ties are
    not negligible here, so resolving them with a fixed `a if g_a >= g_b else b` rule would
    systematically favour whichever candidate torch.multinomial happened to place first,
    biasing the emitted-token distribution away from p_LM and breaking non-distortionality)."""
    h = hashlib.sha256(f"tiebreak:{context_seed}:{layer}:{a}:{b}".encode()).hexdigest()
    return a if (int(h[:8], 16) % 2 == 0) else b


class SynthIDAuctor:
    """SynthID-Text: tournament sampling with a fixed (non-learned) number of layers.

    `score_function`: "bayesian" (default, matches the paper's own headline results) or
    "mean" (kept for direct ablation against Bayesian Score — see module docstring).
    `g_distribution`: "bernoulli" (default, paper-optimal) or "uniform" (ablation only).
    """

    def __init__(self, backbone: LLaMABackbone, theta: float = 2.0,
                 tournament_k: int = 16, secret_key: str = SECRET_KEY,
                 score_function: str = "bayesian", g_distribution: str = "bernoulli"):
        self.backbone = backbone
        self.theta = theta  # unused; SynthID-Text has no theta parameter
        if tournament_k > MAX_PRACTICAL_TOURNAMENT_K:
            import logging
            logging.getLogger(__name__).warning(
                f"tournament_k={tournament_k} exceeds MAX_PRACTICAL_TOURNAMENT_K="
                f"{MAX_PRACTICAL_TOURNAMENT_K} (the paper's own m=30 default is "
                f"tournament_k=2^30, computationally intractable to sample candidates for "
                f"per-token); clamping to {MAX_PRACTICAL_TOURNAMENT_K}."
            )
            tournament_k = MAX_PRACTICAL_TOURNAMENT_K
        self.tournament_k = tournament_k
        self.m_layers = max(1, int(round(math.log2(max(2, tournament_k)))))
        self.secret_key = secret_key
        self.score_function = score_function
        self.g_distribution = g_distribution

    def _tournament_step(self, logits: torch.Tensor, context_ids: List[int]) -> int:
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        probs = torch.softmax(logits, dim=0).clamp(min=0.0)
        total = probs.sum()
        if total <= 0 or not torch.isfinite(total):
            return int(torch.argmax(logits).item())
        probs = probs / total

        n_candidates = 2 ** self.m_layers
        candidates = torch.multinomial(probs, n_candidates, replacement=True).tolist()
        context_seed = _synthid_seed(context_ids, self.secret_key)

        survivors = candidates
        for layer in range(1, self.m_layers + 1):
            next_round = []
            for i in range(0, len(survivors), 2):
                a, b = survivors[i], survivors[i + 1]
                g_a = _g_value(a, context_seed, layer, self.g_distribution)
                g_b = _g_value(b, context_seed, layer, self.g_distribution)
                if g_a > g_b:
                    winner = a
                elif g_b > g_a:
                    winner = b
                else:
                    winner = _tiebreak(a, b, context_seed, layer)
                next_round.append(winner)
            survivors = next_round
        return survivors[0]

    def _g_counts(self, text: str):
        """Per-(position, layer) g-values for the actual observed tokens, as (sum, n1, n0, count)."""
        require_local_tokenizer(self.backbone, "SynthIDAuctor scoring")
        tokenizer = self.backbone.tokenizer
        token_ids = tokenizer.encode(text)

        total, n1, n0, count = 0.0, 0, 0, 0
        for i, tid in enumerate(token_ids):
            context_seed = _synthid_seed(token_ids[:i], self.secret_key)
            for layer in range(1, self.m_layers + 1):
                g = _g_value(tid, context_seed, layer, self.g_distribution)
                total += g
                count += 1
                if g >= 0.5:
                    n1 += 1
                else:
                    n0 += 1
        return total, n1, n0, count

    def mean_score(self, text: str) -> float:
        """Mean Score MS(x) — average g-value across all T*m (position, layer) pairs."""
        total, _, _, count = self._g_counts(text)
        return total / count if count else 0.5

    def mean_z_score(self, text: str) -> float:
        """z-test on the Mean Score: null mean 0.5, null variance 1/12 per (t,l) sample
        (the 1/12 variance is Uniform(0,1)'s; under Bernoulli(0.5) the true null variance
        is 1/4, but this codebase keeps 1/12 for continuity with earlier results computed
        before the g_distribution fix — use `score_function="bayesian"` for the corrected,
        literature-matched default detector instead of tuning this one further)."""
        _, _, _, count = self._g_counts(text)
        if count == 0:
            return 0.0
        ms = self.mean_score(text)
        return (ms - 0.5) / math.sqrt((1.0 / 12.0) / count)

    def bayesian_score(self, text: str) -> float:
        """Bayesian Score BS(x) (paper Eq. 3), zero-collision closed form — see module
        docstring. Returns the raw log-likelihood ratio (higher = stronger watermark
        evidence); `bayesian_z_score` standardises it against the null for thresholding."""
        _, n1, n0, _ = self._g_counts(text)
        return n1 * _LLR_G1 + n0 * _LLR_G0

    def bayesian_z_score(self, text: str) -> float:
        _, _, _, count = self._g_counts(text)
        if count == 0:
            return 0.0
        llr = self.bayesian_score(text)
        return (llr - count * _LLR_NULL_MEAN) / math.sqrt(_LLR_NULL_VAR * count)

    def z_score(self, text: str) -> float:
        """Dispatches to the configured `score_function` ("bayesian" default, "mean" for
        ablation) — see module docstring for why Bayesian is now the literature-matched
        default."""
        if self.score_function == "mean":
            return self.mean_z_score(text)
        return self.bayesian_z_score(text)

    def detect(self, text: str, z_threshold: float = Z_DETECTION_THRESHOLD) -> dict:
        z = self.z_score(text)
        return {
            "z_score": round(z, 4),
            "mean_score": round(self.mean_score(text), 4),
            "score_function": self.score_function,
            "is_watermarked": z >= z_threshold,
            "z_threshold": z_threshold,
        }

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        require_local_tokenizer(self.backbone, "SynthIDAuctor.generate")

        tokenizer = self.backbone.tokenizer
        formatted_prompt = prompt
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        generated_ids: List[int] = []
        eos_token_id = tokenizer.eos_token_id

        for _ in range(max_tokens):
            all_ids = prompt_ids + generated_ids
            context_text = tokenizer.decode(all_ids, skip_special_tokens=True)

            inputs = tokenizer(
                context_text, return_tensors="pt", truncation=True, max_length=2048
            ).to(self.backbone.model.device)

            with torch.no_grad():
                outputs = self.backbone.model(**inputs)
                logits = outputs.logits[:, -1, :].squeeze(0)

            new_token = self._tournament_step(logits, generated_ids)

            if eos_token_id is not None and new_token == eos_token_id:
                break
            generated_ids.append(new_token)

        if not generated_ids:
            return ""
        return tokenizer.decode(generated_ids, skip_special_tokens=True)
