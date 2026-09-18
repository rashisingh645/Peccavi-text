"""
peccavi/auctor_dipmark.py
Baseline: DiPmark (Wu, Hu, Guo, Zhang, Huang, ICML 2024) —
"A Resilient and Accessible Distribution-Preserving Watermark for Large Language Models"

This implements the paper's actual distribution-preserving reweight function, not a
KGW-style additive logit bias. At each generation step:
  1. Derive a context-seeded random permutation of the vocabulary from the last `window`
     token IDs (the "cipher").
  2. Order the LM's next-token distribution by that permutation and take its CDF.
  3. Warp the CDF through F_alpha(v) = max(v-alpha, 0) + max(v-(1-alpha), 0), which zeroes
     probability mass whose cumulative position falls below alpha, leaves the middle band
     unchanged, and doubles mass above (1-alpha). alpha=0.5 collapses the middle band and
     recovers a pure green(x2)/red(x0) split (the paper's "gamma-reweight" special case).
  4. Sample the winning token from the reweighted distribution.

Marginalising over the random permutation, each token's reweighted probability equals its
original LM probability exactly — this is the "DiP" (distribution-preserving) property that
gives the paper its name. This is fundamentally different from KGW/SIR, whose additive delta
bias shifts the output distribution outright and is never claimed to be distortion-free.

`gamma` is reused as the paper's alpha in (0, 0.5]. `delta` is accepted only for call-site
compatibility with the other baselines' constructors — DiPmark has no delta parameter; its
signal strength is controlled entirely by alpha.
"""

from __future__ import annotations
import hashlib
import math
import torch
import numpy as np
from backbone.model import LLaMABackbone, require_local_tokenizer
from typing import List
from peccavi.constants import SECRET_KEY, Z_DETECTION_THRESHOLD


def _dipmark_seed(context_ids: List[int], window: int, secret_key: str = SECRET_KEY) -> int:
    """Derive seed from last `window` token IDs — full n-gram context hash."""
    ctx = context_ids[-window:] if len(context_ids) >= window else context_ids
    key_str = secret_key + ":" + ",".join(str(t) for t in ctx)
    return int(hashlib.sha256(key_str.encode()).hexdigest()[:8], 16)


def _permutation(vocab_size: int, seed: int) -> np.ndarray:
    """Context-seeded random permutation of the vocabulary (the DiPmark 'cipher')."""
    rng = np.random.default_rng(seed % (2 ** 32))
    return rng.permutation(vocab_size)


def _f_alpha(v: torch.Tensor, alpha: float) -> torch.Tensor:
    """F_alpha(v) = max(v-alpha, 0) + max(v-(1-alpha), 0)."""
    return torch.clamp(v - alpha, min=0.0) + torch.clamp(v - (1.0 - alpha), min=0.0)


def _alpha_reweight(probs: torch.Tensor, perm: np.ndarray, alpha: float) -> torch.Tensor:
    """DiPmark's distribution-preserving reweight, vectorised over the full vocab."""
    perm_t = torch.as_tensor(perm, dtype=torch.long, device=probs.device)
    p_perm = probs[perm_t]
    cum_high = torch.cumsum(p_perm, dim=0)
    cum_low = cum_high - p_perm
    new_p_perm = _f_alpha(cum_high, alpha) - _f_alpha(cum_low, alpha)
    new_probs = torch.zeros_like(probs)
    new_probs[perm_t] = new_p_perm
    return new_probs


def per_token_green_flags(backbone: LLaMABackbone, text: str, alpha: float,
                           window: int, secret_key: str = SECRET_KEY) -> List[bool]:
    """
    Re-derives the context-seeded permutation and alpha-reweight at each position and
    returns, per token, whether its cumulative-probability interval landed mostly above
    the alpha threshold ("green"). Shared by DiPMarkAuctor.z_score() (aggregated into a
    z-test) and by Magister's distortion-free policy-gradient proxy (aggregated into a
    centered sum) — factored out so the two never compute this differently.
    """
    require_local_tokenizer(backbone, "per_token_green_flags")
    tokenizer = backbone.tokenizer
    token_ids = tokenizer.encode(text)

    flags: List[bool] = []
    for i, tid in enumerate(token_ids):
        context_ids = token_ids[:i]
        context_text = tokenizer.decode(context_ids, skip_special_tokens=True) if context_ids else ""
        inputs = tokenizer(
            context_text, return_tensors="pt", truncation=True, max_length=2048
        ).to(backbone.model.device)
        with torch.no_grad():
            outputs = backbone.model(**inputs)
            logits = outputs.logits[:, -1, :].squeeze(0)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        probs = torch.softmax(logits, dim=0)
        vocab_size = probs.shape[0]

        seed = _dipmark_seed(context_ids, window, secret_key)
        perm = _permutation(vocab_size, seed)
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(vocab_size)
        rank = int(inv_perm[tid])

        p_perm = probs[torch.as_tensor(perm, dtype=torch.long)]
        cum_low = float(p_perm[:rank].sum().item())
        cum_high = cum_low + float(p_perm[rank].item())
        midpoint = (cum_low + cum_high) / 2.0
        flags.append(midpoint > alpha)

    return flags


class DiPMarkAuctor:
    """DiPmark watermarked generation using the paper's actual reweighting mechanism."""

    def __init__(self, backbone: LLaMABackbone, delta: float = 2.0,
                 gamma: float = 0.5, window: int = 5,
                 secret_key: str = SECRET_KEY):
        self.backbone = backbone
        self.alpha = min(max(gamma, 1e-3), 0.5)
        self.window = window
        self.secret_key = secret_key
        self.delta = delta  # unused; DiPmark has no delta parameter

    def _reweighted_probs(self, logits: torch.Tensor, context_ids: List[int]) -> torch.Tensor:
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        probs = torch.softmax(logits, dim=0)
        vocab_size = probs.shape[0]
        seed = _dipmark_seed(context_ids, self.window, self.secret_key)
        perm = _permutation(vocab_size, seed)
        new_probs = _alpha_reweight(probs, perm, self.alpha).clamp(min=0.0)
        total = new_probs.sum()
        if total <= 0 or not torch.isfinite(total):
            return probs
        return new_probs / total

    def z_score(self, text: str) -> float:
        """
        Re-derives the reweight at each position and tests whether the observed token's
        cumulative-probability interval landed above the alpha threshold ("green") more
        often than the null rate of (1-alpha) expects:
            z = (count_green - n*(1-alpha)) / sqrt(n*alpha*(1-alpha))
        This requires re-running the LM at every position (like SIR's entropy-gated
        detector already does in this codebase) because it faithfully tests the paper's
        probability-weighted partition, rather than falling back to the paper's alternative
        vocabulary-only "accessible" black-box detector (not implemented here).
        """
        flags = per_token_green_flags(self.backbone, text, self.alpha, self.window, self.secret_key)
        n = len(flags)
        if n == 0:
            return 0.0

        green_count = sum(flags)
        expected = n * (1.0 - self.alpha)
        variance = n * self.alpha * (1.0 - self.alpha)
        if variance <= 0:
            return 0.0
        return (green_count - expected) / math.sqrt(variance)

    def detect(self, text: str, z_threshold: float = Z_DETECTION_THRESHOLD) -> dict:
        z = self.z_score(text)
        return {"z_score": round(z, 4), "is_watermarked": z >= z_threshold, "z_threshold": z_threshold}

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        require_local_tokenizer(self.backbone, "DiPMarkAuctor.generate")

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

            new_probs = self._reweighted_probs(logits, all_ids)
            new_token = int(torch.multinomial(new_probs, 1).item())

            if eos_token_id is not None and new_token == eos_token_id:
                break
            generated_ids.append(new_token)

        if not generated_ids:
            return ""
        return tokenizer.decode(generated_ids, skip_special_tokens=True)
