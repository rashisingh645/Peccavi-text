"""
peccavi/auctor.py
Agent: Auctor - Watermarked Token Generation via Tournament Sampling.
Implements the modified distribution:
    p_w(x_t | x_<t, θ) ∝ p_LM(x_t | x_<t) * exp(2θ * g(x_t, r_t))
where g(x_t, r_t) ∈ [0,1] is computed via tournament sampling over candidate tokens.

The generation-time logit boost is θ*(2g-1) rather than θ*g (see
_tournament_sample below) so the bias is symmetric around 0 for a "coin-flip"
green score. Since softmax is shift-invariant, exp(θ*(2g-1)) = exp(-θ)*exp(2θ*g)
and the constant exp(-θ) factor cancels in the normalisation — so the actual
effective tilt strength on g is 2θ, not θ. All calibration elsewhere in this
codebase (THETA_MAX, the learned θ range, theory.py's empirical mu(θ)≈0.05θ fit)
is against this real 2θ*(2-symmetric) formula, not the naive exp(θ*g) form.

Inline approach: tournament sampling applied at every generation step,
not post-hoc, so the autoregressive coherence chain is preserved.
"""

from __future__ import annotations
import torch
import hashlib
import numpy as np
from backbone.model import LLaMABackbone, require_local_tokenizer
from typing import List, Tuple
from peccavi.constants import SECRET_KEY, THETA_MAX


def _watermark_score(token_id: int, random_seed: int) -> float:
    """
    g(x_t, r_t): deterministic score in [0,1] derived from token and seed.
    Uses hash-based green/red list partitioning.
    """
    h = hashlib.sha256(f"{random_seed}:{token_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _context_seed(context_ids: List[int], secret_key: str = SECRET_KEY) -> int:
    """Derive a random seed from rolling context window (last 5 tokens)."""
    key_str = secret_key + "".join(str(x) for x in context_ids[-5:])
    return int(hashlib.sha256(key_str.encode()).hexdigest()[:8], 16)


class Auctor:
    """
    Generates watermarked text using tournament sampling with safeguards.
    Samples K candidates, applies watermark-aware re-weighting, then samples winner.
    Includes fallback mechanisms for numerical stability and quality preservation.
    """

    def __init__(self, backbone: LLaMABackbone, theta: float = 2.0,
                 tournament_k: int = 16, secret_key: str = SECRET_KEY):
        self.backbone = backbone
        self.theta = theta
        self.tournament_k = tournament_k
        self.secret_key = secret_key

    def _tournament_sample(self, context: str, context_ids: List[int]) -> Tuple[int, float]:
        """
        Tournament sampling: sample K candidates from the LM distribution,
        apply watermark-aware re-weighting, return winning token and its watermark score.
        
        Returns: (token_id, watermark_score)
        """
        tokenizer = self.backbone.tokenizer
        
        # Get raw logits
        inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=2048).to(self.backbone.model.device)
        with torch.no_grad():
            outputs = self.backbone.model(**inputs)
            logits = outputs.logits[:, -1, :].squeeze(0)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        
        vocab_size = logits.shape[0]
        k = min(self.tournament_k, vocab_size - 1)
        
        if k < 2:
            top_token = torch.argmax(logits).item()
            r_t = _context_seed(context_ids, self.secret_key)
            g_score = _watermark_score(top_token, r_t)
            return top_token, g_score
        
        # Get top-k candidates
        try:
            top_k_logits, top_k_indices = torch.topk(logits, k)
        except Exception:
            top_token = torch.argmax(logits).item()
            r_t = _context_seed(context_ids, self.secret_key)
            g_score = _watermark_score(top_token, r_t)
            return top_token, g_score
        
        candidate_list = top_k_indices.tolist()
        r_t = _context_seed(context_ids, self.secret_key)
        
        # Apply watermark re-weighting: use top-k logits as base, add watermark bias
        biased_scores = []
        for i, tid in enumerate(candidate_list):
            base_logit = top_k_logits[i].item()
            g_score = _watermark_score(tid, r_t)
            effective_theta = min(self.theta, THETA_MAX)
            watermark_boost = effective_theta * (2.0 * g_score - 1.0)  # Scale [-1, 1]
            biased_logit = base_logit + watermark_boost
            biased_scores.append((tid, biased_logit, g_score))

        # Sample from re-weighted distribution
        logits_array = torch.tensor([s[1] for s in biased_scores], dtype=torch.float32, device=logits.device)
        logits_array = torch.nan_to_num(logits_array, nan=0.0, posinf=1e4, neginf=-1e4)
        probs = torch.softmax(logits_array, dim=0)
        probs = probs.clamp(min=0.0)
        prob_sum = probs.sum()
        if prob_sum <= 0 or not torch.isfinite(prob_sum):
            winner_tid, _, winner_g_score = max(biased_scores, key=lambda x: x[1])
            return winner_tid, winner_g_score
        probs = probs / prob_sum
        idx = torch.multinomial(probs, 1).item()
        winner_tid, _, winner_g_score = biased_scores[idx]
        return winner_tid, winner_g_score

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """
        Inline watermarked generation: tournament sampling applied at every
        token step so each token is chosen under the watermarked distribution
        p_w(x_t | x_<t) ∝ p_LM * exp(2θ·g(x_t, r_t)) (see module docstring for
        why the effective exponent is 2θ, not θ).
        This preserves autoregressive coherence — no post-hoc refinement.
        """
        require_local_tokenizer(self.backbone, "Auctor.generate")

        tokenizer = self.backbone.tokenizer

        # Apply chat template if present
        formatted_prompt = prompt
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        generated_ids: List[int] = []
        eos_token_id = tokenizer.eos_token_id

        for _ in range(max_tokens):
            context_ids_full = prompt_ids + generated_ids
            context_text = tokenizer.decode(context_ids_full, skip_special_tokens=True)

            try:
                new_token, _ = self._tournament_sample(context_text, generated_ids)
            except Exception:
                break

            if eos_token_id is not None and new_token == eos_token_id:
                break

            generated_ids.append(new_token)

        if not generated_ids:
            return ""

        return tokenizer.decode(generated_ids, skip_special_tokens=True)



