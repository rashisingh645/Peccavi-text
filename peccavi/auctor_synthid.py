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

Detection uses the paper's Mean Score statistic:
    MS(x) = (1 / (T*m)) * sum_t sum_l g_l(x_t, r_t)
which has expectation 0.5 under the null (unwatermarked) hypothesis and is pushed above 0.5
by watermarked generation, since every tournament round preferentially keeps the
higher-g_l candidate. `theta` is accepted only for call-site compatibility — SynthID-Text
has no learned/adaptive strength parameter; that is PECCAVI's contribution, not SynthID's.
The only signal-strength knob here is `tournament_k` (number of tournament layers
m = log2(tournament_k)).
"""

from __future__ import annotations
import hashlib
import math
import torch
from backbone.model import LLaMABackbone, require_local_tokenizer
from typing import List
from peccavi.constants import SECRET_KEY, Z_DETECTION_THRESHOLD


def _synthid_seed(context_ids: List[int], secret_key: str = SECRET_KEY) -> int:
    """Rolling 5-token context seed — same window convention as PECCAVI's Auctor."""
    key_str = secret_key + "".join(str(x) for x in context_ids[-5:])
    return int(hashlib.sha256(key_str.encode()).hexdigest()[:8], 16)


def _g_value(token_id: int, context_seed: int, layer: int) -> float:
    """g_l(token, r_t): independent per-layer pseudorandom value in [0,1)."""
    h = hashlib.sha256(f"{context_seed}:{layer}:{token_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


class SynthIDAuctor:
    """SynthID-Text: tournament sampling with a fixed (non-learned) number of layers."""

    def __init__(self, backbone: LLaMABackbone, theta: float = 2.0,
                 tournament_k: int = 16, secret_key: str = SECRET_KEY):
        self.backbone = backbone
        self.theta = theta  # unused; SynthID-Text has no theta parameter
        self.tournament_k = tournament_k
        self.m_layers = max(1, int(round(math.log2(max(2, tournament_k)))))
        self.secret_key = secret_key

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
                g_a = _g_value(a, context_seed, layer)
                g_b = _g_value(b, context_seed, layer)
                next_round.append(a if g_a >= g_b else b)
            survivors = next_round
        return survivors[0]

    def mean_score(self, text: str) -> float:
        """Mean Score MS(x) — average g-value across all T*m (position, layer) pairs."""
        require_local_tokenizer(self.backbone, "SynthIDAuctor.mean_score")
        tokenizer = self.backbone.tokenizer
        token_ids = tokenizer.encode(text)
        n = len(token_ids)
        if n == 0:
            return 0.5

        total = 0.0
        count = 0
        for i, tid in enumerate(token_ids):
            context_seed = _synthid_seed(token_ids[:i], self.secret_key)
            for layer in range(1, self.m_layers + 1):
                total += _g_value(tid, context_seed, layer)
                count += 1
        return total / count if count else 0.5

    def z_score(self, text: str) -> float:
        """z-test on the Mean Score: null mean 0.5, null variance 1/12 per (t,l) sample."""
        require_local_tokenizer(self.backbone, "SynthIDAuctor.z_score")
        n_positions = len(self.backbone.tokenizer.encode(text))
        n_samples = n_positions * self.m_layers
        if n_samples == 0:
            return 0.0
        ms = self.mean_score(text)
        return (ms - 0.5) / math.sqrt((1.0 / 12.0) / n_samples)

    def detect(self, text: str, z_threshold: float = Z_DETECTION_THRESHOLD) -> dict:
        z = self.z_score(text)
        return {
            "z_score": round(z, 4),
            "mean_score": round(self.mean_score(text), 4),
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
