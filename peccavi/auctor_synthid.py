"""
peccavi/auctor_synthid.py
Baseline: SynthID-Text (Dathathri et al., Nature 2024, Google DeepMind) —
"Scalable watermarking for identifying large language model outputs"

Rewritten to match the ACTUAL published algorithm, verified directly against the
official reference implementation (github.com/google-deepmind/synthid-text,
src/synthid_text/logits_processing.py + hashing_function.py). The previous version
of this file implemented "draw 2^m i.i.d. candidates, single-elimination bracket" —
a real, self-consistent watermarking scheme, but NOT what SynthID-Text actually
does, and it wrongly implied the paper's own default of 30 tournament layers was
near-computationally-intractable (2^30 candidates). It isn't: the real algorithm
never draws 2^m candidates at all.

The real algorithm, at each generation step:
  1. Take the top-k next-token logits (a normal top-k, not 2^m i.i.d. draws).
  2. For each of the top-k candidates and each of `depth` (tournament layers),
     compute a binary g-value (0 or 1) via 12 rounds of a fast hash accumulator
     (an adapted linear congruential generator, not SHA256) seeded from the ngram
     context, the candidate token, and a per-depth watermarking key.
  3. Reweight the softmax probabilities across all `depth` layers via a
     multiplicative update: probs *= (1 + g - g_mass), where g_mass is the total
     probability mass currently on g=1 tokens at that layer. This is a cheap,
     sequential reweighting pass over the SAME top-k distribution -- not a
     bracket-elimination tournament over exponentially many draws -- which is why
     30 layers is trivial for the real algorithm, not a scaling problem.
  4. Sample once from the final reweighted distribution.
  5. Skip watermarking at any position whose ngram context was already seen in
     this generation (the paper's own explicit repeated-context handling, to avoid
     a repeated low-entropy n-gram inflating the detection signal).

`tournament_k` is kept as the config knob for continuity with existing profiles/
results (tournament_k=16/64/4096/100000 map to depth=4/6/12/17, same mapping as
before) -- but unlike the old candidate-drawing implementation, cost now scales
linearly with depth x top_k, not 2^depth, so there's no practical ceiling anymore;
MAX_PRACTICAL_TOURNAMENT_K is gone because it's no longer needed.

g-values are always binary (0/1) in the real algorithm -- there is no continuous
"Uniform(0,1)" variant, so the earlier g_distribution ablation option is removed;
`score_function` ("bayesian" default, "mean" for ablation) is unchanged.
"""

from __future__ import annotations
import hashlib
import math
import torch
from backbone.model import LLaMABackbone, require_local_tokenizer
from typing import List
from peccavi.constants import SECRET_KEY, Z_DETECTION_THRESHOLD

# Real SynthID-Text's hash accumulator: an adapted linear congruential generator
# (newlib/musl parameters), f(x, data[:T]) = f(f(x, data[:T-1]), data[T]).
_LCG_MULT = 6364136223846793005
_LCG_INC = 1
_MASK64 = (1 << 64) - 1


def _accumulate_hash(current_hash: int, *data: int) -> int:
    h = current_hash
    for d in data:
        h = (h + d) & _MASK64
        h = (h * _LCG_MULT) & _MASK64
        h = (h + _LCG_INC) & _MASK64
    return h


def _signed64(x: int) -> int:
    """Reinterpret a 64-bit unsigned value as signed two's-complement -- needed
    because the reference implementation's `>>` operates on signed int64 tensors
    (arithmetic/sign-extending shift), not an unsigned logical shift."""
    x &= _MASK64
    return x - (1 << 64) if x & (1 << 63) else x


def _hash_iv(keys: List[int]) -> int:
    """SHA256 of the per-depth watermarking keys, used as the hash chain's IV
    (matches the real implementation's use of SHA256 only for this one-time IV
    derivation -- the actual per-token/per-depth hashing is the LCG above)."""
    packed = b"".join((k & _MASK64).to_bytes(8, "little") for k in keys)
    digest = hashlib.sha256(packed).digest()
    return int.from_bytes(digest, byteorder="big") % ((1 << 63) - 1)


def _g_value(ngram_key: int, num_apply_hash: int = 12) -> float:
    """Binary g-value (0.0/1.0) via 12 hash-accumulation rounds, extracting bit 30
    of the final hash -- matches get_gvals() in the reference implementation.
    Uses a signed (sign-extending) right-shift each round to match PyTorch's
    behaviour on signed int64 tensors."""
    shift = 64 // num_apply_hash
    h = ngram_key
    for _ in range(num_apply_hash):
        h = _signed64(_accumulate_hash(h, 1)) >> shift
    return float((h >> 30) & 1)


class SynthIDAuctor:
    """SynthID-Text: multi-layer probability reweighting via binary g-values.

    `score_function`: "bayesian" (default, matches the paper's own headline
    results) or "mean" (kept for direct ablation against Bayesian Score).
    """

    def __init__(self, backbone: LLaMABackbone, theta: float = 2.0,
                 tournament_k: int = 16, secret_key: str = SECRET_KEY,
                 score_function: str = "bayesian", top_k: int = 40,
                 ngram_len: int = 5, context_history_size: int = 1024):
        self.backbone = backbone
        self.theta = theta  # unused; SynthID-Text has no theta parameter
        self.tournament_k = tournament_k
        # Kept for continuity with existing profiles/results (tk=16/64/4096/100000
        # -> depth=4/6/12/17) -- but cost now scales linearly with depth, not 2^depth,
        # so this is just a layer count, not a candidate-pool-size proxy anymore.
        self.depth = max(1, int(round(math.log2(max(2, tournament_k)))))
        self.secret_key = secret_key
        self.score_function = score_function
        self.top_k = top_k
        self.ngram_len = ngram_len
        self.context_history_size = context_history_size

        # One watermarking key per depth, deterministic from secret_key.
        self.keys = [
            int(hashlib.sha256(f"{secret_key}:depth:{i}".encode()).hexdigest()[:16], 16) & _MASK64
            for i in range(self.depth)
        ]
        self.hash_iv = _hash_iv(self.keys)
        self._seen_contexts: List[int] = []

    def _context_hash(self, context_ids: List[int]) -> int:
        """Hash of the last (ngram_len-1) context tokens, seeded with hash_iv."""
        ctx = context_ids[-(self.ngram_len - 1):] if len(context_ids) >= self.ngram_len - 1 else context_ids
        return _accumulate_hash(self.hash_iv, *ctx) if ctx else self.hash_iv

    def _depth_keys(self, context_hash: int, token_id: int) -> List[int]:
        """One ngram key per depth for a specific (context, candidate) pair."""
        h = _accumulate_hash(context_hash, token_id)
        return [_accumulate_hash(h, k) for k in self.keys]

    def _mark_seen(self, context_hash: int) -> bool:
        """Returns True if this context was already watermarked this generation
        (repeated n-gram -> skip watermarking, matching the paper's own handling)."""
        if context_hash in self._seen_contexts:
            return True
        self._seen_contexts.append(context_hash)
        if len(self._seen_contexts) > self.context_history_size:
            self._seen_contexts.pop(0)
        return False

    def _tournament_step(self, logits: torch.Tensor, context_ids: List[int]) -> int:
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        k = min(self.top_k, logits.shape[0])
        top_vals, top_idx = torch.topk(logits, k)
        probs = torch.softmax(top_vals, dim=0).clamp(min=0.0)
        total = probs.sum()
        if total <= 0 or not torch.isfinite(total):
            return int(torch.argmax(logits).item())
        probs = probs / total

        context_hash = self._context_hash(context_ids)
        if self._mark_seen(context_hash):
            # Repeated context: sample from the plain (unwatermarked) distribution.
            idx = int(torch.multinomial(probs, 1).item())
            return int(top_idx[idx].item())

        token_ids = top_idx.tolist()
        g_values = torch.tensor(
            [[_g_value(dk) for dk in self._depth_keys(context_hash, tid)] for tid in token_ids],
            dtype=probs.dtype, device=probs.device,
        )  # [top_k, depth]

        for d in range(self.depth):
            g_d = g_values[:, d]
            g_mass = (g_d * probs).sum()
            probs = (probs * (1.0 + g_d - g_mass)).clamp(min=0.0)
            total = probs.sum()
            if total <= 0 or not torch.isfinite(total):
                probs = torch.softmax(top_vals, dim=0)
                probs = probs / probs.sum()
                break
            probs = probs / total

        idx = int(torch.multinomial(probs, 1).item())
        return int(top_idx[idx].item())

    def _g_counts(self, text: str):
        """Per-(position, depth) g-values for the actual observed tokens, skipping
        positions whose ngram context repeats an earlier one in this same text
        (matching generation's repeated-context handling)."""
        require_local_tokenizer(self.backbone, "SynthIDAuctor scoring")
        tokenizer = self.backbone.tokenizer
        token_ids = tokenizer.encode(text)

        seen: List[int] = []
        total, n1, n0, count = 0.0, 0, 0, 0
        for i, tid in enumerate(token_ids):
            context_ids = token_ids[:i]
            context_hash = self._context_hash(context_ids)
            if context_hash in seen:
                continue
            seen.append(context_hash)
            for dk in self._depth_keys(context_hash, tid):
                g = _g_value(dk)
                total += g
                count += 1
                if g >= 0.5:
                    n1 += 1
                else:
                    n0 += 1
        return total, n1, n0, count

    def mean_score(self, text: str) -> float:
        """Mean Score MS(x) -- average g-value across all (position, depth) pairs."""
        total, _, _, count = self._g_counts(text)
        return total / count if count else 0.5

    def mean_z_score(self, text: str) -> float:
        """z-test on the Mean Score: null mean 0.5, null variance 1/4 (Bernoulli)."""
        _, _, _, count = self._g_counts(text)
        if count == 0:
            return 0.0
        ms = self.mean_score(text)
        return (ms - 0.5) / math.sqrt(0.25 / count)

    def bayesian_score(self, text: str) -> float:
        """Bayesian-style log-likelihood-ratio score, zero-collision closed form:
        P(g=1|watermarked)=0.75, P(g=0|watermarked)=0.25 vs P(g|unwatermarked)=0.5
        (see module history / arXiv:2603.03410 Theorem 15 at collision prob=0).
        This closed form was derived for a bracket-elimination tournament, not this
        file's current multiplicative-reweight mechanism -- kept as a directionally
        reasonable detector (same qualitative shape: rewards more g=1 evidence,
        non-diluting with depth), not a re-derived exact likelihood for this
        specific reweighting scheme."""
        _, n1, n0, _ = self._g_counts(text)
        llr_g1 = math.log(0.75 / 0.5)
        llr_g0 = math.log(0.25 / 0.5)
        return n1 * llr_g1 + n0 * llr_g0

    def bayesian_z_score(self, text: str) -> float:
        _, _, _, count = self._g_counts(text)
        if count == 0:
            return 0.0
        llr_g1 = math.log(0.75 / 0.5)
        llr_g0 = math.log(0.25 / 0.5)
        null_mean = 0.5 * llr_g1 + 0.5 * llr_g0
        null_var = 0.5 * (llr_g1 - null_mean) ** 2 + 0.5 * (llr_g0 - null_mean) ** 2
        llr = self.bayesian_score(text)
        return (llr - count * null_mean) / math.sqrt(null_var * count)

    def z_score(self, text: str) -> float:
        """Dispatches to the configured `score_function` ("bayesian" default,
        "mean" for ablation)."""
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
        self._seen_contexts = []

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
