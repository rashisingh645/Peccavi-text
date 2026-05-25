"""
peccavi/auctor_dipmark.py
Baseline: DiPMark (Zhao et al., 2024) — "Provably Robust Multi-bit Watermarking for AI-generated Text"
Uses n-gram context window (last N tokens) for partition seeding instead of KGW's single previous token.
The wider context hash makes the watermark more robust to local word substitutions.
"""

from __future__ import annotations
import hashlib
import math
import torch
import numpy as np
from backbone.model import LLaMABackbone
from typing import List
from peccavi.constants import SECRET_KEY


def _dipmark_seed(context_ids: List[int], window: int, secret_key: str = SECRET_KEY) -> int:
    """Derive seed from last `window` token IDs — full n-gram context hash."""
    ctx = context_ids[-window:] if len(context_ids) >= window else context_ids
    key_str = secret_key + ":" + ",".join(str(t) for t in ctx)
    return int(hashlib.sha256(key_str.encode()).hexdigest()[:8], 16)


def _green_mask(vocab_size: int, seed: int, gamma: float) -> torch.BoolTensor:
    rng = np.random.default_rng(seed % (2 ** 32))
    n_green = int(gamma * vocab_size)
    green_idx = rng.choice(vocab_size, n_green, replace=False)
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    mask[torch.tensor(green_idx, dtype=torch.long)] = True
    return mask


class DiPMarkAuctor:
    """
    DiPMark watermarked generation. At each step:
      1. Partition vocab into green list using last `window` tokens as seed.
      2. Add delta to all green token logits.
      3. Sample from biased distribution.

    Compared to KGW, the n-gram context seed makes it harder to defeat with
    local substitution attacks: changing one token shifts the seed for
    all subsequent positions within the window.
    """

    def __init__(self, backbone: LLaMABackbone, delta: float = 2.0,
                 gamma: float = 0.5, window: int = 5,
                 secret_key: str = SECRET_KEY):
        self.backbone = backbone
        self.delta = delta
        self.gamma = gamma
        self.window = window
        self.secret_key = secret_key

    def z_score(self, text: str) -> float:
        """DiPMark detection z-score using n-gram context window for each position."""
        if not hasattr(self.backbone, "tokenizer"):
            return 0.0
        tokenizer = self.backbone.tokenizer
        token_ids = tokenizer.encode(text)
        n = len(token_ids)
        if n == 0:
            return 0.0

        vocab_size = tokenizer.vocab_size or len(tokenizer)
        green_count = 0
        for i, tid in enumerate(token_ids):
            seed = _dipmark_seed(token_ids[:i], self.window, self.secret_key)
            mask = _green_mask(vocab_size, seed, self.gamma)
            if mask[tid]:
                green_count += 1

        return (green_count - n * self.gamma) / math.sqrt(n * self.gamma * (1 - self.gamma))

    def detect(self, text: str, z_threshold: float = 4.0) -> dict:
        z = self.z_score(text)
        return {"z_score": round(z, 4), "is_watermarked": z >= z_threshold, "z_threshold": z_threshold}

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        if not hasattr(self.backbone, "tokenizer"):
            raw = self.backbone.generate(prompt, max_new_tokens=max_tokens)
            return raw["text"] if isinstance(raw, dict) else raw

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
        vocab_size = tokenizer.vocab_size or len(tokenizer)

        for _ in range(max_tokens):
            all_ids = prompt_ids + generated_ids
            context_text = tokenizer.decode(all_ids, skip_special_tokens=True)

            inputs = tokenizer(
                context_text, return_tensors="pt", truncation=True, max_length=2048
            ).to(self.backbone.model.device)

            with torch.no_grad():
                outputs = self.backbone.model(**inputs)
                logits = outputs.logits[:, -1, :].squeeze(0)

            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

            seed = _dipmark_seed(all_ids, self.window, self.secret_key)
            mask = _green_mask(vocab_size, seed, self.gamma).to(logits.device)
            logits[mask] += self.delta

            probs = torch.softmax(logits, dim=0).clamp(min=0.0)
            prob_sum = probs.sum()
            if prob_sum <= 0 or not torch.isfinite(prob_sum):
                new_token = torch.argmax(logits).item()
            else:
                probs = probs / prob_sum
                new_token = torch.multinomial(probs, 1).item()

            if eos_token_id is not None and new_token == eos_token_id:
                break
            generated_ids.append(new_token)

        if not generated_ids:
            return ""
        return tokenizer.decode(generated_ids, skip_special_tokens=True)
