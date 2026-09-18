"""
peccavi/featurizer.py
PromptFeaturizer — extracts lightweight numerical features from a prompt
that predict how much watermarking flexibility the content allows.

Intuition:
  - High-entropy creative prompts tolerate stronger watermarking (higher θ)
    because there is more token choice at each step.
  - Low-entropy factual prompts need gentle watermarking (lower θ)
    to preserve precision and avoid distorting specific terminology.

Features (all normalised to [0, 1]):
  f0  token_entropy      — LM predictive entropy: average Shannon entropy (bits) of the
                            backbone's own next-token distribution over the prompt,
                            normalised by log2(vocab_size)
  f1  length_norm        — token count / max_len
  f2  vocab_diversity    — unique tokens / total tokens (type-token ratio)
  f3  avg_token_len_norm — mean character length / 12

All features are bounded [0, 1] so the learned weight vector w is directly
interpretable: w[0] > 0 means "assign higher θ to high-entropy prompts".
"""

from __future__ import annotations
import math
import re
from typing import List

import numpy as np
import torch

from backbone.model import require_local_tokenizer

N_FEATURES = 4
FEATURE_NAMES = ["token_entropy", "length_norm", "vocab_diversity", "avg_token_len_norm"]


class PromptFeaturizer:
    """
    Maps a raw prompt string to a fixed-size feature vector for the
    content-adaptive θ policy in Magister.

    Requires a backbone with local next-token logit access (backend="transformers") —
    f0 is genuine LM predictive entropy, computed from the backbone's own forward pass
    over the prompt, not a proxy derived from the prompt text alone.
    """

    def __init__(self, backbone, max_len: int = 512):
        self.backbone = backbone
        self.max_len = max_len

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _lm_predictive_entropy(self, prompt: str) -> float:
        """
        Average Shannon entropy (bits) of the LM's own next-token distribution at each
        position in the prompt, normalised by log2(vocab_size).

        This is genuine predictive entropy — "how many plausible continuations does the
        model see at each point in this prompt" — which is the theoretical justification
        for the adaptive-θ policy (high-entropy prompts have more token choice, so the
        watermark can be embedded more strongly without forcing implausible tokens). A
        prior version approximated this with the Shannon entropy of the prompt's own word
        frequencies (a bag-of-words statistic, unrelated to what the LM actually predicts):
        for short prompts where most words are unique — the common case — that proxy is
        close to log2(n_unique_words) regardless of topic, so it tracked prompt length far
        more than "how many plausible continuations exist." This computes the real thing.
        """
        require_local_tokenizer(self.backbone, "PromptFeaturizer entropy")
        tokenizer = self.backbone.tokenizer
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_len
        ).to(self.backbone.model.device)
        if inputs["input_ids"].shape[1] < 2:
            return 0.0

        with torch.no_grad():
            logits = self.backbone.model(**inputs).logits[0]   # [T, vocab_size]
        logits = logits[:-1]                                    # position t predicts token t+1
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        probs = torch.softmax(logits.float(), dim=-1)
        log2_probs = torch.log2(probs.clamp(min=1e-12))
        entropy_per_pos = -(probs * log2_probs).sum(dim=-1)     # bits, shape [T-1]
        avg_entropy = float(entropy_per_pos.mean().item())

        vocab_size = tokenizer.vocab_size or len(tokenizer)
        return float(min(avg_entropy / math.log2(vocab_size), 1.0))

    def extract(self, prompt: str) -> np.ndarray:
        """Returns feature vector of shape (N_FEATURES,), values in [0, 1]."""
        tokens = self._tokenize(prompt)
        n = len(tokens)
        if n == 0:
            return np.zeros(N_FEATURES, dtype=np.float32)

        # f0: LM predictive entropy (see _lm_predictive_entropy)
        f0 = self._lm_predictive_entropy(prompt)

        # f1: normalised length
        f1 = float(min(n / self.max_len, 1.0))

        # f2: type-token ratio (vocabulary diversity)
        f2 = float(len(set(tokens)) / n)

        # f3: normalised average token length (typical range 3–8 chars)
        f3 = float(min(float(np.mean([len(t) for t in tokens])) / 12.0, 1.0))

        return np.array([f0, f1, f2, f3], dtype=np.float32)