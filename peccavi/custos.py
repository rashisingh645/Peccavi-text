"""
peccavi/custos.py
Agent: Custos – Watermark Detection.
Detects watermarks by computing average watermark score across tokens:
    S(x_1:T) = (1/T) * Σ_t g(x_t, r_t)
And effective score across paraphrases:
    S_eff(x_1:T) = mean_i S(x̃^(i)_1:T)
Z-test detection:
    z = (count_green - n * 0.5) / sqrt(n * 0.25)
"""

from __future__ import annotations
from peccavi.auctor import _watermark_score, _context_seed
from backbone.model import LLaMABackbone, require_local_tokenizer
from typing import List
import statistics
import hashlib
import math
from peccavi.constants import SECRET_KEY, Z_DETECTION_THRESHOLD


class Custos:
    def __init__(self, backbone: LLaMABackbone, secret_key: str = SECRET_KEY):
        self.backbone = backbone
        self.secret_key = secret_key

    def _tokenize(self, text: str):
        require_local_tokenizer(self.backbone, "Custos scoring")
        return self.backbone.tokenizer.encode(text)

    def watermark_score(self, text: str) -> float:
        #this is the mean g score across all tokens
        """
        S(x_1:T) = (1/T) * Σ_t g(x_t, r_t)
        Scores all tokens — correct since Auctor watermarks inline at every generation step.
        **INCONSISTENCY #3 — PIPELINE.md says min, code says mean:**
- `PIPELINE.md` Section 4.5: `"S_eff = min_i S(paraphrase_i) — the worst-case score"`
- `PIPELINE.md` Section 4.6: `"S_eff (post-paraphrase minimum score)"`
- Actual code: `statistics.mean(...)` — the mean, not the minimum
- The comment in `custos.py` explains: "min over many samples is dominated by variance and gives a pessimistic floor even when the signal survives"
- A second inconsistency: `peccavi/eval.py` line 85 uses `min(variant_scores)` directly, while `custos.effective_score()` uses `mean`. Two different definitions coexist in the codebase.

        """
        token_ids = self._tokenize(text)
        if not token_ids:
            return 0.0

        scores = []
        for i, tid in enumerate(token_ids):
            if isinstance(tid, str):
                tid_hash = int(hashlib.sha256(tid.encode()).hexdigest()[:8], 16) % 100000
                context_ids = [int(hashlib.sha256(t.encode()).hexdigest()[:8], 16) % 100000
                               for t in token_ids[:i]]
            else:
                tid_hash = tid
                context_ids = token_ids[:i]

            r_t = _context_seed(context_ids, self.secret_key)
            g_score = _watermark_score(tid_hash, r_t)
            scores.append(g_score)

        return statistics.mean(scores) if scores else 0.0

    def z_score(self, text: str) -> float:
        """
        z = (count_green - n * 0.5) / sqrt(n * 0.25)
        Measures statistical significance of green-token bias.
        Positive z indicates watermark signal; threshold ~4.0 for high confidence.
        """
        token_ids = self._tokenize(text)
        n = len(token_ids)
        if n == 0:
            return 0.0

        green_count = 0
        for i, tid in enumerate(token_ids):
            if isinstance(tid, str):
                tid_hash = int(hashlib.sha256(tid.encode()).hexdigest()[:8], 16) % 100000
                context_ids = [int(hashlib.sha256(t.encode()).hexdigest()[:8], 16) % 100000
                               for t in token_ids[:i]]
            else:
                tid_hash = tid
                context_ids = token_ids[:i]

            r_t = _context_seed(context_ids, self.secret_key)
            g_score = _watermark_score(tid_hash, r_t)
            if g_score > 0.5:
                green_count += 1

        return (green_count - n * 0.5) / math.sqrt(n * 0.25)

    def effective_score(self, paraphrases: List[str]) -> float:
        """
        S_eff(x_1:T) = mean_i S(x̃^(i)_1:T)
        Average watermark score across all paraphrased variants.
        Mean is used instead of min: min over many samples is dominated by
        variance and gives a pessimistic floor even when the signal survives.
        """
        if not paraphrases:
            return 0.0
        return statistics.mean(self.watermark_score(p) for p in paraphrases)

    def effective_z_score(self, paraphrases: List[str]) -> float:
        """Average z-score across paraphrases — measures aggregate signal survival."""
        if not paraphrases:
            return 0.0
        return statistics.mean(self.z_score(p) for p in paraphrases)

    def detect(self, text: str, z_threshold: float = Z_DETECTION_THRESHOLD) -> dict:
        """
        Detect watermark using z-test. z >= z_threshold indicates watermarked text.
        z_threshold=4.0 corresponds to p < 0.00003 under H0 (no watermark).
        Also returns raw score for backwards compatibility.
        """
        score = self.watermark_score(text)
        z = self.z_score(text)
        return {
            "score": round(score, 4),
            "z_score": round(z, 4),
            "is_watermarked": z >= z_threshold,
            "z_threshold": z_threshold,
        }
