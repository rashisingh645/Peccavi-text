"""
peccavi/auctor_sir.py
Baseline: SIR (Liu, Pan, Hu, Meng, Wen, ICLR 2024) —
"A Semantic Invariant Robust Watermark for Large Language Models"

Real SIR has nothing to do with token entropy or a KGW-style hash-seeded green list
(an earlier version of this file mislabeled an entropy-gated KGW variant as "SIR" —
that was a different, unrelated technique). The actual mechanism, from `sir_model.py`:

  1. Embed the preceding text with Compositional-BERT -> e (1024-dim).
  2. Pass e through a small trained network T -> a `proj_dim`-length watermark vector,
     tanh-bounded to (-1, 1), then expanded to the full vocabulary via a fixed random
     hash mapping (vocab_size -> proj_dim, feature-hashing trick).
  3. Add `delta * P_W` to the LM logits and sample — a continuous, real-valued bias per
     token rather than a binary green/red split.

T is trained (see `sir_model.py`) so that semantically similar contexts produce
correlated watermark vectors — this is what lets the signal survive paraphrasing that
preserves meaning, unlike KGW/DiPmark/SynthID whose green/red assignment depends on
exact token identity and breaks whenever the token sequence changes.

Detection scores the mean watermark value received by the actual tokens and tests it
against the null (mean 0, since T is trained to be zero-mean/balanced) with a one-sample
z-test — the paper itself just thresholds the mean at an empirically calibrated FPR; the
z-test here is this codebase's standard z_score()/detect() convention layered on top.
"""

from __future__ import annotations
import hashlib
import math
import statistics
import torch
from backbone.model import LLaMABackbone, require_local_tokenizer
from typing import List
from peccavi.constants import SECRET_KEY, Z_DETECTION_THRESHOLD
from peccavi.sir_model import CBertEmbedder, DEFAULT_EMBEDDING_MODEL, load_or_train_transform_model, vocab_mapping


class SIRAuctor:
    """
    SIR watermarked generation via a semantic-embedding-conditioned bias, not a
    context-hash green list. `gamma`/`entropy_threshold` are accepted only for
    call-site compatibility with older configs — real SIR has neither parameter.
    """

    def __init__(self, backbone: LLaMABackbone, delta: float = 2.0,
                 secret_key: str = SECRET_KEY,
                 embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                 checkpoint_path: str = "results/sir_transform_model.pt",
                 proj_dim: int = 1000,
                 embed_device: str = "cpu",
                 gamma: float = None, entropy_threshold: float = None):
        self.backbone = backbone
        self.delta = delta
        self.secret_key = secret_key
        self.embed_device = embed_device
        self.embedder = CBertEmbedder(embedding_model, device=embed_device)
        self.checkpoint_path = checkpoint_path
        self.proj_dim = proj_dim

        self._transform_model = None
        self._k2 = 1.0
        self._mapping_t = None

    def _ensure_model(self):
        if self._transform_model is None:
            self._transform_model, self._k2 = load_or_train_transform_model(
                checkpoint_path=self.checkpoint_path,
                device=self.embed_device,
                proj_dim=self.proj_dim,
            )

    def _ensure_mapping(self, vocab_size: int):
        if self._mapping_t is None:
            seed = int(hashlib.sha256(self.secret_key.encode()).hexdigest()[:8], 16)
            mapping = vocab_mapping(vocab_size, self.proj_dim, seed)
            self._mapping_t = torch.as_tensor(mapping, dtype=torch.long)

    def _watermark_vector(self, context_text: str, vocab_size: int) -> torch.Tensor:
        """Returns a length-`vocab_size` tensor of per-token watermark bias in (-1, 1)."""
        self._ensure_model()
        self._ensure_mapping(vocab_size)
        e = self.embedder.embed(context_text).to(self.embed_device)
        with torch.no_grad():
            raw = self._transform_model(e.unsqueeze(0)).squeeze(0)
            compressed = torch.tanh(self._k2 * raw)
        return compressed[self._mapping_t]

    # ------------------------------------------------------------------ #
    #  Generation                                                          #
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        require_local_tokenizer(self.backbone, "SIRAuctor.generate")

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
            context_ids = prompt_ids + generated_ids
            context_text = tokenizer.decode(context_ids, skip_special_tokens=True)

            inputs = tokenizer(
                context_text, return_tensors="pt", truncation=True, max_length=2048
            ).to(self.backbone.model.device)

            with torch.no_grad():
                outputs = self.backbone.model(**inputs)
                logits = outputs.logits[:, -1, :].squeeze(0)

            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            p_w = self._watermark_vector(context_text, vocab_size).to(logits.device)
            biased_logits = logits + self.delta * p_w

            probs = torch.softmax(biased_logits, dim=0).clamp(min=0.0)
            prob_sum = probs.sum()
            if prob_sum <= 0 or not torch.isfinite(prob_sum):
                new_token = int(torch.argmax(biased_logits).item())
            else:
                probs = probs / prob_sum
                new_token = int(torch.multinomial(probs, 1).item())

            if eos_token_id is not None and new_token == eos_token_id:
                break
            generated_ids.append(new_token)

        return tokenizer.decode(generated_ids, skip_special_tokens=True) if generated_ids else ""

    # ------------------------------------------------------------------ #
    #  Detection                                                           #
    # ------------------------------------------------------------------ #

    def _per_token_scores(self, text: str) -> List[float]:
        require_local_tokenizer(self.backbone, "SIRAuctor scoring")
        tokenizer = self.backbone.tokenizer
        token_ids = tokenizer.encode(text)
        vocab_size = tokenizer.vocab_size or len(tokenizer)

        scores = []
        for i, tid in enumerate(token_ids):
            context_ids = token_ids[:i]
            context_text = tokenizer.decode(context_ids, skip_special_tokens=True) if context_ids else ""
            p_w = self._watermark_vector(context_text, vocab_size)
            scores.append(float(p_w[tid].item()))
        return scores

    def mean_score(self, text: str) -> float:
        """S(x) = mean_t P_W(t) over generated tokens — the paper's raw detection statistic."""
        scores = self._per_token_scores(text)
        return statistics.mean(scores) if scores else 0.0

    def z_score(self, text: str) -> float:
        """
        One-sample z-test of the per-token watermark values against null mean 0
        (T is trained to be zero-mean/balanced, so unwatermarked text should average ~0).
        """
        scores = self._per_token_scores(text)
        n = len(scores)
        if n < 2:
            return 0.0
        mean_v = statistics.mean(scores)
        std_v = statistics.pstdev(scores)
        if std_v == 0:
            return 0.0
        return mean_v * math.sqrt(n) / std_v

    def detect(self, text: str, z_threshold: float = Z_DETECTION_THRESHOLD) -> dict:
        z = self.z_score(text)
        return {
            "z_score": round(z, 4),
            "mean_score": round(self.mean_score(text), 4),
            "is_watermarked": z >= z_threshold,
            "z_threshold": z_threshold,
        }
