"""
peccavi/sir_model.py
Supporting machinery for SIR (Liu, Pan, Hu, Meng, Wen, ICLR 2024 —
"A Semantic Invariant Robust Watermark for Large Language Models").

Real SIR does not partition the vocabulary via a context-token hash (that is KGW's
mechanism). Instead it:
  1. Embeds the preceding text with a semantic sentence-embedding model
     (the paper uses Compositional-BERT).
  2. Passes that embedding through a small trained network T (4 FC layers, ReLU,
     residual connections) to get a compressed watermark vector, expanded to the full
     vocabulary via a fixed random hash mapping (a feature-hashing trick — many vocab
     tokens share one of `proj_dim` learned output slots).
  3. Trains T with two losses so that (a) semantically similar contexts get correlated
     watermark vectors (this is what survives paraphrasing — the "semantic invariant"
     property) and (b) the output stays zero-mean/balanced per-example and per-dimension
     so the null-hypothesis score is 0 for unwatermarked text.

This module provides the embedder, the T network, the two training losses, and a
lazy train-or-load helper. `peccavi/auctor_sir.py` wires these into a generator/detector.
"""

from __future__ import annotations
import logging
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "perceptiveshawty/compositional-bert-large-uncased"


class CBertEmbedder:
    """
    Mean-pooled sentence embedding via Compositional-BERT (the paper's embedding model).
    Runs on CPU by default, mirroring how Scriba's MarianMT models are kept off the GPU
    to avoid competing with the LLaMA-2-7B backbone for VRAM.
    Tokenizer/model are process-wide singletons so repeated instantiation is cheap.
    """
    _tokenizer = None
    _model = None
    _loaded_name = None

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device

    def _ensure_loaded(self):
        if CBertEmbedder._model is None or CBertEmbedder._loaded_name != self.model_name:
            from transformers import AutoTokenizer, AutoModel
            logger.info(f"Loading SIR embedding model {self.model_name} (one-time)...")
            CBertEmbedder._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            CBertEmbedder._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            CBertEmbedder._model.eval()
            CBertEmbedder._loaded_name = self.model_name

    def embed(self, text: str) -> torch.Tensor:
        self._ensure_loaded()
        text = text if text.strip() else "[EMPTY]"
        inputs = CBertEmbedder._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)
        with torch.no_grad():
            out = CBertEmbedder._model(**inputs)
        token_embeddings = out.last_hidden_state.squeeze(0)                # [T, H]
        mask = inputs["attention_mask"].squeeze(0).unsqueeze(-1).float()   # [T, 1]
        pooled = (token_embeddings * mask).sum(0) / mask.sum().clamp(min=1e-6)
        return pooled                                                      # [H]

    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        return torch.stack([self.embed(t) for t in texts])


class TransformModel(nn.Module):
    """
    The paper's watermark model T: 4 fully-connected layers, ReLU, residual connections
    around the two middle layers. Maps a semantic embedding to a `proj_dim`-length raw
    watermark vector (pre-tanh); the caller applies tanh(k2 * raw) to bound it to (-1, 1).
    output_dim=300 matches the reference repo (github.com/THU-BPM/Robust_Watermark)'s
    actual TransformModel/generate_mappings.py default -- an earlier version of this file
    used 1000, an unrelated guess made before the repo's actual defaults were checked.
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, output_dim: int = 300):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.l4 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        h1 = self.act(self.l1(e))
        h2 = self.act(self.l2(h1)) + h1
        h3 = self.act(self.l3(h2)) + h2
        return self.l4(h3)


def _pairwise_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dot = (x * y).sum(dim=-1)
    return dot / (x.norm(p=2, dim=-1) * y.norm(p=2, dim=-1))


def median_pairwise_cosine(embeddings: torch.Tensor) -> float:
    """
    The reference repo's get_median_value_of_similarity: the median of the full all-pairs
    cosine-similarity matrix over the training embeddings, computed once up front and used
    to re-center the similarity target below. BERT-family sentence embeddings don't cluster
    around cosine similarity 0 (their own runs reported ~0.4), so without this centering the
    tanh target saturates almost everywhere in one direction and the network never learns a
    useful signal -- confirmed as a real gap versus an earlier version of this loss that
    applied tanh(k1 * raw_similarity) directly, with no centering.
    """
    normed = F.normalize(embeddings, dim=-1)
    sim = normed @ normed.T
    return float(torch.median(sim))


def similarity_loss(raw_out_a: torch.Tensor, raw_out_b: torch.Tensor,
                     emb_a: torch.Tensor, emb_b: torch.Tensor,
                     median_value: float, k1: float = 20.0) -> torch.Tensor:
    """
    Pulls cosine-similarity of watermark outputs T(e_a), T(e_b) toward
    tanh(k1 * (cosine_similarity(e_a, e_b) - median_value)) for each paired example in the
    batch -- matching the reference repo's loss_fn exactly (input_a/input_b are two
    independently-shuffled batches, paired element-wise, not an all-pairs matrix). This is
    what gives the watermark its semantic-invariance property: paraphrases with similar
    meaning get correlated (not identical, but detectably aligned) watermark vectors.
    """
    input_sim = _pairwise_cosine_similarity(emb_a, emb_b)
    target = torch.tanh(k1 * (input_sim - median_value))
    output_sim = _pairwise_cosine_similarity(raw_out_a, raw_out_b)
    return (target - output_sim).abs().mean()


def _row_col_mean_penalty(raw_out: torch.Tensor) -> torch.Tensor:
    """Reference repo's row_col_mean_penalty: squared per-example mean (row) plus squared
    per-dimension mean (col), summed over the batch -- keeps T's output zero-mean along
    both axes so an unwatermarked text's expected score is 0 (the paper's null hypothesis)."""
    row = raw_out.mean(dim=1).pow(2).sum()
    col = raw_out.mean(dim=0).pow(2).sum()
    return row + col


def _range_penalty(raw_out: torch.Tensor, floor: float = 0.05) -> torch.Tensor:
    """Reference repo's abs_value_penalty: a one-sided push away from zero for entries
    still inside (-floor, floor), masked so already-saturated entries contribute nothing --
    unlike a two-sided pull toward a fixed magnitude, this only nudges near-zero (i.e.
    under-watermarked) outputs outward, leaving confidently-saturated ones alone."""
    deficit = F.relu(floor - raw_out.abs())
    mask = (deficit > 0).float()
    denom = torch.clamp(mask.sum(), min=1.0)
    return (deficit * mask).sum() / denom


def sir_loss(raw_out_a: torch.Tensor, raw_out_b: torch.Tensor,
             emb_a: torch.Tensor, emb_b: torch.Tensor, median_value: float,
             k1: float = 20.0, lam1: float = 0.1, lam2: float = 1.0) -> torch.Tensor:
    """
    Matches the reference repo's loss_fn total: original_loss + lambda1*mean_penalty +
    lambda2*range_penalty, with lam1/lam2 defaults taken directly from their hardcoded
    loss_fn(..., lambda1=0.1, lambda2=1, ...) values (train_watermark_model.py never
    overrides them from the CLI). An earlier version of this module used lam1=10, lam2=0.1
    based on a reading of the paper's prose in Section 4.3 -- since those values conflict
    with the actual repo's code, the code's own operative defaults take precedence here.
    """
    sim_loss = similarity_loss(raw_out_a, raw_out_b, emb_a, emb_b, median_value, k1)
    mean_penalty = _row_col_mean_penalty(raw_out_a) + _row_col_mean_penalty(raw_out_b)
    range_penalty = _range_penalty(raw_out_a) + _range_penalty(raw_out_b)
    return sim_loss + lam1 * mean_penalty + lam2 * range_penalty


def vocab_mapping(vocab_size: int, proj_dim: int, seed: int) -> np.ndarray:
    """Fixed random hash: vocab token id -> one of `proj_dim` learned output slots."""
    rng = np.random.default_rng(seed % (2 ** 32))
    return rng.integers(0, proj_dim, size=vocab_size)


def train_transform_model(
    corpus_path: str = "datasets/arxiv_5000.csv",
    text_column: str = "experiment",
    texts: List[str] | None = None,
    checkpoint_path: str = "results/sir_transform_model.pt",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    proj_dim: int = 300,
    hidden_dim: int = 512,
    # k1/k2 confirmed against the reference repo's actual hardcoded constants: k1=20 is the
    # "20" inside train_watermark_model.py's loss_fn's tanh(20*(sim - median)); k2=1000 is
    # the "1000" inside watermark.py's scale_vector's tanh(1000*v_minus_mean) (applied at
    # generation/detection time, not here).
    #
    # lam1/lam2/optimizer reverted back to the paper's own explicit statement after directly
    # quoting it (Section 6.1): "Hyperparameters are set to k1=20, k2=1000, lambda1=10,
    # lambda2=0.1, and the Adam optimizer (lr=1e-5) is used for training." A previous version
    # of this file switched to the repo's own train_watermark_model.py loss_fn/optimizer
    # defaults (lambda1=0.1, lambda2=1, SGD lr=0.006/weight_decay=0.2/StepLR/epochs=2000) on
    # the reasoning that "the repo's code should take precedence over a reading of the
    # paper's prose" -- but this isn't inferred from surrounding prose, it's the paper's own
    # unambiguous numeric statement, directly contradicting the repo's unrelated CLI-script
    # defaults. The paper's explicit statement wins here.
    k1: float = 20.0,
    k2: float = 1000.0,
    lam1: float = 10.0,
    lam2: float = 0.1,
    batch_size: int = 32,
    epochs: int = 200,
    lr: float = 1e-5,
    max_examples: int = 2000,
    seed: int = 42,
    device: str = "cpu",
) -> TransformModel:
    """
    Trains T, mirroring the official repo's two-step generate_embeddings.py +
    train_watermark_model.py pipeline but folded into one lazy call. Embeddings are
    computed once per run; the MLP itself trains fast (small network, in-memory
    vectors). Pass `texts` directly to train on a specific corpus (e.g. Praeco's
    stratified multi-domain prompt pool, matching what SIRAuctor actually does);
    otherwise falls back to reading `corpus_path`'s single CSV. Not the paper's
    original checkpoint or exact training corpus (they use WikiText-103) — a
    faithful reproduction of the described training objective on available text.
    """
    if texts is None:
        import csv
        texts = []
        with open(corpus_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                text = (row.get(text_column) or "").strip()
                if text:
                    texts.append(text)
    texts = texts[:max_examples]
    if len(texts) < 2 * batch_size:
        raise ValueError(f"Need at least {2 * batch_size} training texts (two paired batches), got {len(texts)}")

    embedder = CBertEmbedder(embedding_model, device=device)
    logger.info(f"SIR: embedding {len(texts)} training texts with {embedding_model} (one-time cost)...")
    embeddings = embedder.embed_batch(texts).to(device)  # [N, input_dim]

    median_value = median_pairwise_cosine(embeddings)
    logger.info(f"SIR: median pairwise cosine similarity of training embeddings = {median_value:.4f}")

    torch.manual_seed(seed)
    model = TransformModel(input_dim=embeddings.shape[1], hidden_dim=hidden_dim, output_dim=proj_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = embeddings.shape[0]
    n_pair_batches = n // (2 * batch_size)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        for b in range(n_pair_batches):
            idx_a = perm[2 * b * batch_size:(2 * b + 1) * batch_size]
            idx_b = perm[(2 * b + 1) * batch_size:(2 * b + 2) * batch_size]
            emb_a, emb_b = embeddings[idx_a], embeddings[idx_b]
            raw_a, raw_b = model(emb_a), model(emb_b)
            loss = sir_loss(raw_a, raw_b, emb_a, emb_b, median_value, k1=k1, lam1=lam1, lam2=lam2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            logger.info(f"SIR transform model epoch {epoch + 1}/{epochs} loss={total_loss / max(n_batches, 1):.4f}")

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": embeddings.shape[1],
        "hidden_dim": hidden_dim,
        "proj_dim": proj_dim,
        "embedding_model": embedding_model,
        "k2": k2,
    }, checkpoint_path)
    logger.info(f"SIR transform model saved -> {checkpoint_path}")
    model.eval()
    return model


def load_or_train_transform_model(
    checkpoint_path: str = "results/sir_transform_model.pt",
    device: str = "cpu",
    **train_kwargs,
) -> Tuple[TransformModel, float]:
    """Loads a cached checkpoint if present; otherwise trains and caches one."""
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model = TransformModel(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["proj_dim"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, float(ckpt.get("k2", 1.0))

    # Must match train_transform_model's own k2 default (1000.0, the paper's saturating
    # value) -- an earlier version of this default (1.0) silently overrode that every time
    # this function trained a fresh model without an explicit k2 kwarg (SIRAuctor's actual
    # call path), reintroducing the exact unsaturated-signal problem that default was
    # supposed to fix.
    k2 = train_kwargs.pop("k2", 1000.0)
    model = train_transform_model(checkpoint_path=checkpoint_path, device=device, k2=k2, **train_kwargs)
    return model, float(k2)
