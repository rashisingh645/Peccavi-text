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
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, output_dim: int = 1000):
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


def similarity_loss(raw_out: torch.Tensor, embeddings: torch.Tensor, k1: float = 1.0) -> torch.Tensor:
    """
    Pulls cosine-similarity of watermark outputs T(e_i), T(e_j) toward tanh(k1 * cosine
    similarity of the underlying embeddings e_i, e_j) for every pair in the batch. This is
    what gives the watermark its semantic-invariance property: paraphrases with similar
    meaning get correlated (not identical, but detectably aligned) watermark vectors.
    """
    t_norm = F.normalize(raw_out, dim=-1)
    e_norm = F.normalize(embeddings, dim=-1)
    sim_out = t_norm @ t_norm.T
    sim_emb = e_norm @ e_norm.T
    target = torch.tanh(k1 * sim_emb)
    return (sim_out - target).abs().mean()


def normalization_loss(raw_out: torch.Tensor, R: float = 1.0, lam1: float = 1.0) -> torch.Tensor:
    """
    Keeps T's raw output zero-mean per example and per output dimension (so an
    unwatermarked text's expected score is 0, matching the paper's null hypothesis),
    plus an anti-collapse term pulling |raw_out| toward a target magnitude R so the
    network can't trivially satisfy the balance terms by outputting all zeros.
    """
    per_example_balance = raw_out.sum(dim=1).abs().mean()
    per_dim_balance = raw_out.sum(dim=0).abs().mean()
    anti_collapse = lam1 * (R - raw_out.abs()).abs().mean()
    return per_example_balance + per_dim_balance + anti_collapse


def sir_loss(raw_out: torch.Tensor, embeddings: torch.Tensor,
             k1: float = 1.0, R: float = 1.0, lam1: float = 1.0, lam2: float = 1.0) -> torch.Tensor:
    return similarity_loss(raw_out, embeddings, k1) + lam2 * normalization_loss(raw_out, R, lam1)


def vocab_mapping(vocab_size: int, proj_dim: int, seed: int) -> np.ndarray:
    """Fixed random hash: vocab token id -> one of `proj_dim` learned output slots."""
    rng = np.random.default_rng(seed % (2 ** 32))
    return rng.integers(0, proj_dim, size=vocab_size)


def train_transform_model(
    corpus_path: str = "datasets/arxiv_5000.csv",
    text_column: str = "experiment",
    checkpoint_path: str = "results/sir_transform_model.pt",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    proj_dim: int = 1000,
    hidden_dim: int = 512,
    k1: float = 1.0,
    k2: float = 1.0,
    R: float = 1.0,
    lam1: float = 1.0,
    lam2: float = 1.0,
    batch_size: int = 32,
    epochs: int = 20,
    lr: float = 1e-3,
    max_examples: int = 2000,
    seed: int = 42,
    device: str = "cpu",
) -> TransformModel:
    """
    Trains T on this codebase's existing datasets/arxiv_5000.csv corpus, mirroring the
    official repo's two-step generate_embeddings.py + train_watermark_model.py pipeline
    but folded into one lazy call. Embeddings are computed once per run; the MLP itself
    trains fast (small network, in-memory vectors). Not the paper's original checkpoint
    or exact training corpus — a faithful reproduction of the described objective on
    locally available text.
    """
    import csv

    texts: List[str] = []
    with open(corpus_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = (row.get(text_column) or "").strip()
            if text:
                texts.append(text)
    texts = texts[:max_examples]
    if len(texts) < batch_size:
        raise ValueError(f"Need at least {batch_size} training texts, got {len(texts)}")

    embedder = CBertEmbedder(embedding_model, device=device)
    logger.info(f"SIR: embedding {len(texts)} training texts with {embedding_model} (one-time cost)...")
    embeddings = embedder.embed_batch(texts).to(device)  # [N, input_dim]

    torch.manual_seed(seed)
    model = TransformModel(input_dim=embeddings.shape[1], hidden_dim=hidden_dim, output_dim=proj_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = embeddings.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) < 2:
                continue
            batch_e = embeddings[idx]
            raw_out = model(batch_e)
            loss = sir_loss(raw_out, batch_e, k1=k1, R=R, lam1=lam1, lam2=lam2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
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

    k2 = train_kwargs.pop("k2", 1.0)
    model = train_transform_model(checkpoint_path=checkpoint_path, device=device, k2=k2, **train_kwargs)
    return model, float(k2)
