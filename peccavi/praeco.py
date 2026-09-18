"""
peccavi/praeco.py
Agent: Praeco Dynamic Prompting Orchestrator.
Manages prompt construction each PECCAVI generation round.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import random

DATASET_COLUMN_NAMES = ("experiment", "experiments")
DEFAULT_DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"

PROMPT_BANK = [
    "Summarize recent AI safety research.",
    "Explain the importance of watermarking in AI-generated content.",
    "Describe the risks of large language models in misinformation.",
    "Write a short paragraph about responsible AI deployment.",
    "Discuss how adversarial robustness improves AI reliability.",
    "Explain what content authenticity means in the context of generative AI.",
    "Describe how reinforcement learning is used in AI alignment.",
]


class Praeco:
    """
    Samples prompts stratified by source dataset: each call first picks a source
    uniformly at random, then a prompt uniformly within that source. This gives every
    dataset equal representation regardless of how many non-empty rows it happened to
    contribute — e.g. c4_multilingual_5000.csv yields only ~833 usable prompts (most
    rows have an empty `experiment` field) against ~5000 each from the other three
    datasets, so sampling uniformly over the pooled list would draw from it roughly
    6x less often than intended. Stratifying by source fixes that, and matters for the
    content-adaptive claim specifically: Figure 2's entropy quartiles should reflect
    all four content domains (Reddit/arctic, arxiv abstracts, multilingual web, literary
    Gutenberg chunks), not be dominated by whichever dataset happened to have the most
    non-empty rows.
    """

    def __init__(self, custom_prompts: List[str] | None = None, dataset_dir: str | Path | None = None):
        self.pools_by_source: Dict[str, List[str]] = {}
        if custom_prompts is not None:
            self.prompts = custom_prompts
            self.prompt_sources = {p: "custom" for p in custom_prompts}
            self.pools_by_source["custom"] = list(custom_prompts)
        else:
            pairs = self.load_prompts_from_dataset(dataset_dir=dataset_dir)
            if pairs:
                self.prompts = [p for p, _ in pairs]
                self.prompt_sources = {p: src for p, src in pairs}
                for p, src in pairs:
                    self.pools_by_source.setdefault(src, []).append(p)
            else:
                self.prompts = PROMPT_BANK
                self.prompt_sources = {p: "builtin" for p in PROMPT_BANK}
                self.pools_by_source["builtin"] = list(PROMPT_BANK)

        self.sources = sorted(self.pools_by_source.keys())

    @staticmethod
    def load_prompts_from_dataset(dataset_dir: str | Path | None = None) -> List[Tuple[str, str]]:
        """Returns list of (prompt_text, dataset_name) pairs."""
        dataset_path = Path(dataset_dir) if dataset_dir else DEFAULT_DATASETS_DIR
        if not dataset_path.exists() or not dataset_path.is_dir():
            return []

        pairs: List[Tuple[str, str]] = []
        for csv_file in sorted(dataset_path.glob("*.csv")):
            dataset_name = csv_file.stem
            try:
                with csv_file.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if not reader.fieldnames:
                        continue

                    normalized = {name.strip().lower(): name for name in reader.fieldnames}
                    prompt_key = next(
                        (original for canonical, original in normalized.items() if canonical in DATASET_COLUMN_NAMES),
                        None,
                    )
                    if not prompt_key:
                        continue

                    for row in reader:
                        prompt_text = row.get(prompt_key, "")
                        if prompt_text is None:
                            continue
                        prompt_text = prompt_text.strip()
                        if prompt_text:
                            pairs.append((prompt_text, dataset_name))
            except Exception:
                continue

        return pairs

    def get_source(self, prompt: str) -> str:
        return self.prompt_sources.get(prompt, "unknown")

    def next_prompt(self) -> str:
        """Pick a source uniformly at random, then a prompt uniformly within it."""
        source = random.choice(self.sources)
        return random.choice(self.pools_by_source[source])

    def batch_prompts(self, n: int) -> List[str]:
        """
        Split n as evenly as possible across sources (deterministic balance, unlike
        next_prompt()'s per-call random source pick — matters more here since eval
        batches are meant to represent all content domains, not just converge to
        balance in expectation over many calls), sampling with replacement within
        each source.
        """
        if not self.sources:
            return []
        base, remainder = divmod(n, len(self.sources))
        counts = [base + (1 if i < remainder else 0) for i in range(len(self.sources))]
        random.shuffle(counts)  # avoid always giving the remainder to the same (sorted-first) source

        batch: List[str] = []
        for source, count in zip(self.sources, counts):
            pool = self.pools_by_source[source]
            batch.extend(random.choices(pool, k=count))
        random.shuffle(batch)
        return batch