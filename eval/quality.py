"""
eval/quality.py
Text quality scoring: perplexity under a fixed oracle model, plus a cheap Flesch
readability score for reporting.

No baseline paper in this literature (KGW, SIR, DiPmark, both RL-watermarking papers)
uses an LLM-as-judge for quality -- all of them use perplexity as the sole or primary
quality metric, computed under a dedicated oracle model rather than the generating model
itself (which watermarking could trivially "improve" perplexity on without saying
anything about actual fluency). This codebase previously had a GPT-4o-mini judge here,
but OPENAI_API_KEY was never set for any run this session, so every "gpt4_quality" value
ever produced was silently the Flesch fallback -- the GPT-judge path was removed rather
than fixed, since the literature doesn't use one anyway.
"""

from __future__ import annotations
import logging
import textstat

logger = logging.getLogger(__name__)


def flesch_quality_score(text: str) -> float:
    """Maps Flesch Reading Ease to a 1-5 scale, clamped since FRE can fall
    outside its nominal 0-100 range (very hard or very easy text). Not a metric
    any baseline paper reports -- kept as a cheap secondary readability signal
    alongside `perplexity()`, which is the literature-comparable one."""
    fre = textstat.flesch_reading_ease(text)
    return round(min(max(1 + (fre / 100) * 4, 1.0), 5.0), 2)


# Larger, same-family oracle model for perplexity -- the actual literature convention:
# KGW judges its OPT-1.3B generations with an OPT-2.7B oracle; SIR judges its LLaMA-7B
# generations with a LLaMA-13B oracle. A watermarked model's perplexity under itself, or
# under an unrelated small model like GPT-2, isn't what either paper reports. Update this
# if the generation backbone changes to a different model family.
ORACLE_MODEL_NAME = "meta-llama/Llama-2-13b-hf"

_ppl_backbone = None


def _get_ppl_backbone():
    global _ppl_backbone
    if _ppl_backbone is not None:
        return _ppl_backbone
    try:
        from backbone.model import LLaMABackbone
        _ppl_backbone = LLaMABackbone(model_name=ORACLE_MODEL_NAME, load_in_4bit=True)
        return _ppl_backbone
    except Exception as e:
        logger.warning(f"Oracle PPL model ({ORACLE_MODEL_NAME}) unavailable: {e}")
        return None


def perplexity(text: str) -> float:
    """
    Perplexity of `text` under the fixed oracle model (see ORACLE_MODEL_NAME), following
    the KGW/SIR convention of a larger same-family judge model. Lower = more natural.
    Used to compute PPL_watermarked / PPL_baseline (should be close to 1.0).
    """
    backbone = _get_ppl_backbone()
    if backbone is None:
        return float("nan")
    try:
        import torch
        inputs = backbone.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(backbone.model.device)
        with torch.no_grad():
            loss = backbone.model(**inputs, labels=inputs["input_ids"]).loss
        return round(float(torch.exp(loss)), 4)
    except Exception as e:
        logger.warning(f"PPL computation failed: {e}")
        return float("nan")
