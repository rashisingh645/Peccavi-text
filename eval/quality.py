"""
eval/quality.py
Text quality scoring: GPT-4o-mini judge (primary) with Flesch fallback.
GPT-4 judge is significantly stronger than Flesch for EMNLP evaluation.
"""

from __future__ import annotations
import os
import logging
import textstat

logger = logging.getLogger(__name__)

_gpt4_client = None


def _get_client():
    global _gpt4_client
    if _gpt4_client is not None:
        return _gpt4_client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        _gpt4_client = OpenAI(api_key=api_key)
        return _gpt4_client
    except ImportError:
        logger.warning("openai package not installed — falling back to Flesch score")
        return None


def flesch_quality_score(text: str) -> float:
    """Maps Flesch Reading Ease to a 1-5 scale, clamped since FRE can fall
    outside its nominal 0-100 range (very hard or very easy text)."""
    fre = textstat.flesch_reading_ease(text)
    return round(min(max(1 + (fre / 100) * 4, 1.0), 5.0), 2)


def gpt4_quality_score(text: str, prompt: str = None) -> float:
    """
    Rate text quality 1–5 using GPT-4o-mini as judge.
    Criteria: coherence, fluency, relevance to prompt, informativeness.
    Falls back to Flesch if OpenAI key unavailable.
    """
    client = _get_client()
    if client is None:
        return flesch_quality_score(text)

    system = (
        "You are a strict text quality evaluator for NLP research. "
        "Rate the text on a scale of 1 to 5 based on coherence, fluency, "
        "and informativeness. Respond with only a single integer (1, 2, 3, 4, or 5)."
    )
    user_parts = []
    if prompt:
        user_parts.append(f"Prompt: {prompt[:200]}")
    user_parts.append(f"Text: {text[:400]}")
    user_msg = "\n".join(user_parts)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=3,
            temperature=0,
        )
        score = float(response.choices[0].message.content.strip())
        return round(min(max(score, 1.0), 5.0), 2)
    except Exception as e:
        logger.warning(f"GPT-4 quality judge failed: {e} — falling back to Flesch")
        return flesch_quality_score(text)


def quality_score(text: str, prompt: str = None) -> float:
    """Primary quality function: GPT-4 if available, else Flesch."""
    return gpt4_quality_score(text, prompt=prompt)


_ppl_model = None
_ppl_tokenizer = None


def _get_ppl_model():
    global _ppl_model, _ppl_tokenizer
    if _ppl_model is not None:
        return _ppl_model, _ppl_tokenizer
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        _ppl_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        _ppl_model = GPT2LMHeadModel.from_pretrained("gpt2")
        _ppl_model.eval()
        return _ppl_model, _ppl_tokenizer
    except Exception as e:
        logger.warning(f"GPT-2 PPL model unavailable: {e}")
        return None, None


def perplexity(text: str) -> float:
    """
    Compute GPT-2 perplexity of text. Lower = more natural.
    Used to measure fluency cost of watermarking (PPL_watermarked / PPL_baseline).
    """
    model, tokenizer = _get_ppl_model()
    if model is None:
        return float("nan")
    try:
        import torch
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
        return round(float(torch.exp(loss)), 4)
    except Exception as e:
        logger.warning(f"PPL computation failed: {e}")
        return float("nan")
