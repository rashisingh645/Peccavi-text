"""
backbone/model.py
Shared backbone for PECCAVI.
Supports local Transformers models and API-based models (OpenAI, Anthropic).
"""

from __future__ import annotations
import logging
import os
import sys
from typing import Optional

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

try:
    import torch
except ImportError:
    raise ImportError("\n[PECCAVI] torch not found.\nRun: pip install torch\n")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    raise ImportError("\n[PECCAVI] transformers not found.\nRun: pip install transformers accelerate\n")

def require_local_tokenizer(backbone, feature_name: str) -> None:
    """
    Raise a clear error if `backbone` lacks local next-token logit access.

    Token-level watermarking/detection (Auctor, KGW/SIR/DiPmark/SynthID, Custos) needs a
    local tokenizer + model forward pass — only the `transformers` backend provides that.
    API-only backends (openai/anthropic/deepseek) have neither: without this check, callers
    silently fell back to plain unwatermarked generation or whitespace-split scoring, which
    produces results that look like a real experiment but don't measure anything for that
    backend. Fail loudly instead so a `--baselines` run reports a clear per-model error
    (caught by eval/benchmarks.py's existing try/except) rather than a fake near-chance AUC.
    """
    if not hasattr(backbone, "tokenizer"):
        raise RuntimeError(
            f"{feature_name} requires local next-token logit access (backend='transformers'). "
            f"backend='{getattr(backbone, 'backend', '?')}' has no local tokenizer/model — "
            f"token-level watermarking/detection cannot run against it without a "
            f"logprobs-based implementation, which does not exist for this backend."
        )


try:
    from transformers import BitsAndBytesConfig
    import bitsandbytes  # noqa: F401
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

logger = logging.getLogger(__name__)

class LLaMABackbone:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        backend: str = "transformers",  # "transformers", "openai", or "anthropic"
        device: str = "auto",
        load_in_4bit: bool = True,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.backend = backend
        self.device = device
        if api_key:
            self.api_key = api_key
        elif backend == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
        elif backend == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        elif backend == "deepseek":
            self.api_key = os.getenv("DEEPSEEK_API_KEY")
        else:
            self.api_key = os.getenv("API_KEY")

        if backend == "transformers":
            # Existing local model loading logic
            print(f"[PECCAVI] Loading tokenizer: {model_name}", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.vocab_size = self.tokenizer.vocab_size

            load_kwargs: dict = {"device_map": device, "trust_remote_code": True}
            if load_in_4bit and BNB_AVAILABLE:
                print("[PECCAVI] Using 4-bit quantization", flush=True)
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                print("[PECCAVI] bitsandbytes unavailable — using fp16", flush=True)
                load_kwargs["dtype"] = torch.float16

            print("[PECCAVI] Loading model weights...", flush=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

            self.model.eval()
            print("[PECCAVI] Backbone ready!", flush=True)

        elif backend == "openai":
            if openai is None:
                raise ImportError("openai not installed. Run: pip install openai")
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
            self.vocab_size = 100000  # Approximate for token counting
            print("[PECCAVI] OpenAI backend ready!", flush=True)

        elif backend == "anthropic":
            if anthropic is None:
                raise ImportError("anthropic not installed. Run: pip install anthropic")
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.vocab_size = 100000  # Approximate
            print("[PECCAVI] Anthropic backend ready!", flush=True)

        else:
            raise ValueError(f"Unsupported backend: {backend}")

        self._initialized = True

    def generate(self, prompt, max_new_tokens=256, temperature=0.7,
                 top_p=0.9, do_sample=True, return_logits=False):
        """Generate text. torch.no_grad() only applies to transformers backend."""
        if self.backend == "transformers":
            return self._generate_transformers(prompt, max_new_tokens, temperature, top_p, do_sample, return_logits)
        elif self.backend == "openai":
            return self._generate_openai(prompt, max_new_tokens, temperature, top_p)
        elif self.backend == "anthropic":
            return self._generate_anthropic(prompt, max_new_tokens, temperature, top_p)

    @torch.no_grad()
    def _generate_transformers(self, prompt, max_new_tokens, temperature, top_p, do_sample, return_logits):
        """Generate text using local transformers model."""
        from transformers import LogitsProcessor, LogitsProcessorList

        class _NanInfClamp(LogitsProcessor):
            def __call__(self, input_ids, scores):
                return torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)

        # Handle chat models
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, do_sample=do_sample, output_scores=return_logits,
            return_dict_in_generate=True,
            logits_processor=LogitsProcessorList([_NanInfClamp()]),
        )
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        result = {"text": text}
        if return_logits and hasattr(outputs, "scores"):
            result["scores"] = outputs.scores
        return result

    @torch.no_grad()
    def token_distribution(self, context: str):
        #this returns the softmax distribution over the next token given the context. This is used for tournament sampling.
        """Get the token distribution for the next token given the context.this is what the tournament sampling is  based on."""
        if self.backend != "transformers":
            raise NotImplementedError(f"token_distribution not supported for backend: {self.backend}")
        
        # Handle chat models
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": context}]
            context = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(context, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Last token logits
            probs = torch.softmax(logits, dim=-1)
        return probs.squeeze(0).cpu()  # Return as 1D tensor on CPU

    def _generate_openai(self, prompt, max_new_tokens, temperature, top_p):
        """Generate text using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model_name,  # e.g., "gpt-4"
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        text = response.choices[0].message.content
        return {"text": text}

    def _generate_anthropic(self, prompt, max_new_tokens, temperature, top_p):
        """Generate text using Anthropic API."""
        response = self.client.messages.create(
            model=self.model_name,  # e.g., "claude-3-sonnet-20240229"
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        return {"text": text}

    # Add a simple tokenizer simulation for API backends
    def encode(self, text: str) -> list:
        if self.backend == "transformers":
            return self.tokenizer.encode(text)
        else:
            # Approximate token count (use tiktoken if installed for accuracy)
            return text.split()  # Fallback: word-based

    def decode(self, ids: list) -> str:
        if self.backend == "transformers":
            return self.tokenizer.decode(ids, skip_special_tokens=True)
        else:
            return " ".join(ids)  # Fallback