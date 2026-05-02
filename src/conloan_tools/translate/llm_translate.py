"""LLM-backed translator using instruction-tuned decoder-only models."""

from __future__ import annotations

import click

SUPPORTED_MODELS = {
    "mistralai/Mistral-Nemo-Instruct-2407",
    "Unbabel/TowerInstruct-13B-v0.2",
    "Unbabel/TowerInstruct-7B-v0.2",
}

DEFAULT_LLM_MODEL = "mistralai/Mistral-Nemo-Instruct-2407"

# Used by Mistral-style chat models.
_SYSTEM_PROMPT_TEMPLATE = (
    "You are a professional translator from {src} to {tgt}. "
    "Translate with strict word-for-word fidelity, preserving the original "
    "sentence structure as closely as {tgt} grammar allows. "
    "Do not paraphrase, summarize, or interpret. "
    "Preserve adjectives, physical descriptors, and literary tone exactly as written. "
    "The input may contain XML-like tags such as <CS1>, <NE1>, <L1>, etc. "
    "These tags are structural markers — preserve them exactly as-is, in their "
    "original positions, wrapping the same content. "
    "Do NOT translate, remove, or alter any tag names or their content. "
    "Output ONLY the translated text — no explanations, no quotes, no labels."
)

# Tower-Instruct expects a rigid source/target template — no system prompt.
_TOWER_PROMPT_TEMPLATE = (
    "Translate the following text from {src} into {tgt}.\n"
    "{src}: {text}\n"
    "{tgt}:"
)

_TOWER_MODELS = {
    "Unbabel/TowerInstruct-13B-v0.2",
    "Unbabel/TowerInstruct-7B-v0.2",
}


class LLMTranslator:
    """Decoder-only LLM translator with the same interface as Translator."""

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        *,
        model: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 512,
        quiet: bool = True,
        nllb_src: str | None = None,
        nllb_tgt: str | None = None,
        precision: str = "fp16",
        use_4bit: bool = False,
    ) -> None:
        import torch

        model_id = model or DEFAULT_LLM_MODEL
        if model_id not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported LLM model {model_id!r}. "
                f"Supported: {sorted(SUPPORTED_MODELS)}"
            )

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self._device == "cpu":
            raise ValueError(
                "LLM backend requires a CUDA device. "
                "bitsandbytes 4-bit quantization is not supported on CPU."
            )

        precision_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        self._compute_dtype = precision_map[precision]
        self._use_4bit = use_4bit
        self._is_tower = model_id in _TOWER_MODELS
        self.max_new_tokens = max_new_tokens
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            src=src_lang, tgt=tgt_lang
        )

        self._tokenizer, self._model = self._load(model_id, quiet)

    def translate(self, text: str) -> str:
        if not text:
            return ""
        return self.batch_translate([text])[0]

    def batch_translate(self, texts: list[str]) -> list[str]:
        import torch

        prompts = self._build_prompts(texts)
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = outputs[:, input_len:]
        decoded = self._tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [t.strip() for t in decoded]

    def _build_prompts(self, texts: list[str]) -> list[str]:
        if self._is_tower:
            return [
                _TOWER_PROMPT_TEMPLATE.format(
                    src=self._src_lang, tgt=self._tgt_lang, text=text
                )
                for text in texts
            ]
        prompts = []
        for text in texts:
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": text},
            ]
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
        return prompts

    def _load(self, model_id: str, quiet: bool) -> tuple:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from transformers import logging as tf_logging

        prev_verbosity = tf_logging.get_verbosity()
        if quiet:
            tf_logging.set_verbosity_error()

        click.echo(f"Loading LLM model: {model_id} ({self._device})")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token

            load_kwargs: dict = {"device_map": "auto"}
            if self._use_4bit:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self._compute_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                load_kwargs["torch_dtype"] = self._compute_dtype

            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            model.eval()
        except OSError as exc:
            raise ValueError(
                f"Failed to load {model_id!r} from Hugging Face. "
                f"Check that the model exists and you have network access.\n"
                f"Original error: {exc}"
            ) from exc
        finally:
            if quiet:
                tf_logging.set_verbosity(prev_verbosity)

        return tokenizer, model

