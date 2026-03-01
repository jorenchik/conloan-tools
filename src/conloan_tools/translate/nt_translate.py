from __future__ import annotations

import re
import os
from enum import Enum

import click

from conloan_tools.resources import load_known_languages


class ModelFamily(str, Enum):
    OPUS = "opus"
    NLLB = "nllb"
    M2M100 = "m2m100"


class ModelSize(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


# ISO code -> NLLB flores-200 code
NLLB_LANG_CODES: dict[str, str] = {
    "ar": "arb_Arab", "bg": "bul_Cyrl", "cs": "ces_Latn", "da": "dan_Latn",
    "de": "deu_Latn", "en": "eng_Latn", "es": "spa_Latn", "et": "est_Latn",
    "fi": "fin_Latn", "fr": "fra_Latn", "hi": "hin_Deva", "it": "ita_Latn",
    "ja": "jpn_Jpan", "lt": "lit_Latn", "lv": "lvs_Latn", "nl": "nld_Latn",
    "no": "nob_Latn", "pl": "pol_Latn", "pt": "por_Latn", "ro": "ron_Latn",
    "ru": "rus_Cyrl", "sv": "swe_Latn", "tr": "tur_Latn", "uk": "ukr_Cyrl",
    "zh": "zho_Hans",
}

# ISO code -> M2M-100 code (subset; M2M uses plain ISO for most)
M2M100_LANG_CODES: dict[str, str] = {
    "ar": "ar", "bg": "bg", "cs": "cs", "da": "da", "de": "de", "en": "en",
    "es": "es", "et": "et", "fi": "fi", "fr": "fr", "hi": "hi", "it": "it",
    "ja": "ja", "lt": "lt", "lv": "lv", "nl": "nl", "no": "no", "pl": "pl",
    "pt": "pt", "ro": "ro", "ru": "ru", "sv": "sv", "tr": "tr", "uk": "uk",
    "zh": "zh",
}

# (distilled, size) -> HF model id
_NLLB_MODELS: dict[tuple[bool, ModelSize], str] = {
    (True, ModelSize.SMALL): "facebook/nllb-200-distilled-600M",
    (True, ModelSize.MEDIUM): "facebook/nllb-200-distilled-1.3B",
    (False, ModelSize.MEDIUM): "facebook/nllb-200-1.3B",
    (False, ModelSize.LARGE): "facebook/nllb-200-3.3B",
}

_M2M100_MODELS: dict[ModelSize, str] = {
    ModelSize.SMALL: "facebook/m2m100_418M",
    ModelSize.MEDIUM: "facebook/m2m100_1.2B",
}

# Known Opus-MT "big" variants (tc-big). Extend as needed.
_OPUS_BIG_OVERRIDES: dict[tuple[str, str], str] = {
    ("en", "de"): "Helsinki-NLP/opus-mt-tc-big-en-de",
    ("de", "en"): "Helsinki-NLP/opus-mt-tc-big-de-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-tc-big-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-tc-big-fr-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-tc-big-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-tc-big-es-en",
    ("en", "pt"): "Helsinki-NLP/opus-mt-tc-big-en-pt",
    ("pt", "en"): "Helsinki-NLP/opus-mt-tc-big-pt-en",
    ("en", "ru"): "Helsinki-NLP/opus-mt-tc-big-en-ru",
    ("ru", "en"): "Helsinki-NLP/opus-mt-tc-big-ru-en",
    ("en", "fi"): "Helsinki-NLP/opus-mt-tc-big-en-fi",
    ("fi", "en"): "Helsinki-NLP/opus-mt-tc-big-fi-en",
    ("en", "sv"): "Helsinki-NLP/opus-mt-tc-big-en-sv",
    ("sv", "en"): "Helsinki-NLP/opus-mt-tc-big-sv-en",
    ("en", "nl"): "Helsinki-NLP/opus-mt-tc-big-en-nl",
    ("nl", "en"): "Helsinki-NLP/opus-mt-tc-big-nl-en",
    ("en", "it"): "Helsinki-NLP/opus-mt-tc-big-en-it",
    ("it", "en"): "Helsinki-NLP/opus-mt-tc-big-it-en",
}

# Sentence-ending punctuation splitter.  Handles .!? followed by whitespace.
# Keeps the delimiter attached to the preceding segment.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def resolve_model_name(
    family: ModelFamily,
    src_code: str | None = None,
    tgt_code: str | None = None,
    *,
    distilled: bool = True,
    size: ModelSize = ModelSize.SMALL,
    prefer_big: bool = False,
) -> str:
    """Build a HF model id from structured parameters."""

    if family == ModelFamily.NLLB:
        key = (distilled, size)
        if key not in _NLLB_MODELS:
            valid = [
                f"distilled={d}, size={s.value}"
                for (d, s) in _NLLB_MODELS
            ]
            raise ValueError(
                f"No NLLB variant for distilled={distilled}, "
                f"size={size.value!r}.\nValid combinations:\n"
                + "\n".join(f"  - {v}" for v in valid)
            )
        return _NLLB_MODELS[key]

    if family == ModelFamily.M2M100:
        if size not in _M2M100_MODELS:
            valid = [s.value for s in _M2M100_MODELS]
            raise ValueError(
                f"No M2M-100 variant for size={size.value!r}. "
                f"Valid: {valid}"
            )
        return _M2M100_MODELS[size]

    if family == ModelFamily.OPUS:
        if not src_code or not tgt_code:
            raise ValueError(
                "Opus-MT requires both src and tgt language codes."
            )
        if prefer_big:
            pair = (src_code, tgt_code)
            if pair in _OPUS_BIG_OVERRIDES:
                return _OPUS_BIG_OVERRIDES[pair]
        return f"Helsinki-NLP/opus-mt-{src_code}-{tgt_code}"

    raise ValueError(f"Unknown model family: {family}")


class Translator:
    """Seq2seq translator with parametrized source/target languages.

    Languages are accepted as either a raw ISO code (e.g. ``"lv"``) or a
    canonical name (e.g. ``"Latvian"``) as returned by
    ``load_known_languages()``.
    """

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        *,
        family: ModelFamily | str = ModelFamily.OPUS,
        distilled: bool = True,
        size: ModelSize | str = ModelSize.SMALL,
        prefer_big: bool = False,
        device: str | None = None,
        max_new_tokens: int = 512,
        quiet: bool = True,
    ) -> None:
        import torch

        self.family = ModelFamily(family)
        self.size = ModelSize(size)
        self.distilled = distilled
        self.prefer_big = prefer_big
        self.max_new_tokens = max_new_tokens
        self.quiet = quiet
        self._device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Resolve canonical names -> ISO codes
        known = load_known_languages()
        name_to_code = {v.lower(): k for k, v in known.items()}

        self._src_code = self._resolve_code(src_lang, name_to_code, known)
        self._tgt_code = self._resolve_code(tgt_lang, name_to_code, known)

        # Pre-resolve family-specific language tokens
        if self.family == ModelFamily.NLLB:
            self._src_nllb = self._to_nllb_code(self._src_code)
            self._tgt_nllb = self._to_nllb_code(self._tgt_code)
        elif self.family == ModelFamily.M2M100:
            self._src_m2m = self._to_m2m_code(self._src_code)
            self._tgt_m2m = self._to_m2m_code(self._tgt_code)

        model_name = resolve_model_name(
            self.family,
            self._src_code,
            self._tgt_code,
            distilled=self.distilled,
            size=self.size,
            prefer_big=self.prefer_big,
        )
        self._tokenizer, self._model = self._load(model_name)

    def translate(self, text: str) -> str:
        """Translate a single string.

        For sentence-level models (Opus-MT) the input is automatically
        split into sentences, each translated independently, and the
        results rejoined.
        """
        if not text:
            return ""

        if self.family == ModelFamily.OPUS:
            sentences = self._split_sentences(text)
            if not sentences:
                return ""
            translated = self.batch_translate(sentences)
            return " ".join(translated)

        return self.batch_translate([text])[0]

    def batch_translate(self, texts: list[str]) -> list[str]:
        """Translate a list of strings (no automatic sentence splitting)."""
        if not texts:
            return []

        import torch

        tok = self._tokenizer

        if self.family == ModelFamily.NLLB:
            tok.src_lang = self._src_nllb
        elif self.family == ModelFamily.M2M100:
            tok.src_lang = self._src_m2m

        inputs = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._device)

        # gen_kwargs: dict = {"max_new_tokens": self.max_new_tokens}
        # Use a slightly more "strict" generation config
        gen_kwargs: dict = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": 4, # Beams help preserve structure better than greedy search
            "length_penalty": 1.0,
        }

        if self.family == ModelFamily.NLLB:
            gen_kwargs["forced_bos_token_id"] = tok.convert_tokens_to_ids(
                self._tgt_nllb
            )
        elif self.family == ModelFamily.M2M100:
            gen_kwargs["forced_bos_token_id"] = tok.get_lang_id(
                self._tgt_m2m
            )

        with torch.no_grad():
            tokens = self._model.generate(**inputs, **gen_kwargs, early_stopping=None)

        return tok.batch_decode(tokens, skip_special_tokens=True)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Naive sentence splitter.  Splits on .!? followed by whitespace."""
        parts = _SENTENCE_RE.split(text.strip())
        return [p for p in parts if p.strip()]

    @staticmethod
    def _resolve_code(
        lang: str,
        name_to_code: dict[str, str],
        code_to_name: dict[str, str],
    ) -> str:
        """Accept either an ISO code or a canonical name."""
        if lang in code_to_name:
            return lang
        key = lang.lower()
        if key in name_to_code:
            return name_to_code[key]
        raise ValueError(
            f"Unknown language {lang!r}. Use a code or canonical "
            f"name from load_known_languages()."
        )

    @staticmethod
    def _to_nllb_code(code: str) -> str:
        if code in NLLB_LANG_CODES:
            return NLLB_LANG_CODES[code]
        raise ValueError(
            f"No NLLB flores code for {code!r}. "
            f"Add it to NLLB_LANG_CODES."
        )

    @staticmethod
    def _to_m2m_code(code: str) -> str:
        if code in M2M100_LANG_CODES:
            return M2M100_LANG_CODES[code]
        raise ValueError(
            f"No M2M-100 code for {code!r}. "
            f"Add it to M2M100_LANG_CODES."
        )

    def _load(self, model_name: str) -> tuple:
        import torch
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            logging as tf_logging,
        )

        prev_verbosity = tf_logging.get_verbosity()
        if self.quiet:
            tf_logging.set_verbosity_error()

        click.echo(f"Loading model: {model_name} ({self._device})")

        try:
            if self.family == ModelFamily.NLLB:
                from transformers import NllbTokenizer
                tokenizer = NllbTokenizer.from_pretrained(model_name, use_fast=True)
            elif self.family == ModelFamily.M2M100:
                from transformers import M2M100Tokenizer
                tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            elif self.family == ModelFamily.OPUS:
                from transformers import MarianTokenizer
                tokenizer = MarianTokenizer.from_pretrained(model_name)
            # else:
            #     tokenizer = AutoTokenizer.from_pretrained(model_name)

            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except OSError as exc:
            raise ValueError(
                f"Failed to load {model_name!r} from Hugging Face. "
                f"Check that the model exists and you have network "
                f"access.\nOriginal error: {exc}"
            ) from exc
        finally:
            if self.quiet:
                tf_logging.set_verbosity(prev_verbosity)

        if self._device == "cuda":
            model = model.half()
        model = model.to(self._device).eval()

        return tokenizer, model


@click.command("interactive")
@click.option(
    "--src",
    required=True,
    help="Source language (code or canonical name)",
)
@click.option(
    "--tgt",
    required=True,
    help="Target language (code or canonical name)",
)
@click.option(
    "--family",
    type=click.Choice([f.value for f in ModelFamily]),
    default=ModelFamily.OPUS.value,
)
@click.option(
    "--size",
    type=click.Choice([s.value for s in ModelSize]),
    default=ModelSize.SMALL.value,
)
@click.option("--distilled/--no-distilled", default=True)
@click.option(
    "--prefer-big",
    is_flag=True,
    default=False,
    help="Prefer tc-big Opus-MT variant when available.",
)
@click.option(
    "--max-new-tokens",
    type=int,
    default=512,
    show_default=True,
    help="Maximum tokens to generate per segment.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Show full HuggingFace/transformers log output.",
)
def interactive(
    src: str,
    tgt: str,
    family: str,
    size: str,
    distilled: bool,
    prefer_big: bool,
    max_new_tokens: int,
    verbose: bool,
) -> None:
    """Interactive REPL for translating text line by line."""
    translator = Translator(
        src_lang=src,
        tgt_lang=tgt,
        family=family,
        distilled=distilled,
        size=size,
        prefer_big=prefer_big,
        max_new_tokens=max_new_tokens,
        quiet=not verbose,
    )

    click.echo(
        f"Translating {src} → {tgt} "
        f"({family}, size={size}, distilled={distilled})"
    )
    click.echo(
        "Enter text (multiple lines ok). "
        "Empty line translates, two empty lines or Ctrl+C quits.\n"
    )

    last_was_empty = False
    while True:
        buffer: list[str] = []
        try:
            while True:
                line = click.prompt(
                    "",
                    prompt_suffix="> " if not buffer else ". ",
                    default="",
                    show_default=False,
                )
                if not line.strip():
                    break
                last_was_empty = False
                buffer.append(line)
        except (KeyboardInterrupt, EOFError):
            click.echo("\nBye.")
            return

        if not buffer:
            if last_was_empty:
                click.echo("Bye.")
                return
            last_was_empty = True
            continue

        last_was_empty = False
        results = translator.batch_translate(buffer)
        click.echo("")
        for translated in results:
            click.echo(translated)
        click.echo("")
