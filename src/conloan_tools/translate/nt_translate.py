from __future__ import annotations

import re

import click

from conloan_tools.resources import load_known_languages


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

DEFAULT_MODEL = "Helsinki-NLP/opus-mt-{src}-{tgt}"
NLLB_MODEL_PREFIX = "facebook/nllb"


def resolve_model_name(src_code: str, tgt_code: str, model: str | None = None) -> str:
    """Return a HF model id.

    If *model* is given it is returned as-is (allows arbitrary overrides).
    Otherwise the default Opus-MT pattern is used.
    """
    if model:
        return model
    return DEFAULT_MODEL.format(src=src_code, tgt=tgt_code)


def _is_nllb(model_name: str) -> bool:
    return model_name.startswith(NLLB_MODEL_PREFIX)


class Translator:
    """Seq2seq translator backed by a Marian (Opus-MT) model by default.

    Languages are accepted as either a raw ISO code (e.g. ``"lv"``) or a
    canonical name (e.g. ``"Latvian"``) as returned by
    ``load_known_languages()``.

    Pass *model* to override the default Opus-MT model with any HF seq2seq
    model id.  In that case you are responsible for ensuring the tokenizer
    and model are compatible.
    """

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
    ) -> None:
        import torch

        self.max_new_tokens = max_new_tokens
        self.quiet = quiet
        self._precision = precision
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        known = load_known_languages()
        name_to_code = {v.lower(): k for k, v in known.items()}
        self._src_code = self._resolve_code(src_lang, name_to_code, known)
        self._tgt_code = self._resolve_code(tgt_lang, name_to_code, known)

        model_name = resolve_model_name(self._src_code, self._tgt_code, model)
        self._nllb = _is_nllb(model_name)

        if self._nllb:
            if not nllb_src or not nllb_tgt:
                raise ValueError(
                    "NLLB models require --nllb-src and --nllb-tgt "
                    "(e.g. 'lvs_Latn', 'eng_Latn')."
                )
            self._nllb_src = nllb_src
            self._nllb_tgt = nllb_tgt
        else:
            self._nllb_src = self._nllb_tgt = None

        self._tokenizer, self._model = self._load(model_name)

    def translate(self, text: str) -> str:
        """Translate a single string, splitting on sentence boundaries."""
        if not text:
            return ""
        sentences = _SENTENCE_RE.split(text.strip())
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            return ""
        return " ".join(self.batch_translate(sentences))

    def batch_translate(self, texts: list[str]) -> list[str]:
        """Translate a list of strings without sentence splitting."""
        import torch

        tokenizer_kwargs = dict(
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if self._nllb:
            tokenizer_kwargs["src_lang"] = self._nllb_src

        inputs = self._tokenizer(texts, **tokenizer_kwargs).to(self._device)

        generate_kwargs: dict = dict(
            max_new_tokens=self.max_new_tokens,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=None,
        )
        if self._nllb:
            generate_kwargs["forced_bos_token_id"] = (
                self._tokenizer.convert_tokens_to_ids(self._nllb_tgt)
            )

        with torch.no_grad():
            tokens = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
                repetition_penalty=1.3,
            )

        return self._tokenizer.batch_decode(tokens, skip_special_tokens=True)

    @staticmethod
    def _resolve_code(
        lang: str,
        name_to_code: dict[str, str],
        code_to_name: dict[str, str],
    ) -> str:
        if lang in code_to_name:
            return lang
        key = lang.lower()
        if key in name_to_code:
            return name_to_code[key]
        raise ValueError(
            f"Unknown language {lang!r}. Use an ISO code or canonical "
            f"name from load_known_languages()."
        )

    def _load(self, model_name: str) -> tuple:  # noqa: C901
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from transformers import logging as tf_logging

        prev_verbosity = tf_logging.get_verbosity()
        if self.quiet:
            tf_logging.set_verbosity_error()

        click.echo(f"Loading model: {model_name} ({self._device})")

        try:
            tokenizer_kwargs = {}
            if self._nllb:
                tokenizer_kwargs["src_lang"] = self._nllb_src
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, **tokenizer_kwargs
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except OSError as exc:
            raise ValueError(
                f"Failed to load {model_name!r} from Hugging Face. "
                f"Check that the model exists and you have network access.\n"
                f"Original error: {exc}"
            ) from exc
        finally:
            if self.quiet:
                tf_logging.set_verbosity(prev_verbosity)

        if self._device == "cuda":
            if self._precision == "fp16":
                model = model.half()
            elif self._precision == "bf16":
                model = model.bfloat16()
        model = model.to(self._device).eval()

        return tokenizer, model


@click.command("interactive")
@click.option("--src", required=True, help="Source language (code or canonical name)")
@click.option("--tgt", required=True, help="Target language (code or canonical name)")
@click.option(
    "--model",
    default=None,
    help=(
        "Override the default Opus-MT model with any HF seq2seq model id. "
        "Example: Helsinki-NLP/opus-mt-tc-big-en-de"
    ),
)
@click.option(
    "--nllb-src",
    default=None,
    help="NLLB source language code (e.g. 'lvs_Latn'). Required for NLLB models.",
)
@click.option(
    "--nllb-tgt",
    default=None,
    help="NLLB target language code (e.g. 'eng_Latn'). Required for NLLB models.",
)
@click.option(
    "--max-new-tokens",
    type=int,
    default=512,
    show_default=True,
    help="Maximum tokens to generate per segment.",
)
@click.option(
    "--precision",
    type=click.Choice(["fp32", "fp16", "bf16"]),
    default="fp16",
    show_default=True,
    help="Model precision (fp32 on CPU, fp16/bf16 on CUDA).",
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
    model: str | None,
    nllb_src: str | None,
    nllb_tgt: str | None,
    max_new_tokens: int,
    precision: str,
    verbose: bool,
) -> None:
    """Interactive REPL for translating text line by line."""
    translator = Translator(
        src_lang=src,
        tgt_lang=tgt,
        model=model,
        nllb_src=nllb_src,
        nllb_tgt=nllb_tgt,
        max_new_tokens=max_new_tokens,
        precision=precision,
        quiet=not verbose,
    )

    click.echo(f"Translating {src} → {tgt}")
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
