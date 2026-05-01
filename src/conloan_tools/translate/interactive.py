"""Interactive REPL for translating text line by line."""

from __future__ import annotations

import click

from conloan_tools.translate.nt_translate import Translator
from conloan_tools.translate.llm_translate import LLMTranslator


@click.command("interactive")
@click.option("--src", required=True, help="Source language (code or canonical name)")
@click.option("--tgt", required=True, help="Target language (code or canonical name)")
@click.option(
    "--backend",
    type=click.Choice(["opus-mt", "llm"]),
    default="opus-mt",
    show_default=True,
    help="Translation backend: seq2seq Opus-MT or decoder-only LLM.",
)
@click.option(
    "--model",
    default=None,
    help=(
        "Override the default model with any supported model id. "
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
    backend: str,
    model: str | None,
    nllb_src: str | None,
    nllb_tgt: str | None,
    max_new_tokens: int,
    precision: str,
    verbose: bool,
) -> None:
    """Interactive REPL for translating text line by line."""
    if backend == "llm":
        translator = LLMTranslator(
            src_lang=src,
            tgt_lang=tgt,
            model=model,
            max_new_tokens=max_new_tokens,
            precision=precision,
            quiet=not verbose,
        )
    else:
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

    click.echo(f"Translating {src} → {tgt} (backend: {backend})")
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
