"""Post-hoc translation of sentences in an annotation sheet."""

import re

import click
import pandas as pd
from tqdm import tqdm

from conloan_tools.translate.nt_translate import (
    ModelFamily,
    ModelSize,
    Translator,
)
from .excel import write_sheet


# ── helpers ──────────────────────────────────────────────────────────


def strip_tags(sentence: str) -> str:
    """Remove ``<L1></L1>`` / ``<N1></N1>`` tags for translation."""
    return re.sub(r"</?[LN]\d+>", "", sentence)


# ── CLI ──────────────────────────────────────────────────────────────


@click.command("translate-target")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--src-lang",
    required=True,
    help="Source language (ISO code or canonical name, e.g. 'lv')",
)
@click.option(
    "--tgt-lang",
    required=True,
    help="Target language (ISO code or canonical name, e.g. 'en')",
)
@click.option(
    "--family",
    type=click.Choice([f.value for f in ModelFamily], case_sensitive=False),
    default=ModelFamily.OPUS.value,
    show_default=True,
    help="Model family",
)
@click.option(
    "--distilled/--no-distilled",
    default=True,
    show_default=True,
    help="Use distilled model variant",
)
@click.option(
    "--size",
    type=click.Choice([s.value for s in ModelSize], case_sensitive=False),
    default=ModelSize.SMALL.value,
    show_default=True,
    help="Model size",
)
@click.option(
    "--prefer-big",
    is_flag=True,
    default=False,
    show_default=True,
    help="Prefer larger model when available",
)
@click.option(
    "--device",
    default=None,
    help="Torch device (e.g. 'cuda', 'cpu'); auto-detected if omitted",
)
@click.option(
    "--max-new-tokens",
    type=int,
    default=512,
    show_default=True,
    help="Maximum new tokens per translation",
)
@click.option(
    "--batch-size",
    type=int,
    default=32,
    show_default=True,
    help="Translation batch size",
)
@click.option(
    "--output",
    default=None,
    help="Output path (defaults to overwriting input)",
)
@click.option(
    "--source-col",
    default="Loanword sentence",
    show_default=True,
    help="Column containing sentences to translate",
)
@click.option(
    "--target-col",
    default="Target",
    show_default=True,
    help="Column to write translations into",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    show_default=True,
    help="Overwrite existing translations in target column",
)
@click.option(
    "--keep-tags/--strip-tags",
    default=False,
    show_default=True,
    help="Keep / strip <L#> and <N#> tags before translation",
)
def translate_target(
    input_file,
    src_lang,
    tgt_lang,
    family,
    distilled,
    size,
    prefer_big,
    device,
    max_new_tokens,
    batch_size,
    output,
    source_col,
    target_col,
    overwrite,
    keep_tags,
):
    """Translate a column of sentences in an annotation sheet.

    Reads SOURCE_COL, optionally strips ``<L𝑛>``/``<N𝑛>`` tags, translates
    from SRC_LANG to TGT_LANG, and writes results into TARGET_COL.
    """
    output = output or input_file

    if input_file.endswith(".csv"):
        df = pd.read_csv(input_file)
    else:
        df = pd.read_excel(input_file)

    if source_col not in df.columns:
        raise click.ClickException(
            f"Column '{source_col}' not found. "
            f"Available: {list(df.columns)}"
        )

    if target_col not in df.columns:
        df[target_col] = ""

    # identify rows needing translation
    if overwrite:
        indices = df.index.tolist()
    else:
        empty = df[target_col].isna() | (
            df[target_col].astype(str).str.strip() == ""
        )
        indices = df[empty].index.tolist()

    if not indices:
        click.echo("No rows to translate.")
        return

    click.echo(f"Found {len(indices)} rows to translate.")

    # prepare source texts
    sentences: list[str] = []
    for idx in indices:
        text = str(df.at[idx, source_col])
        if not keep_tags:
            text = strip_tags(text)
        sentences.append(text)

    # init translator
    click.echo("Loading translation model…")
    translator = Translator(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        family=ModelFamily(family),
        distilled=distilled,
        size=ModelSize(size),
        prefer_big=prefer_big,
        device=device,
        max_new_tokens=max_new_tokens,
        quiet=True,
    )

    # translate in batches
    click.echo(f"Translating {len(sentences)} sentences…")
    translations: list[str] = []
    for i in tqdm(
        range(0, len(sentences), batch_size),
        desc="Translating",
        unit="batch",
    ):
        batch = sentences[i : i + batch_size]
        translations.extend(translator.batch_translate(batch))

    df["Target"] = translations

    write_sheet(df, output)
    click.echo(f"Done. Translated {len(translations)} rows → {output}.")
