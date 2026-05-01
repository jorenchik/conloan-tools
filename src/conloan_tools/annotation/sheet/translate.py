"""Post-hoc translation of sentences in an annotation sheet."""

import re

import click
import pandas as pd
from tqdm import tqdm

from conloan_tools.translate.nt_translate import Translator
from .excel import write_sheet


def strip_tags(sentence: str) -> str:
    """Remove ``<TAG123>``/``</TAG123>`` style numbered tags before translation."""
    return re.sub(r"</?[A-Za-z]+\d+>", "", sentence)


@click.command("translate")
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
    "--model",
    default=None,
    show_default=True,
    help="Override default Opus-MT model with any HF seq2seq model id.",
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
    "--precision",
    type=click.Choice(["fp32", "fp16", "bf16"]),
    default="fp16",
    show_default=True,
    help="Model precision (fp32 on CPU, fp16/bf16 on CUDA).",
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
    default="Label sentence",
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
def translate_sheet(
    input_file,
    src_lang,
    tgt_lang,
    model,
    nllb_src,
    nllb_tgt,
    device,
    max_new_tokens,
    precision,
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
    empty = df[target_col].isna() | (
        df[target_col].astype(str).str.strip() == ""
    )
    if overwrite:
        indices = df.index.tolist()
    else:
        indices = df[empty].index.tolist()

    skipped_nonempty = (~empty).sum() if not overwrite else 0

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

    click.echo("Loading translation model…")
    translator = Translator(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        model=model,
        nllb_src=nllb_src,
        nllb_tgt=nllb_tgt,
        device=device,
        max_new_tokens=max_new_tokens,
        precision=precision,
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
    click.echo(
        f"Processed: {len(translations)} translated, "
        f"{skipped_nonempty} skipped (non-empty)."
    )
    click.echo(f"Output written to: {output}")
