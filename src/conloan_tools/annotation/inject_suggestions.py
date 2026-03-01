"""Post-hoc injection of WordNet synonym suggestions into an annotation sheet."""

import re

import click
import pandas as pd

from conloan_tools.stz.lemmatize import Lemmatizer
from conloan_tools.wordnet.query import WordNet
from .excel import write_sheet


# ── helpers ──────────────────────────────────────────────────────────


def extract_tagged_words(sentence: str) -> dict[int, str]:
    """Extract ``{tag_num: surface_form}`` from ``<L1>word</L1>`` tags."""
    return {
        int(n): w
        for n, w in re.findall(r"<L(\d+)>([^<]+)</L\d+>", sentence)
    }


def _format_word_synonyms(lemma: str, wn: WordNet) -> str:
    result = wn.get_synonym_groups(lemma)
    if not result.found:
        return ""

    lines: list[str] = []
    sense_num = 1
    for entry in result.entries:
        for sense in entry.senses:
            syns = set(sense.synonyms) if sense.synonyms else set()
            syns.discard(lemma)
            if not syns:
                continue
            definition = getattr(sense, "definition", "") or ""
            syn_str = ", ".join(sorted(syns))
            if definition:
                lines.append(f"{sense_num}. [{definition}]: {syn_str}")
            else:
                lines.append(f"{sense_num}. {syn_str}")
            sense_num += 1

    return "\n".join(lines)


def format_suggestions(
    tagged_words: dict[int, str],
    lemmatizer: Lemmatizer,
    wn: WordNet,
) -> str:
    """Lemmatize each tagged surface form, look up synonyms, format."""
    lines: list[str] = []
    for tag_num in sorted(tagged_words):
        surface = tagged_words[tag_num]
        lemma = lemmatizer.lemmatize_word(surface)
        word_syns = _format_word_synonyms(lemma, wn)
        if word_syns:
            lines.append(f"L{tag_num} ({lemma}):")
            for line in word_syns.split("\n"):
                lines.append(f"  {line}")
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────


@click.command("inject-suggestions")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--wordnet-xml",
    required=True,
    type=click.Path(exists=True),
    help="Path to WordNet LMF XML",
)
@click.option(
    "--language",
    required=True,
    help="Canonical language name (e.g. 'Latvian')",
)
@click.option(
    "--output",
    default=None,
    help="Output path (defaults to overwriting input)",
)
@click.option(
    "--column",
    default="Loanword sentence",
    show_default=True,
    help="Column containing tagged sentences",
)
def inject_suggestions(input_file, wordnet_xml, language, output, column):
    """Inject WordNet synonym suggestions into an annotation sheet.

    Reads <L𝑛>…</L𝑛> tags from COLUMN, lemmatizes each surface form
    with Stanza, queries WordNet, and writes the Suggestions column.
    """
    output = output or input_file

    click.echo("Initializing WordNet…")
    wn = WordNet(wordnet_xml)

    click.echo(f"Initializing Stanza lemmatizer for '{language}'…")
    lemmatizer = Lemmatizer(language)
    click.echo(f"Initialized lemmatizer.")
    click.echo(f"Model identifier: '{lemmatizer.model_identifier}'.")

    if input_file.endswith(".csv"):
        df = pd.read_csv(input_file)
    else:
        df = pd.read_excel(input_file)

    if column not in df.columns:
        raise click.ClickException(f"Column '{column}' not found.")

    suggestions: list[str] = []
    for sentence in df[column].fillna(""):
        tagged = extract_tagged_words(str(sentence))
        if tagged:
            suggestions.append(
                format_suggestions(tagged, lemmatizer, wn)
            )
        else:
            suggestions.append("")

    df["Suggestions"] = suggestions

    write_sheet(df, output)
    click.echo(f"Done. Wrote {len(df)} rows to {output}.")
