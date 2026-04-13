import sys
import csv
from pathlib import Path

import click


def _open_csv(path: str) -> tuple[list[dict], list[str]]:
    p = Path(path)
    if not p.exists():
        click.echo(f"Error: file not found: {path}", err=True)
        sys.exit(1)
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, list(fieldnames)


def _require_column(fieldnames: list[str], col: str, label: str) -> None:
    if col not in fieldnames:
        click.echo(
            f"Error: {label} column '{col}' not found. "
            f"Available: {', '.join(fieldnames)}",
            err=True,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group("csv-extract")
def csv_extract():
    """Extract and filter columns from CSV files."""


# ---------------------------------------------------------------------------
# words
# ---------------------------------------------------------------------------


@csv_extract.command("words")
@click.argument("path")
@click.option(
    "--word-col",
    default="word",
    show_default=True,
    help="Column name containing the words.",
)
@click.option(
    "--valid-col",
    default=None,
    help="Column name for validity filter. Only rows where this column equals "
         "--valid-value are kept.",
)
@click.option(
    "--valid-value",
    default="x",
    show_default=True,
    help="Value in --valid-col that marks a row as valid.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output file path. Defaults to stdout.",
)
@click.option(
    "--strip",
    is_flag=True,
    default=False,
    help="Strip leading/trailing whitespace from word values.",
)
@click.option(
    "--skip-empty",
    is_flag=True,
    default=False,
    help="Skip rows where the word cell is empty.",
)
def cmd_words(
    path: str,
    word_col: str,
    valid_col: str | None,
    valid_value: str,
    output: str | None,
    strip: bool,
    skip_empty: bool,
) -> None:
    """
    Extract words from a CSV column, one per line.

    Optionally filter by a validity column (--valid-col / --valid-value).
    Unexpected values in the validity column trigger a warning.
    """
    rows, fieldnames = _open_csv(path)
    _require_column(fieldnames, word_col, "word")

    if valid_col is not None:
        _require_column(fieldnames, valid_col, "valid")

    words: list[str] = []
    skipped_empty    = 0
    skipped_invalid  = 0
    warn_values: set[str] = set()

    for i, row in enumerate(rows, start=2):  # row 1 is header
        word = row[word_col]
        if strip:
            word = word.strip()

        if skip_empty and not word:
            skipped_empty += 1
            continue

        if valid_col is not None:
            cell = row[valid_col].strip()
            if cell != valid_value:
                if cell:  # non-empty unexpected value
                    warn_values.add(cell)
                skipped_invalid += 1
                continue

        words.append(word)

    if warn_values:
        click.echo(
            f"Warning: unexpected value(s) in '{valid_col}' column "
            f"(expected '{valid_value}'): "
            + ", ".join(sorted(f"'{v}'" for v in warn_values)),
            err=True,
        )

    if skipped_empty:
        click.echo(f"Skipped {skipped_empty:,} empty word row(s).", err=True)
    if skipped_invalid and valid_col is not None:
        click.echo(
            f"Skipped {skipped_invalid:,} row(s) where "
            f"'{valid_col}' != '{valid_value}'.",
            err=True,
        )

    click.echo(f"Extracted {len(words):,} word(s).", err=True)

    out_text = "\n".join(words) + ("\n" if words else "")

    if output is not None:
        Path(output).write_text(out_text, encoding="utf-8")
        click.echo(f"Written to {output}", err=True)
    else:
        click.echo(out_text, nl=False)
