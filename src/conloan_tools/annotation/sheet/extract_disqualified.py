"""Extract rows whose Reason matches a given set of values."""
from __future__ import annotations

import click
import pandas as pd

from .excel import write_sheet, DEFAULT_ANNOTATION_COLUMNS

REQUIRED_COLUMNS = {"Label sentence", "Replacement sentence", "Target", "Valid", "Reason", "Notes"}


def extract_disqualified(
    input_path: str,
    output_path: str,
    reasons: list[str],
) -> int:
    """Load *input_path*, keep rows whose trimmed Reason is in *reasons*, write to *output_path*.

    Returns the number of rows written.
    """
    df = pd.read_excel(input_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    reason_set = {r.strip() for r in reasons}

    mask = df["Reason"].apply(
        lambda v: isinstance(v, str) and v.strip() in reason_set
    )
    filtered = df[mask]

    write_sheet(filtered, output_path, DEFAULT_ANNOTATION_COLUMNS)
    return len(filtered)


@click.command("extract-disqualified")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", "output_path", type=click.Path(), required=True,
              help="Path to output file (.xlsx or .csv)")
@click.option("--reasons", "-r", required=True,
              help='Comma-separated Reason values to keep, e.g. "NE,CS"')
def extract_disqualified_cmd(input_file, output_path, reasons):
    """Extract rows whose Reason (trimmed) matches one of the given values."""
    reason_list = [r.strip() for r in reasons.split(",") if r.strip()]
    if not reason_list:
        raise click.UsageError("--reasons must contain at least one non-empty value.")

    count = extract_disqualified(input_file, output_path, reason_list)
    click.echo(f"Extracted {count} row(s) matching reasons: {reason_list}")
    click.echo(f"Output written to: {output_path}")


if __name__ == "__main__":
    extract_disqualified_cmd()
