"""Reverse transform: JSON dataset → annotated XLSX/CSV sheet."""

import json
import click
import pandas as pd
from pathlib import Path

from conloan_tools.annotation.sheet.excel import (
    write_sheet,
    DEFAULT_ANNOTATION_COLUMNS,
)


@click.command("sheet-from-json")
@click.argument("input_json", type=click.Path(exists=True))
@click.argument("output_xlsx", type=click.Path())
@click.option(
    "--sheet-name",
    default="Annotation",
    show_default=True,
    help="Sheet name in the output XLSX.",
)
def sheet_from_json(input_json, output_xlsx, sheet_name):
    """Reconstruct an annotation sheet from a JSON dataset."""
    input_path = Path(input_json)
    output_path = Path(output_xlsx)

    if output_path.exists():
        if not click.confirm(f"Overwrite {output_path}?"):
            click.echo("Aborted.")
            return

    with open(input_path, encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        click.secho("Error: JSON root must be a list of entries.", fg="red")
        raise SystemExit(1)

    rows = []
    for entry in dataset:
        rows.append(
            {
                "Label sentence": entry.get("source_annotated_loanwords", ""),
                "Replacement sentence": entry.get(
                    "source_annotated_loanwords_replaced", ""
                ),
                "Target": entry.get("target", ""),
                "Valid": "+",
                "Reason": "",
                "Notes": "",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_sheet(
        rows,
        str(output_path),
        columns=DEFAULT_ANNOTATION_COLUMNS,
        sheet_name=sheet_name,
    )

    click.secho(
        f"Success: {len(rows)} rows written to {output_path}", fg="green"
    )
