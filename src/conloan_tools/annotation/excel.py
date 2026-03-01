"""Standardised Excel/CSV export for ConLoan annotation sheets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass
class ColumnSpec:
    """One column in the output sheet."""

    name: str
    width_cm: float = 6.4


DEFAULT_ANNOTATION_COLUMNS: list[ColumnSpec] = [
    ColumnSpec("Loanword sentence", 6.4),
    ColumnSpec("Native sentence", 6.4),
    ColumnSpec("Target", 6.4),
    ColumnSpec("Valid.", 1.8),
    ColumnSpec("Suggestions", 6.4),
    ColumnSpec("Etymology", 10.0),
]

CM_TO_EXCEL_WIDTH = 3.89


def write_sheet(
    rows: list[dict] | pd.DataFrame,
    output: str,
    columns: Sequence[ColumnSpec] = DEFAULT_ANNOTATION_COLUMNS,
    sheet_name: str = "Annotation",
) -> None:
    """Write *rows* to *output* (.xlsx or .csv) with consistent styling.

    Only columns present in *columns* (and in the data) are written,
    in the order specified by *columns*.
    """
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)

    ordered = [c.name for c in columns if c.name in df.columns]
    df = df[ordered]

    if output.endswith(".csv"):
        df.to_csv(output, index=False)
        return

    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, index=False, sheet_name=sheet_name)

    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    wrap_format = workbook.add_format(
        {"text_wrap": True, "valign": "top"}
    )
    header_format = workbook.add_format(
        {"bold": True, "text_wrap": True, "valign": "top"}
    )

    for col_idx, spec in enumerate(columns):
        if spec.name not in df.columns:
            continue
        real_idx = ordered.index(spec.name)
        worksheet.set_column(
            real_idx, real_idx, spec.width_cm * CM_TO_EXCEL_WIDTH, wrap_format
        )

    for col_idx, col_name in enumerate(ordered):
        worksheet.write(0, col_idx, col_name, header_format)

    writer.close()
