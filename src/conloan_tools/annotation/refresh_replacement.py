import re
import click
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple
from collections import defaultdict

from .excel import write_sheet, DEFAULT_ANNOTATION_COLUMNS
import sys

@dataclass
class RowTransformResult:
    row_index: int
    status: str  # transformed | skipped_precondition | skipped_nonempty | skipped_unselected
    warnings: List[str] = field(default_factory=list)

REQUIRED_COLUMNS = {"Label sentence", "Replacement sentence", "Target", "Valid", "Reason", "Notes"}

MODE_CONFIG = {
    "baseline": {"allowed_prefixes": {"L", "N"}, "pairs": [("L", "N")]},
    "extended": {"allowed_prefixes": {"L", "N", "CS", "CN", "NE"}, "pairs": [("L", "N"), ("CS", "CN"), ("NE", "NE")]},
}

TRANSFORMATION_MAP = {
    "baseline": {"L": "N"},
    "extended": {"L": "N", "CS": "CN", "NE": "NE"},
}


def get_tag_spans(text: str) -> List[Tuple[str, str, int, int]]:
    """Extract all tag spans: (prefix, content, start_pos, end_pos)."""
    spans = []
    for tag_match in re.finditer(r"<([A-Z]+)(\d+)>", text):
        prefix = tag_match.group(1)
        num = int(tag_match.group(2))
        start = tag_match.start()
        close_match = re.search(rf"</{prefix}{num}>", text)
        if close_match:
            end = close_match.end()
            content = text[start + len(f"<{prefix}{num}>"):end - len(f"</{prefix}{num}>")]
            spans.append((prefix, content, start, end))
    return spans


def check_row_preconditions(text: str, field_name: str, allowed_prefixes: set) -> List[str]:
    """Check V1.1-V1.7 preconditions. Returns list of warning messages."""
    warnings = []
    if not isinstance(text, str) or pd.isna(text):
        return warnings

    # V1.1: Check for line breaks
    if "\n" in text or "\r" in text:
        warnings.append(f"V1.1: Line breaks found in {field_name}")

    # V1.2/V1.3: Check for illegal tags
    all_tags = re.findall(r"<([A-Z]+)(\d*)>", text)
    for prefix, digits in all_tags:
        if not digits:
            warnings.append(f"V1.2: Illegal digit-free tag <{prefix}> in {field_name}")
        elif prefix not in allowed_prefixes:
            warnings.append(f"V1.3: Illegal tag prefix <{prefix}{digits}> in {field_name}")

    # V1.4/V1.5/V1.6/V1.7: Tag balance, nesting, correspondence, orphans
    stack = []
    for tag_match in re.finditer(r"<(/)?([A-Z]+)(\d+)>", text):
        is_closing = tag_match.group(1) == "/"
        prefix = tag_match.group(2)
        num = tag_match.group(3)
        full_tag = f"<{prefix}{num}>"
        start_pos = tag_match.start()

        if not is_closing:
            if stack:
                prev_full = stack[-1][2]
                prev_match = re.match(r"<([A-Z]+)(\d+)>", prev_full)
                if prev_match:
                    prev_prefix, prev_num = prev_match.group(1, 2)
                    warnings.append(f"V1.5: Nested tags <{prefix}{num}> inside <{prev_prefix}{prev_num}> in {field_name}")
            stack.append((prefix, num, full_tag, start_pos))
        else:
            if not stack:
                warnings.append(f"V1.7: Closing tag </{prefix}{num}> has no opening tag in {field_name}")
            else:
                top_prefix, top_num, top_full, _ = stack[-1]
                if top_prefix != prefix or top_num != num:
                    warnings.append(f"V1.6: Tag {top_full} closed by </{prefix}{num}> in {field_name}")
                else:
                    stack.pop()

    # V1.4: Unclosed tags
    for prefix, num, full_tag, _ in stack:
        warnings.append(f"V1.4: Tag {full_tag} is never closed in {field_name}")

    return warnings


def transform_row(row, mode: str) -> Tuple[str, List[str]]:
    """
    Pure function: transform Label sentence to Replacement sentence.
    Returns (new_replacement_sentence, warnings).
    No I/O.
    """
    label_sent = str(row.get("Label sentence", ""))
    if pd.isna(row.get("Label sentence")):
        return "", []

    config = MODE_CONFIG[mode]
    allowed_prefixes = config["allowed_prefixes"]
    trans_map = TRANSFORMATION_MAP[mode]

    # Check preconditions
    warnings = check_row_preconditions(label_sent, "Label sentence", allowed_prefixes)
    if warnings:
        return "", warnings

    # Perform transformation
    result = label_sent
    for tag_match in re.finditer(r"<([A-Z]+)(\d+)>", label_sent):
        prefix = tag_match.group(1)
        num = tag_match.group(2)
        open_tag = tag_match.group(0)
        close_tag = f"</{prefix}{num}>"

        if prefix in trans_map:
            new_prefix = trans_map[prefix]
            new_open = f"<{new_prefix}{num}>"
            new_close = f"</{new_prefix}{num}>"

            # Replace opening tag
            result = result.replace(open_tag, new_open, 1)
            # Replace closing tag
            result = result.replace(close_tag, new_close, 1)

    return result, warnings


def transform_file(input_path: str, output_path: str, mode: str,
                   process_all: bool = False, overwrite: bool = False) -> List[RowTransformResult]:
    """Load Excel, transform selected rows, write output via write_sheet."""
    df = pd.read_excel(input_path)

    # Ensure Replacement sentence column is object type to allow string assignment
    df["Replacement sentence"] = df["Replacement sentence"].astype(object)

    # V0.0 gate: check columns
    actual_columns = set(df.columns)
    extra_columns = actual_columns - REQUIRED_COLUMNS
    missing_columns = REQUIRED_COLUMNS - actual_columns

    if extra_columns or missing_columns:
        error_messages = []
        if missing_columns:
            error_messages.append(f"Missing required columns: {sorted(missing_columns)}")
        if extra_columns:
            error_messages.append(f"Extra columns present: {sorted(extra_columns)}")
        raise ValueError("; ".join(error_messages))

    results = []
    for idx, row in df.iterrows():
        row_index = idx + 2  # 1-based Excel line number
        row_valid = str(row.get("Valid", "")).strip()

        selected = process_all or row_valid == "+"

        current_replacement = row.get("Replacement sentence", "")
        has_replacement = isinstance(current_replacement, str) and current_replacement.strip() != ""

        if not selected:
            status = "skipped_unselected"
            warnings = []
        elif has_replacement and not overwrite:
            status = "skipped_nonempty"
            warnings = ["Replacement sentence is already non-empty"]
        else:
            new_replacement, warnings = transform_row(row, mode)
            if warnings:
                status = "skipped_precondition"
            else:
                status = "transformed"
                df.at[idx, "Replacement sentence"] = new_replacement

        results.append(RowTransformResult(row_index=row_index, status=status, warnings=warnings))

    write_sheet(df, output_path, DEFAULT_ANNOTATION_COLUMNS)
    return results


@click.command("refresh-replacement")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", "output_path", type=click.Path(), required=True,
              help="Path to output file (.xlsx or .csv)")
@click.option("--mode", type=click.Choice(["baseline", "extended"]), default="baseline",
              show_default=True)
@click.option("--process-all", is_flag=True, help="Process all rows, not just Valid='+'")
@click.option("--overwrite", is_flag=True, help="Overwrite existing Replacement sentence")
@click.option("--verbose", "-v", is_flag=True, help="Show per-row status")
def refresh_replacement(input_file, output_path, mode, process_all, overwrite, verbose):
    """Derive Replacement sentence from Label sentence by tag prefix remapping."""
    results = transform_file(input_file, output_path, mode, process_all, overwrite)

    transformed = sum(1 for r in results if r.status == "transformed")
    skipped_precondition = sum(1 for r in results if r.status == "skipped_precondition")
    skipped_nonempty = sum(1 for r in results if r.status == "skipped_nonempty")
    skipped_unselected = sum(1 for r in results if r.status == "skipped_unselected")

    click.echo(f"Processed: {transformed} transformed, "
               f"{skipped_precondition} skipped (precondition), "
               f"{skipped_nonempty} skipped (non-empty), "
               f"{skipped_unselected} skipped (unselected)")

    if verbose:
        for r in results:
            if r.warnings:
                click.echo(f"Row {r.row_index}: {r.status} - {'; '.join(r.warnings)}")

    click.echo(f"Output written to: {output_path}")


if __name__ == "__main__":
    refresh_replacement()
