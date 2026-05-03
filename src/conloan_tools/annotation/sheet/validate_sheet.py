import re
import click
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Set, Counter, Optional, Dict, Tuple
from collections import defaultdict

@dataclass
class ValidationError:
    rule_id: str
    field: str
    message: str
    severity: str = "error"

@dataclass
class RowResult:
    row_index: int
    errors: List[ValidationError] = field(default_factory=list)

# Mode configurations
MODE_CONFIG = {
    "baseline": {
        "allowed_prefixes": {"L", "N"},
        "column_prefixes": {
            "Label sentence": {"L"},
            "Replacement sentence": {"N"},
        },
        "pairs": [("L", "N")],
    },
    "extended": {
        "allowed_prefixes": {"L", "N", "CS", "NE"},
        "column_prefixes": {
            "Label sentence": {"L", "CS", "NE"},
            "Replacement sentence": {"N", "CS", "NE"},
        },
        "pairs": [("L", "N"), ("CS", "CS"), ("NE", "NE")],
    },
}

REQUIRED_COLUMNS = {"Label sentence", "Replacement sentence", "Target", "Valid", "Reason"}

def get_tag_spans(text: str) -> List[Tuple[str, str, int, int]]:
    """Extract all tag spans: (prefix, content, start_pos, end_pos)."""
    spans = []
    for tag_match in re.finditer(r"<([A-Z]+)(\d+)>", text):
        prefix = tag_match.group(1)
        num = int(tag_match.group(2))
        start = tag_match.start()
        # Find closing tag
        close_match = re.search(rf"</{prefix}{num}>", text)
        if close_match:
            end = close_match.end()
            content = text[start + len(f"<{prefix}{num}>"):end - len(f"</{prefix}{num}>")]
            spans.append((prefix, content, start, end))
    return spans

def get_tag_stats(text: str, field_name: str, allowed_prefixes: Set[str], column_prefixes: Optional[Set[str]] = None) -> Tuple[List[str], List[ValidationError]]:
    """Returns list of opening tags and list of ValidationErrors."""
    errors = []
    if not isinstance(text, str) or pd.isna(text):
        return [], errors

    # V1.1: Check for line breaks
    if "\n" in text or "\r" in text:
        errors.append(ValidationError("V1.1", field_name, "Sentence contains forbidden line breaks."))
        return [], errors

    open_tags = []

    # V1.2: Check for illegal digit-free tags (<L>, <UNK> etc)
    # and collect all found tags
    all_tags = re.findall(r"<([A-Z]+)(\d*)>", text)
    for prefix, digits in all_tags:
        full_tag = f"<{prefix}{digits}>" if digits else f"<{prefix}>"
        if not digits:
            errors.append(ValidationError("V1.2", field_name, f"Illegal digit-free tag: {full_tag}"))
        elif prefix not in allowed_prefixes:
            errors.append(ValidationError("V1.3", field_name, f"Illegal tag prefix: {full_tag}"))
        elif column_prefixes is not None and prefix not in column_prefixes:
            errors.append(ValidationError("V1.8", field_name,
                f"Tag prefix '{prefix}' is not allowed in this column (allowed: {sorted(column_prefixes)})."))
        else:
            open_tags.append(f"{prefix}{digits}")

    # V1.4/V1.5/V1.6/V1.7: Tag balance, nesting, exact correspondence, orphans
    stack = []  # Stores (tag_id, full_match_string, start_pos)
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
                    errors.append(ValidationError("V1.5", field_name,
                        f"Nested tags are forbidden (<{prefix}{num}> inside <{prev_prefix}{prev_num}>)"))
            stack.append((prefix, num, full_tag, start_pos))
        else:
            if not stack:
                errors.append(ValidationError("V1.7", field_name, f"Closing tag </{prefix}{num}> has no opening tag."))
            else:
                top_prefix, top_num, top_full, _ = stack[-1]
                if top_prefix != prefix or top_num != num:
                    errors.append(ValidationError("V1.6", field_name,
                        f"Tag {top_full} closed by incorrect tag </{prefix}{num}>."))
                else:
                    stack.pop()

    # Unclosed tags
    for prefix, num, full_tag, _ in stack:
        errors.append(ValidationError("V1.4", field_name, f"Tag {full_tag} is never closed."))

    # V4.2: Punctuation inside tags - only for L and N prefixes
    # V4.4: Empty tag spans
    # V4.5: Whitespace-only tag spans
    for prefix, content, start, end in get_tag_spans(text):
        num_match = re.search(r'(\d+)', text[start:])
        if not num_match:
            continue
        num = num_match.group(1)
        close_tag = f"</{prefix}{num}>"
        close_pos = text.find(close_tag, start)
        if close_pos == -1:
            continue
        inner = text[start + len(f"<{prefix}{num}>"):close_pos]

        # V4.4: Empty span
        if not inner:
            errors.append(ValidationError("V4.4", field_name, f"Empty tag span: <{prefix}{num}></{prefix}{num}>"))
            continue

        # V4.5: Whitespace-only span
        if inner.isspace():
            errors.append(ValidationError("V4.5", field_name, f"Whitespace-only tag span: <{prefix}{num}></{prefix}{num}>"))
            continue

        # V4.2: Punctuation inside tags (only for L, N)
        if prefix in ("L", "N"):
            punct_match = re.search(r'[.?!,;]', inner)
            if punct_match:
                errors.append(ValidationError("V4.2", field_name,
                    f"Punctuation '{punct_match.group()}' inside tag <{prefix}{num}>"))

    # V4.3: Fused tags - closing tag followed by non-whitespace/punct
    fused_matches = re.finditer(r'</([A-Z]+)(\d+)>([^\s,.!?;:"\')\]}]+)', text)
    for match in fused_matches:
        errors.append(ValidationError("V4.3", field_name,
            f"Missing space/punctuation after closing tag: {match.group(0)}"))

    return open_tags, errors

def extract_tag_indices(text: str) -> Dict[str, Set[int]]:
    """Extract prefix -> set of indices from text."""
    result = defaultdict(set)
    for prefix, digits in re.findall(r"<([A-Z]+)(\d+)>", text):
        result[prefix].add(int(digits))
    return result

def validate_row(row: pd.Series, mode: str, warn_undecided: bool = True) -> List[ValidationError]:
    """Validate a single row according to spec rules."""
    errors = []

    # Get field values
    valid_raw = row.get("Valid", "")
    reason_raw = row.get("Reason", "")

    # Handle Valid field - distinguish blank from whitespace-only
    if pd.isna(valid_raw):
        valid = ""
        is_whitespace_only = False
    elif isinstance(valid_raw, str):
        stripped = valid_raw.strip()
        is_whitespace_only = (stripped == "" and valid_raw != "")
        valid = stripped
    else:
        valid = str(valid_raw).strip()
        is_whitespace_only = False

    # S9: Whitespace-only Valid is a warning (distinct from blank V0.4)
    if is_whitespace_only and warn_undecided:
        errors.append(ValidationError("S9", "Valid", "Valid contains only whitespace.", severity="warning"))

    reason = "" if pd.isna(reason_raw) else str(reason_raw).strip()
    target = str(row.get("Target", ""))
    loan_sent = str(row.get("Label sentence", ""))
    native_sent = str(row.get("Replacement sentence", ""))
    label_sent = str(row.get("Label", ""))
    replacement_sent = str(row.get("Replacement", ""))

    config = MODE_CONFIG[mode]
    allowed_prefixes = config["allowed_prefixes"]

    # V0: Metadata Integrity
    if valid not in ("", "+", "-"):
        errors.append(ValidationError("V0.1", "Valid", f"Invalid value '{valid}' for Valid. Must be blank, '+', or '-'."))

    if valid == "-":
        if reason not in ("NL", "CS", "NE", "NF"):
            errors.append(ValidationError("V0.2", "Reason", f"Reason must be one of: NL, CS, NE, NF. Got: '{reason}'"))
    elif valid in ("+", ""):
        if reason:
            errors.append(ValidationError("V0.3", "Reason", f"Reason must be empty when Valid is '+' or blank. Got: '{reason}'"))

    # V0.4: Truly blank Valid (not whitespace-only - that's S9)
    if valid == "" and not is_whitespace_only and warn_undecided:
        errors.append(ValidationError("V0.4", "Valid", "Row is undecided (blank Valid).", severity="warning"))

    # V4.1: Target must contain no tags
    if re.search(r"</?[A-Z]+\d+>", target):
        errors.append(ValidationError("V4.1", "Target", "Target must not contain tags."))

    # V1: Tag syntax and legality for sentences
    column_prefixes = config.get("column_prefixes", {})
    loan_tags, l_errors = get_tag_stats(loan_sent, "Label sentence", allowed_prefixes, column_prefixes.get("Label sentence"))
    native_tags, n_errors = get_tag_stats(native_sent, "Replacement sentence", allowed_prefixes, column_prefixes.get("Replacement sentence"))
    errors.extend(l_errors)
    errors.extend(n_errors)

    # V2: Within-sentence tag set integrity
    for tags, field in [(loan_tags, "Label sentence"), (native_tags, "Replacement sentence")]:
        counts = Counter(tags)
        for tag, count in counts.items():
            if count > 1:
                errors.append(ValidationError("V2.1", field, f"Duplicate tag: {tag}"))

        prefix_map = extract_tag_indices("".join([f"<{t}>" for t in tags]))
        for pref, nums in prefix_map.items():
            sorted_nums = sorted(nums)
            expected = list(range(1, len(nums) + 1))
            if sorted_nums != expected:
                errors.append(ValidationError("V2.2", field,
                    f"{pref} tags must start at 1 and be incremental. Found: {sorted_nums}"))

    # V3: Cross-sentence parity
    def get_prefix_indices(text: str) -> Dict[str, Set[int]]:
        result = defaultdict(set)
        for match in re.finditer(r"<([A-Z]+)(\d+)>", text):
            result[match.group(1)].add(int(match.group(2)))
        return result

    loan_indices = get_prefix_indices(loan_sent)
    native_indices = get_prefix_indices(native_sent)
    label_indices = get_prefix_indices(label_sent)
    replacement_indices = get_prefix_indices(replacement_sent)

    for l_pref, n_pref in config["pairs"]:
        if l_pref == "NE":
            # V3.3: NE indices in Label must equal NE in Replacement (extended only)
            if label_indices.get("NE", set()) != replacement_indices.get("NE", set()):
                l_set = sorted(label_indices.get("NE", []))
                r_set = sorted(replacement_indices.get("NE", []))
                errors.append(ValidationError("V3.3", "Label/Replacement",
                    f"NE indices mismatch: Label has {l_set}, Replacement has {r_set}"))
            # V3.4: Content inside matched NE index must be identical
            ne_nums = label_indices.get("NE", set()) & replacement_indices.get("NE", set())
            for ne_num in sorted(ne_nums):
                # Extract content for this NE tag in Label
                label_match = re.search(rf"<NE{ne_num}>(.*?)</NE{ne_num}>", label_sent)
                repl_match = re.search(rf"<NE{ne_num}>(.*?)</NE{ne_num}>", replacement_sent)
                if label_match and repl_match:
                    if label_match.group(1) != repl_match.group(1):
                        errors.append(ValidationError("V3.4", "Label/Replacement",
                            f"NE{ne_num} content mismatch: Label='{label_match.group(1)}' vs Replacement='{repl_match.group(1)}'"))
        else:
            # L↔N and CS↔CS
            l_indices = loan_indices.get(l_pref, set())
            n_indices = native_indices.get(n_pref, set())
            if l_indices != n_indices:
                errors.append(
                    ValidationError(
                        "V3.1" if l_pref == "L" else "V3.2",
                        "Label/Replacement",
                        f"{l_pref} indices mismatch: Label has {sorted(l_indices)}, Replacement has {sorted(n_indices)}",
                    )
                )

            # V3.5: Content inside matched CS index must be identical
            if l_pref == "CS":
                cs_nums = l_indices & n_indices
                for cs_num in sorted(cs_nums):
                    label_match = re.search(
                        rf"<CS{cs_num}>(.*?)</CS{cs_num}>", loan_sent
                    )
                    repl_match = re.search(
                        rf"<CS{cs_num}>(.*?)</CS{cs_num}>", native_sent
                    )
                    if (
                        label_match
                        and repl_match
                        and label_match.group(1) != repl_match.group(1)
                    ):
                        errors.append(
                            ValidationError(
                                "V3.5",
                                "Label/Replacement",
                                f"CS{cs_num} content mismatch: Label='{label_match.group(1)}' vs Replacement='{repl_match.group(1)}'",
                            )
                        )

    return errors

def validate_file(input_file: str, mode: str, validate_all: bool = False, warn_undecided: bool = True) -> List[RowResult]:
    """Load Excel file and validate all rows. Returns list of RowResult."""
    df = pd.read_excel(input_file)
    results = []

    # V0.0: Metadata integrity - check columns first
    actual_columns = set(df.columns)
    extra_columns = actual_columns - REQUIRED_COLUMNS
    missing_columns = REQUIRED_COLUMNS - actual_columns

    if extra_columns or missing_columns:
        errors = []
        if missing_columns:
            errors.append(ValidationError("V0.0", "file",
                f"Missing required columns: {sorted(missing_columns)}"))
        # V0.0 gates all further validation
        return [RowResult(row_index=1, errors=errors)]

    for idx, row in df.iterrows():
        row_valid = str(row.get("Valid", "")).strip().lower()

        # If not validate_all, only process '+' rows
        if not validate_all and row_valid != "+":
            continue

        errors = validate_row(row, mode, warn_undecided)
        if errors:
            results.append(RowResult(row_index=idx + 2, errors=errors))

    return results

@click.command("validate")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--mode",
    type=click.Choice(["baseline", "extended"]),
    default="baseline",
    show_default=True,
)
@click.option(
    "--validate-all",
    is_flag=True,
    help="Validate all rows, not just those marked with '+'.",
)
@click.option(
    "--warn-undecided/--no-warn-undecided",
    default=True,
    help="Warn about rows with blank Valid.",
)
@click.option("--verbose", is_flag=True, help="Show detailed error messages.")
def validate(input_file, mode, validate_all, warn_undecided, verbose):
    """Validate ConLoan XLSX for tag consistency and numbering."""
    results = validate_file(input_file, mode, validate_all, warn_undecided)

    error_rows = [r for r in results if any(e.severity == "error" for e in r.errors)]
    warn_rows = [r for r in results if all(e.severity == "warning" for e in r.errors)]

    if not error_rows:
        click.secho("Validation Passed.", fg="green")
        if warn_rows:
            click.echo(f"{len(warn_rows)} rows have warnings (use --verbose to see).")
    else:
        click.secho(f"Validation Failed: {len(error_rows)} rows have errors.", fg="red")

    if verbose or error_rows:
        for result in results:
            for err in result.errors:
                if err.severity == "error" or verbose:
                    click.echo(f"Row {result.row_index} [{err.rule_id}] {err.field}: {err.message}")

    if not verbose and error_rows:
        click.echo("Use --verbose to see all error details.")

if __name__ == "__main__":
    validate()
