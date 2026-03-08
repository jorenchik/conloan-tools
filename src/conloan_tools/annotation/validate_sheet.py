import re
import click
import pandas as pd
from typing import List, Set, Counter
from collections import defaultdict

def get_tag_stats(text: str, label: str) -> tuple[List[str], List[str]]:
    """Returns list of opening tags and list of errors regarding malformed tags."""
    if not isinstance(text, str) or pd.isna(text):
        return [], []

    if "\n" in text or "\r" in text:
        return [], [f"{label}: Sentence contains forbidden line breaks."]
    
    errors = []
    open_tags = re.findall(r"<([A-Z]+\d+)>", text)
    
    # Find all tags including illegal ones (e.g., <UNK>) to detect malformed structures
    stack = []  # Stores (tag_id, full_match_string)
    for tag_match in re.finditer(r"<(/?)([A-Z]+\d*)>", text):
        full_tag = tag_match.group(0)
        is_closing = tag_match.group(1) == "/"
        tag_id = tag_match.group(2)
        # full_tag = tag_match.group(0)
        # full_content = tag_match.group(1)
        # is_closing = tag_match.group(1) == "/"
        # is_closing = full_content.startswith("/")
        # tag_id = full_content.lstrip("/")

        if not is_closing:
            if stack:
                errors.append(f"{label}: Nested tags are forbidden (<{tag_id}> inside <{stack[-1]}>)")
            stack.append((tag_id, tag_match.group(0)))
        else:
            if not stack:
                errors.append(f"{label}: Closing tag </{tag_id}> has no opening tag.")
            else:
                last_open_id, last_open_full = stack.pop()
                if last_open_id != tag_id:
                    errors.append(
                        f"{label}: Tag {last_open_full} closed by incorrect tag </{tag_id}>."
                    )

    # Remaining items in stack are unclosed tags
    for unclosed_id, unclosed_full in stack:
        errors.append(f"{label}: Tag {unclosed_full} is never closed.")

    # 3. Punctuation inside tags: <L1>word.</L1>
    bad_internal_punct = re.findall(r"<([A-Z]+\d+)>.*?[,.!?;]</\1>", text)
    if bad_internal_punct:
        errors.append(f"{label}: Punctuation caught inside tags: {bad_internal_punct}")

    # 4. Fused tags: </L1>word instead of </L1> word
    fused_tags = re.findall(r"</[A-Z]+\d+>[^\s,.!?;\"')\]}]", text)
    if fused_tags:
        errors.append(f"{label}: Missing space/punctuation after closing tag: {fused_tags}")

    return open_tags, errors

def validate_row(row: pd.Series, mode: str) -> List[str]:
    errors = []
    loan_sent = str(row.get("Loanword sentence", ""))
    native_sent = str(row.get("Native sentence", ""))
    target = str(row.get("Target", ""))

    # 1. Target purity
    if re.search(r"</?[A-Z]+\d+>", target):
        errors.append("Target contains tags.")

    # Extract tags and check balance
    loan_tags, l_structural_errs = get_tag_stats(loan_sent, "Loanword")
    native_tags, n_structural_errs = get_tag_stats(native_sent, "Native")
    errors.extend(l_structural_errs + n_structural_errs)

    # 3. Define schema
    if mode == "base":
        l_allowed, n_allowed = {"L"}, {"N"}
    else:  # code_switch
        l_allowed, n_allowed = {"L", "CS"}, {"N", "CN"}

    # 4. Global illegal tag check (catches <UNK>, <FOO1>, etc.)
    all_found_tags = re.findall(r"</?([A-Z]+\d*)>", loan_sent + native_sent)
    all_allowed = {"L", "N", "CS", "CN"} if mode != "base" else {"L", "N"}
    
    for t in all_found_tags:
        prefix = re.match(r"([A-Z]+)", t).group(1)
        if prefix not in all_allowed:
            errors.append(f"Illegal tag detected: <{t}>")

    def check_tag_logic(tags: List[str], allowed: Set[str], label: str):
        counts = Counter(tags)
        prefix_map = defaultdict(list)
        for tag, count in counts.items():
            match = re.match(r"([A-Z]+)(\d+)", tag)
            if not match: continue
            prefix, num = match.group(1), int(match.group(2))
            prefix_map[prefix].append(num)
            if count > 1:
                errors.append(f"{label} contains duplicate tag: {tag}")

        # 5. Incremental sequence check (must start at 1 and have no gaps)
        for pref, nums in prefix_map.items():
            sorted_nums = sorted(nums)
            expected = list(range(1, len(nums) + 1))
            if sorted_nums != expected:
                errors.append(
                    f"{label} {pref} tags must start at 1 and be incremental. "
                    f"Found: {sorted_nums}"
                )

        return set(tags)

    l_set = check_tag_logic(loan_tags, l_allowed, "Loanword")
    n_set = check_tag_logic(native_tags, n_allowed, "Native")

    # 5. Structural Parity (L <-> N and CS <-> CN)
    # Rules: For every <LX> there is <NX>. For every <CSX> there is <CNX>.
    mapping = {"L": "N", "CS": "CN"}
    
    # 6. Parity Count Check (Simplified)
    l_prefixes = Counter(re.match(r"([A-Z]+)", t).group(1) for t in l_set)
    n_prefixes = Counter(re.match(r"([A-Z]+)", t).group(1) for t in n_set)

    for l_pref, n_pref in mapping.items():
        if l_prefixes[l_pref] != n_prefixes[n_pref]:
            errors.append(
                f"Count mismatch: {l_pref} tags ({l_prefixes[l_pref]}) vs "
                f"{n_pref} tags ({n_prefixes[n_pref]})"
            )

    return errors

@click.command("validate")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--mode",
    type=click.Choice(["base", "code_switch"]),
    default="base",
    show_default=True,
)
@click.option(
    "--only-valid",
    is_flag=True,
    help="Only validate rows where 'Valid' column is 'x'.",
)
@click.option("--verbose", is_flag=True, help="Show error details.")
def validate(input_file, mode, only_valid, verbose):
    """Validate ConLoan XLSX for tag consistency and numbering."""
    df = pd.read_excel(input_file)
    
    if only_valid:
        initial_count = len(df)
        df = df[df["Valid"].astype(str).str.lower() == "x"]
        click.echo(f"Filtering: {len(df)}/{initial_count} rows are marked 'x'.")

    error_log = []
    for idx, row in df.iterrows():
        row_errors = validate_row(row, mode)
        if row_errors:
            error_log.append((idx + 2, row_errors))

    if not error_log:
        click.secho("Validation Passed.", fg="green")
    else:
        click.secho(f"Validation Failed: {len(error_log)} rows have errors.", fg="red")
        for line_num, errors in error_log:
            msg = f"Row {line_num}: " + " | ".join(errors)
            if verbose:
                click.echo(msg)
        if not verbose:
            click.echo("Use --verbose to see error details.")

if __name__ == "__main__":
    validate()
