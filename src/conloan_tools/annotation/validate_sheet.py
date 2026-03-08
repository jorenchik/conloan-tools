import re
import click
import pandas as pd
from typing import List, Set, Counter
from collections import defaultdict

def get_tag_stats(text: str, label: str) -> tuple[List[str], List[str]]:
    """Returns list of opening tags and list of errors regarding malformed tags."""
    if not isinstance(text, str) or pd.isna(text):
        return [], []
    
    errors = []
    # Find all opening and closing tags
    open_tags = re.findall(r"<([A-Z]+\d+)>", text)
    close_tags = re.findall(r"</([A-Z]+\d+)>", text)
    
    # 1. Structural check: Every opening tag must have a matching closing tag
    if len(open_tags) != len(close_tags):
        errors.append(f"{label}: Unbalanced tags (Open: {len(open_tags)}, Close: {len(close_tags)})")
    
    # 2. Correct nesting/closure check (simple sequence check)
    all_tags = re.findall(r"</?([A-Z]+\d+)>", text)
    stack = []
    for tag_match in re.finditer(r"<(/?)([A-Z]+\d+)>", text):
        is_closing = tag_match.group(1) == "/"
        tag_id = tag_match.group(2)
        if not is_closing:
            stack.append(tag_id)
        else:
            if not stack or stack.pop() != tag_id:
                errors.append(f"{label}: Tag <{tag_id}> closed incorrectly or out of order.")
                
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

    # 4. Check for duplicates and illegal types
    def check_tags(tags: List[str], allowed: Set[str], label: str):
        counts = Counter(tags)
        prefix_map = defaultdict(list)

        for tag, count in counts.items():
            prefix = re.match(r"([A-Z]+)", tag).group(1)
            num = int(tag.replace(prefix, ""))
            prefix_map[prefix].append(num)

            if prefix not in allowed:
                errors.append(f"{label} contains illegal tag type: {tag}")
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

    l_set = check_tags(loan_tags, l_allowed, "Loanword")
    n_set = check_tags(native_tags, n_allowed, "Native")

    # 5. Structural Parity (L <-> N and CS <-> CN)
    # Rules: For every <LX> there is <NX>. For every <CSX> there is <CNX>.
    mapping = {"L": "N", "CS": "CN"}
    
    # Check Loan -> Native
    for l_tag in l_set:
        prefix = re.match(r"([A-Z]+)", l_tag).group(1)
        num = l_tag.replace(prefix, "")
        expected_n = f"{mapping[prefix]}{num}"
        if expected_n not in n_set:
            errors.append(f"Missing corresponding tag {expected_n} in Native")

    # Check Native -> Loan (orphans)
    for n_tag in n_set:
        prefix = re.match(r"([A-Z]+)", n_tag).group(1)
        # Reverse mapping search
        inv_map = {v: k for k, v in mapping.items()}
        num = n_tag.replace(prefix, "")
        expected_l = f"{inv_map[prefix]}{num}"
        if expected_l not in l_set:
            errors.append(f"Orphan tag {n_tag} in Native (no {expected_l})")

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
