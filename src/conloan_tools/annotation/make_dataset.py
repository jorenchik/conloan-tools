import json
import re
import os
import click
import pandas as pd
from pathlib import Path
from .validate_sheet import validate_row
from conloan_tools.annotation import annotation

def strip_tags(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"</?[A-Z]+\d+>", "", text)

def extract_tags(text: str, prefix: str) -> dict[str, str]:
    if not isinstance(text, str):
        return {}
    pattern = rf"<{prefix}(\d+)>([^<]+)</{prefix}\1>"
    return {m[0]: m[1] for m in re.findall(pattern, text)}

@click.command("make")
@click.argument("lang_name")
@click.argument("input_xlsx", type=click.Path(exists=True))
@click.option(
    "--output-dir",
    default="./output",
    show_default=True,
    help="Target directory for JSON."
)
@click.option(
    "--mode",
    type=click.Choice(["baseline", "extended"]),
    default="baseline",
    show_default=True
)
@click.option(
    "--all-rows",
    is_flag=True,
    help="Process all rows regardless of the 'Valid' column status."
)
def make_dataset(lang_name, input_xlsx, output_dir, mode, all_rows):
    """Transform an annotated XLSX sheet into a validated JSON dataset."""
    output_path = Path(output_dir) / f"{lang_name}.json"

    if output_path.exists():
        if not click.confirm(f"Overwrite {output_path}?"):
            click.echo("Aborted.")
            return

    df = pd.read_excel(input_xlsx)
    
    # Filter for rows marked 'x' unless --all-rows is specified
    if not all_rows and "Valid" in df.columns:
        initial_len = len(df)
        df = df[df["Valid"].astype(str).str.lower() == "+"]
        click.echo(f"Filtered {len(df)}/{initial_len} valid rows.")

    dataset = []
    error_count = 0

    for idx, row in df.iterrows():
        # Validate using existing logic
        errors = validate_row(row, mode)
        if errors:
            error_count += 1
            click.secho(f"Row {idx+2} invalid: {' | '.join(errors)}", fg="red")
            continue

        l_sent = str(row.get("Label sentence", ""))
        n_sent = str(row.get("Replacement sentence", ""))
        target = str(row.get("Target", ""))

        l_tags = extract_tags(l_sent, "L")
        n_tags = extract_tags(n_sent, "N")

        # Construct specific JSON schema
        entry = {
            "source_annotated_loanwords": l_sent,
            "source_annotated_loanwords_replaced": n_sent,
            "target": target,
            "source_plain": strip_tags(l_sent),
            "source_annotated_plain": strip_tags(n_sent),
            "words_in_L_tags": l_tags,
            "words_in_N_tags": n_tags,
            "corresponding_words": {
                k: [l_tags[k], n_tags[k]] for k in l_tags if k in n_tags
            }
        }
        dataset.append(entry)

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    click.secho(f"Success: {len(dataset)} entries written to {output_path}", fg="green")
    if error_count:
        click.secho(f"Warning: {error_count} rows skipped due to validation errors.", fg="yellow")

if __name__ == "__main__":
    annotation()
