import json
import re
import os
import click
import pandas as pd
from pathlib import Path
from conloan_tools.annotation.sheet.validate_sheet import validate_row
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

def _get_validated_df(input_xlsx, mode, all_rows):
    df = pd.read_excel(input_xlsx)
    if not all_rows and "Valid" in df.columns:
        initial_len = len(df)
        df = df[df["Valid"].astype(str).str.lower() == "+"].copy()
        click.echo(f"Filtered {len(df)}/{initial_len} valid rows.")
    
    valid_rows = []
    error_count = 0
    for idx, row in df.iterrows():
        errors = validate_row(row, mode)
        if errors:
            error_count += 1
            click.secho(f"Row {idx+2} invalid: {' | '.join(errors)}", fg="red")
            continue
        valid_rows.append(row)
    
    if error_count:
        click.secho(f"Warning: {error_count} rows skipped.", fg="yellow")
    return valid_rows

@click.command("json-from-sheet")
@click.option("--output", "-o", "output_json", type=click.Path(), required=True, help="Path to output JSON file.")
@click.argument("input_xlsx", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("--mode", type=click.Choice(["baseline", "extended"]), default="baseline")
@click.option("--all-rows", is_flag=True)
def json_from_sheet(output_json, input_xlsx, mode, all_rows):
    """Transform annotated XLSX sheets into a validated JSON dataset."""
    rows = []
    for path in input_xlsx:
        rows.extend(_get_validated_df(path, mode, all_rows))
    dataset = []

    for row in rows:
        l_sent, n_sent = str(row.get("Label sentence", "")), str(row.get("Replacement sentence", ""))
        l_tags, n_tags = extract_tags(l_sent, "L"), extract_tags(n_sent, "N")
        
        dataset.append({
            "source_annotated_loanwords": l_sent,
            "source_annotated_loanwords_replaced": n_sent,
            "target": str(row.get("Target", "")),
            "source_plain": strip_tags(l_sent),
            "source_annotated_plain": strip_tags(n_sent),
            "words_in_L_tags": l_tags,
            "words_in_N_tags": n_tags,
            "corresponding_words": {k: [l_tags[k], n_tags[k]] for k in l_tags if k in n_tags}
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    click.secho(f"Success: {len(dataset)} entries -> {output_json}", fg="green")

@click.command("replacements-from-sheet")
@click.option("--output", "-o", "output_tsv", type=click.Path(), required=True, help="Path to output TSV file.")
@click.argument("input_xlsx", nargs=-1, type=click.Path(exists=True), required=True)
@click.option("--all", "include_all", is_flag=True, help="Include non-changed words.")
@click.option("--mode", type=click.Choice(["baseline", "extended"]), default="baseline")
@click.option("--all-rows", is_flag=True)
def replacements_from_sheet(output_tsv, input_xlsx, include_all, mode, all_rows):
    """Extract word-level replacements to a TSV file."""
    rows = []
    for path in input_xlsx:
        rows.extend(_get_validated_df(path, mode, all_rows))
    pairs = []

    for row in rows:
        l_tags = extract_tags(str(row.get("Label sentence", "")), "L")
        n_tags = extract_tags(str(row.get("Replacement sentence", "")), "N")
        origin = str(row.get("Origin", ""))
        
        for k in l_tags:
            if k in n_tags:
                src, repl = l_tags[k], n_tags[k]
                if include_all or src != repl:
                    pairs.append((src, repl, origin))

    df_repl = pd.DataFrame(pairs, columns=["source", "replacement", "origin"])
    df_repl.to_csv(output_tsv, sep="\t", index=False, header=False)
    click.secho(f"Success: {len(df_repl)} pairs -> {output_tsv}", fg="green")

if __name__ == "__main__":
    annotation()
