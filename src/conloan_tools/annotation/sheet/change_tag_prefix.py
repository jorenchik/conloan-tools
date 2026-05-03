import re
import click
import pandas as pd
from .excel import write_sheet, DEFAULT_ANNOTATION_COLUMNS

def remap_tag_in_text(text: str, source_prefix: str, target_prefix: str) -> str:
    """Replaces <A1>... </A1> with <C1>... </C1>."""
    if not isinstance(text, str) or pd.isna(text):
        return text

    # Matches <PREFIX123> or </PREFIX123>
    pattern = rf"<(/)?{re.escape(source_prefix)}(\d+)>"
    replacement = rf"<\1{target_prefix}\2>"
    
    return re.sub(pattern, replacement, text)

@click.command("change-tag-prefix")
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("column_name")
@click.argument("source_prefix")
@click.argument("target_prefix")
@click.option("--output", "-o", "output_path", type=click.Path(), required=True,
              help="Path to output file (.xlsx)")
def change_tag_prefix(input_file, column_name, source_prefix, target_prefix, output_path):
    """
    Replace all span types SOURCE_PREFIX in COLUMN_NAME to TARGET_PREFIX.
    Example: change-tag-prefix input.xlsx "Replacement sentence" CN CS -o output.xlsx
    """
    df = pd.read_excel(input_file)

    if column_name not in df.columns:
        raise click.ClickException(f"Column '{column_name}' not found in file.")

    df[column_name] = df[column_name].apply(
        lambda x: remap_tag_in_text(x, source_prefix, target_prefix)
    )

    write_sheet(df, output_path, DEFAULT_ANNOTATION_COLUMNS)
    click.echo(f"Successfully remapped <{source_prefix}> to <{target_prefix}> in '{column_name}'")
    click.echo(f"Output written to: {output_path}")

if __name__ == "__main__":
    change_tag_prefix()
