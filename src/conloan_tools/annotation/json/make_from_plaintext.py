import json
import click
from pathlib import Path

@click.command("make-from-plaintext")
@click.option(
    "--output",
    "-o",
    "output_json",
    type=click.Path(),
    required=True,
    help="Path to output JSON file.",
)
@click.argument("input_txt", type=click.Path(exists=True), required=True)
def make_from_plaintext(output_json, input_txt):
    """
    Create a JSON dataset from a plaintext file where each line is a source sentence.
    Useful for creating ad hoc test cases.
    """
    dataset = []

    with open(input_txt, "r", encoding="utf-8") as f:
        # Strip whitespace and ignore empty lines
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        dataset.append(
            {
                "source_annotated_loanwords": line,
                "source_annotated_loanwords_replaced": "",
                "target": "",
                "source_plain": "",
                "source_annotated_plain": "",
                "words_in_L_tags": {},
                "words_in_N_tags": {},
                "corresponding_words": {},
            }
        )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    click.secho(
        f"Success: {len(dataset)} entries created -> {output_json}", fg="green"
    )

if __name__ == "__main__":
    make_from_plaintext()
