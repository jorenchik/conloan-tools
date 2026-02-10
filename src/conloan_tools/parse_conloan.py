import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any
from pathlib import Path


@dataclass
class LoanwordEntry:
    source_plain: str
    source_annotated_loanwords: str
    source_annotated_loanwords_replaced: str
    target: str
    words_in_l: Dict[str, str]
    words_in_n: Dict[str, str]
    corresponding_words: Dict[str, List[str]]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoanwordEntry":
        return cls(
            source_plain=data["source_plain"],
            source_annotated_loanwords=data["source_annotated_loanwords"],
            source_annotated_loanwords_replaced=data[
                "source_annotated_loanwords_replaced"
            ],
            target=data["target"],
            words_in_l=data["words_in_L_tags"],
            words_in_n=data["words_in_N_tags"],
            corresponding_words=data["corresponding_words"],
        )


def load_datasets(files: List[str]) -> List[LoanwordEntry]:
    all_entries = []
    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File {file_path} not found. Skipping.")
            continue

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # Handle both single-object and list-of-objects JSON files
            if isinstance(data, dict):
                data = [data]
            all_entries.extend([LoanwordEntry.from_dict(item) for item in data])
    return all_entries


def main():
    parser = argparse.ArgumentParser(
        description="Parse loanword datasets into Python dataclasses."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Path to one or more JSON dataset files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print summary of loaded data.",
    )

    args = parser.parse_args()
    dataset = load_datasets(args.inputs)

    if args.verbose:
        print(f"Successfully loaded {len(dataset)} entries.")
        if dataset:
            print(f"Sample source: {dataset[0].source_plain[:50]}...")


if __name__ == "__main__":
    main()
