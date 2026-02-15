from dataclasses import dataclass, fields
from typing import Any
import json
import re
from enum import IntEnum
from transformers import PreTrainedTokenizerFast, BatchEncoding
from datasets import Dataset
from datasets.formatting.formatting import LazyBatch
from pathlib import Path
from datasets import Dataset, DatasetDict


class LoanLabel(IntEnum):
    O = 0
    B_LOAN = 1
    I_LOAN = 2

    @property
    def label(self) -> str:
        return self.name.replace("_", "-")


LABEL_TO_ID = {l.label: l.value for l in LoanLabel}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

_TAG_PATTERN = re.compile(r"<L(\d+)>(.*?)</L\1>", re.DOTALL)
_WORD_PATTERN = re.compile(r"\S+")


@dataclass
class LoanwordEntry:
    source_annotated_loanwords: str
    source_annotated_loanwords_replaced: str
    target: str
    source_plain: str
    source_annotated_plain: str
    words_in_L_tags: str  # stored as JSON string
    words_in_N_tags: str  # stored as JSON string
    corresponding_words: str  # stored as JSON string

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoanwordEntry":
        return cls(
            source_annotated_loanwords=data[
                "source_annotated_loanwords"
            ],
            source_annotated_loanwords_replaced=data[
                "source_annotated_loanwords_replaced"
            ],
            target=data["target"],
            source_plain=data["source_plain"],
            source_annotated_plain=data["source_annotated_plain"],
            words_in_L_tags=json.dumps(
                data["words_in_L_tags"], ensure_ascii=False
            ),
            words_in_N_tags=json.dumps(
                data["words_in_N_tags"], ensure_ascii=False
            ),
            corresponding_words=json.dumps(
                data["corresponding_words"], ensure_ascii=False
            ),
        )

def build_and_save_splits(
    raw_data: list[LoanwordEntry],
    splits_dir: Path,
    seed: int = 42,
) -> DatasetDict:

    breakpoint()
    columns = {f.name: [] for f in fields(LoanwordEntry)}
    for e in raw_data:
        for f in fields(LoanwordEntry):
            columns[f.name].append(e[f.name])

    full = Dataset.from_dict(columns)

    train_rest = full.train_test_split(test_size=0.2, seed=seed)
    dev_test = train_rest["test"].train_test_split(
        test_size=0.5, seed=seed
    )

    splits = DatasetDict(
        {
            "train": train_rest["train"],
            "dev": dev_test["train"],
            "test": dev_test["test"],
        }
    )

    splits_dir.mkdir(parents=True, exist_ok=True)
    splits.save_to_disk(str(splits_dir))
    print(
        f"Splits saved to {splits_dir}  "
        f"(train={len(splits['train'])}, "
        f"dev={len(splits['dev'])}, "
        f"test={len(splits['test'])})"
    )
    return splits

def load_conloan(file_paths: list[str]) -> Dataset:
    entries: list[LoanwordEntry] = []
    for fp in file_paths:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            entries.extend(LoanwordEntry.from_dict(d) for d in data)

    columns = {f.name: [] for f in fields(LoanwordEntry)}
    for e in entries:
        for f in fields(LoanwordEntry):
            columns[f.name].append(getattr(e, f.name))

    return Dataset.from_dict(columns)


def _parse_loanword_spans(
    annotated: str,
) -> tuple[str, list[tuple[int, int]]]:
    """Strip <Ln>…</Ln> tags, return (plain_text, [(start, end), …])
    with character offsets into the reconstructed plain text."""
    spans: list[tuple[int, int]] = []
    parts: list[str] = []
    offset = 0
    last_end = 0

    for m in _TAG_PATTERN.finditer(annotated):
        before = annotated[last_end : m.start()]
        parts.append(before)
        offset += len(before)

        content = m.group(2)
        spans.append((offset, offset + len(content)))
        parts.append(content)
        offset += len(content)

        last_end = m.end()

    parts.append(annotated[last_end:])
    return "".join(parts), spans


def _word_char_spans(text: str) -> list[tuple[str, int, int]]:
    return [
        (m.group(), m.start(), m.end())
        for m in _WORD_PATTERN.finditer(text)
    ]


def _build_word_labels(
    word_spans: list[tuple[str, int, int]],
    loan_spans: list[tuple[int, int]],
) -> list[str]:
    labels = ["O"] * len(word_spans)

    for loan_start, loan_end in loan_spans:
        first = True
        for i, (_word, ws, we) in enumerate(word_spans):
            if ws >= loan_start and we <= loan_end:
                labels[i] = "B-LOAN" if first else "I-LOAN"
                first = False
    return labels


def tokenize_and_align_labels(
    examples: LazyBatch,
    tokenizer: PreTrainedTokenizerFast,
    label_to_id: dict[str, int] = LABEL_TO_ID,
) -> BatchEncoding:
    all_words: list[list[str]] = []
    all_word_labels: list[list[str]] = []

    for annotated in examples["source_annotated_loanwords"]:
        plain, loan_spans = _parse_loanword_spans(annotated)

        word_spans = _word_char_spans(plain)
        words = [w for w, _, _ in word_spans]
        word_labels = _build_word_labels(word_spans, loan_spans)

        all_words.append(words)
        all_word_labels.append(word_labels)

    tokenized_inputs = tokenizer(
        all_words,
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
    )

    labels: list[list[int]] = []
    for i, word_labels in enumerate(all_word_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids: list[int] = []
        prev_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx == prev_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(
                    label_to_id[word_labels[word_idx]]
                )
            prev_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
