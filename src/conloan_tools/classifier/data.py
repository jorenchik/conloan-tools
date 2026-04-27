from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset
    from datasets.formatting.formatting import LazyBatch
    from transformers import BatchEncoding, PreTrainedTokenizerFast

    from .schema import LabelSchema

# Tag patterns: <L1>…</L1>, <CS1>…</CS1>, <NE1>…</NE1>, etc.
# Map tag-name prefix → entity name used in label schema (e.g. "L" → "LOAN")
_TAG_PREFIX_TO_ENTITY: dict[str, str] = {
    "L": "LOAN",
    "CS": "CS",
    "NE": "NE",
}

# Matches any supported tag: captures (prefix, index, content)
_TAG_PATTERN = re.compile(
    r"<(L|CS|NE)(\d+)>(.*?)</(L|CS|NE)\2>",
    re.DOTALL,
)
_WORD_PATTERN = re.compile(r"\S+")


# ---------------------------------------------------------------------------
# Conloan data model
# ---------------------------------------------------------------------------


@dataclass
class LoanwordEntry:
    source_annotated_loanwords: str
    source_annotated_loanwords_replaced: str
    target: str
    source_plain: str
    source_annotated_plain: str
    words_in_L_tags: list[Any]
    words_in_N_tags: list[Any]
    corresponding_words: list[Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoanwordEntry":
        return cls(
            source_annotated_loanwords=data["source_annotated_loanwords"],
            source_annotated_loanwords_replaced=data[
                "source_annotated_loanwords_replaced"
            ],
            target=data["target"],
            source_plain=data["source_plain"],
            source_annotated_plain=data["source_annotated_plain"],
            words_in_L_tags=data["words_in_L_tags"],
            words_in_N_tags=data["words_in_N_tags"],
            corresponding_words=data["corresponding_words"],
        )


def load_conloan(file_paths: list[str]) -> "Dataset":
    from datasets import Dataset

    entries: list[LoanwordEntry] = []
    for fp in file_paths:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            entries.extend(LoanwordEntry.from_dict(d) for d in data)

    return Dataset.from_list([asdict(e) for e in entries])


# ---------------------------------------------------------------------------
# Span parsing
# ---------------------------------------------------------------------------


def _parse_all_spans(
    annotated: str,
) -> tuple[str, list[tuple[str, int, int]]]:
    """Strip all supported tags, return (plain_text, [(entity, start, end), …]).

    Entity names follow _TAG_PREFIX_TO_ENTITY (e.g. "L" → "LOAN").
    Character offsets are into the reconstructed plain text.
    Overlapping spans are not supported and will produce undefined label order.
    """
    plain = _TAG_PATTERN.sub(r"\3", annotated)

    spans: list[tuple[str, int, int]] = []
    offset = 0
    last_plain_end = 0

    for m in _TAG_PATTERN.finditer(annotated):
        prefix = m.group(1)
        content = m.group(3)
        entity = _TAG_PREFIX_TO_ENTITY.get(prefix, prefix)

        between = _TAG_PATTERN.sub(r"\3", annotated[last_plain_end : m.start()])
        offset += len(between)
        spans.append((entity, offset, offset + len(content)))
        offset += len(content)
        last_plain_end = m.end()

    return plain, spans


def _word_char_spans(text: str) -> list[tuple[str, int, int]]:
    return [
        (m.group(), m.start(), m.end())
        for m in _WORD_PATTERN.finditer(text)
    ]


def _build_word_labels(
    word_spans: list[tuple[str, int, int]],
    entity_spans: list[tuple[str, int, int]],
    schema: "LabelSchema",
) -> list[str]:
    """Assign BIO labels to each word given entity spans and a label schema.

    Words that fall inside an entity span get B-<entity> or I-<entity>.
    Entities whose B-/I- labels are absent from the schema are silently skipped,
    which lets SCHEMA_LOAN_ONLY safely ignore CS/NE spans.
    """
    labels = ["O"] * len(word_spans)

    for entity, span_start, span_end in entity_spans:
        b_label = f"B-{entity}"
        i_label = f"I-{entity}"
        if b_label not in schema.label_to_id:
            # This entity is not part of the active schema — ignore.
            continue
        first = True
        for i, (_word, ws, we) in enumerate(word_spans):
            if ws < span_end and we > span_start:  # overlap
                labels[i] = b_label if first else i_label
                first = False

    return labels


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def tokenize_and_align_labels(
    examples: "LazyBatch",
    tokenizer: "PreTrainedTokenizerFast",
    schema: "LabelSchema",
    max_length: int = 128,
    padding: str = "max_length",
    truncation: bool = True,
    word_level: bool = True,
) -> "BatchEncoding":
    """Tokenize a batch and align BIO labels to subword tokens.

    word_level=True  → only the first subword of each word gets a label;
                       subsequent subwords are masked with -100.
    word_level=False → all subwords of a word share the same label id.
    """
    if not tokenizer.is_fast:
        raise ValueError("tokenize_and_align_labels requires a fast tokenizer.")

    all_words: list[list[str]] = []
    all_word_labels: list[list[str]] = []

    for annotated in examples["source_annotated_loanwords"]:
        plain, entity_spans = _parse_all_spans(annotated)
        word_spans = _word_char_spans(plain)
        words = [w for w, _, _ in word_spans]
        word_labels = _build_word_labels(word_spans, entity_spans, schema)
        all_words.append(words)
        all_word_labels.append(word_labels)

    tokenized_inputs = tokenizer(
        all_words,
        truncation=truncation,
        is_split_into_words=True,
        padding=padding,
        max_length=max_length,
    )

    label_ids_batch: list[list[int]] = []
    for i, word_labels in enumerate(all_word_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids: list[int] = []
        prev_word_idx: int | None = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx == prev_word_idx:
                label_ids.append(
                    -100
                    if word_level
                    else schema.label_to_id[word_labels[word_idx]]
                )
            else:
                label_ids.append(schema.label_to_id[word_labels[word_idx]])
            prev_word_idx = word_idx

        label_ids_batch.append(label_ids)

    tokenized_inputs["labels"] = label_ids_batch
    return tokenized_inputs

