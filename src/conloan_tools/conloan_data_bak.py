
from dataclasses import dataclass
from typing import Any
import json
from enum import IntEnum
from transformers import PreTrainedTokenizerFast, BatchEncoding

class LoanLabel(IntEnum):
    O = 0
    B_LOAN = 1
    I_LOAN = 2

    @property
    def label(self) -> str:
        """Returns the identifier with hyphens for external compatibility."""
        return self.name.replace("_", "-")

@dataclass
class LoanwordEntry:
    source_plain: str
    words_in_l: dict[str, str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoanwordEntry":
        return cls(
            source_plain=data["source_plain"],
            words_in_l=data.get("words_in_L_tags", {}),
        )

def load_conloan(file_paths: list[str]) -> list[LoanwordEntry]:
    entries = []
    for fp in file_paths:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict): data = [data]
            entries.extend([LoanwordEntry.from_dict(d) for d in data])
    return entries

def tokenize_and_align_labels(
    examples: list[LoanwordEntry],
    tokenizer: PreTrainedTokenizerFast,
    label_to_id: dict[str, int]
) -> BatchEncoding:

    # ConLoan words are space-separated in source_plain
    tokenized_inputs = tokenizer(
        [ex.split() for ex in examples["source_plain"]],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
    )

    labels = []
    for i, words_list in enumerate(examples["source_plain"]):
        word_list = words_list.split()
        entry_l_tags = examples["words_in_l"][i]
        
        # Simple BIO logic: if word in L_tags, mark B-LOAN (simplified for example)
        word_labels = [label_to_id["B-LOAN"] if w.strip(".,!?;:") in entry_l_tags 
                       else label_to_id["O"] for w in word_list]
        
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None or word_idx == prev_word_idx:
                label_ids.append(-100) # Paper 3.3: ignore subwords/special
            else:
                label_ids.append(word_labels[word_idx])
            prev_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
