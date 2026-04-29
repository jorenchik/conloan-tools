from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class LoanLabel(IntEnum):
    O = 0
    B_LOAN = 1
    I_LOAN = 2
    B_CS = 3
    I_CS = 4
    B_NE = 5
    I_NE = 6

    @property
    def label(self) -> str:
        return self.name.replace("_", "-")


@dataclass(frozen=True)
class LabelSchema:
    name: str
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]
    report_labels: tuple[str, ...]
    primary_label: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label_to_id": self.label_to_id,
            "id_to_label": {str(k): v for k, v in self.id_to_label.items()},
            "report_labels": list(self.report_labels),
            "primary_label": self.primary_label,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LabelSchema":
        return cls(
            name=d["name"],
            label_to_id=d["label_to_id"],
            id_to_label={int(k): v for k, v in d["id_to_label"].items()},
            report_labels=tuple(d["report_labels"]),
            primary_label=d["primary_label"],
        )


def _make_schema(
    members: list[LoanLabel],
    name: str,
    report_labels: tuple[str, ...],
    primary_label: str,
) -> LabelSchema:
    label_to_id = {m.label: i for i, m in enumerate(members)}
    id_to_label = {i: m.label for i, m in enumerate(members)}
    return LabelSchema(
        name=name,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        report_labels=report_labels,
        primary_label=primary_label,
    )


SCHEMA_LOAN_ONLY = _make_schema(
    members=[LoanLabel.O, LoanLabel.B_LOAN, LoanLabel.I_LOAN],
    name="loan",
    report_labels=("LOAN",),
    primary_label="LOAN",
)

SCHEMA_CONTRASTIVE = _make_schema(
    members=[
        LoanLabel.O,
        LoanLabel.B_LOAN,
        LoanLabel.I_LOAN,
        LoanLabel.B_CS,
        LoanLabel.I_CS,
        LoanLabel.B_NE,
        LoanLabel.I_NE,
    ],
    name="contrastive",
    report_labels=("LOAN", "CS", "NE"),
    primary_label="LOAN",
)

SCHEMAS: dict[str, LabelSchema] = {
    "loan": SCHEMA_LOAN_ONLY,
    "contrastive": SCHEMA_CONTRASTIVE,
}

