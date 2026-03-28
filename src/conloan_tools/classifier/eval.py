from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    import numpy as np
    from datasets import Dataset
    from transformers import AutoModelForTokenClassification, AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_label_maps() -> tuple[dict[str, int], dict[int, str]]:
    from conloan_tools.classifier.data import LoanLabel
    return LoanLabel.label_to_id(), LoanLabel.id_to_label()


def _load_split(splits_dir: str, split: str) -> Dataset:
    from datasets import DatasetDict

    splits = DatasetDict.load_from_disk(splits_dir)
    if split not in splits:
        raise click.BadParameter(
            f"Split '{split}' not found in {list(splits.keys())}"
        )
    return splits[split]


def _tokenize(
    dataset: Dataset,
    model_path: str,
    label_to_id: dict[str, int],
) -> tuple[AutoTokenizer, Dataset]:
    from transformers import AutoTokenizer

    from conloan_tools.classifier.data import tokenize_and_align_labels

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenized = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id),
        batched=True,
    )
    return tokenizer, tokenized


def _predict(
    tokenized: Dataset,
    model_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np
    import torch
    from transformers import (
        AutoModelForTokenClassification,
        DataCollatorForTokenClassification,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model: AutoModelForTokenClassification = (
        AutoModelForTokenClassification.from_pretrained(model_path)
    )
    model.to(device)
    model.eval()

    keep = {"input_ids", "attention_mask", "labels"}
    drop = [c for c in tokenized.column_names if c not in keep]
    collator = DataCollatorForTokenClassification(tokenizer)
    dataloader = torch.utils.data.DataLoader(
        tokenized.remove_columns(drop),
        batch_size=batch_size,
        collate_fn=collator,
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_preds.append(outputs.logits.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

    return np.concatenate(all_preds, axis=0), np.concatenate(all_labels, axis=0)


def _align_predictions(
    predictions: np.ndarray,
    label_ids: np.ndarray,
    id_to_label: dict[int, str],
) -> tuple[list[list[str]], list[list[str]]]:
    import numpy as np

    preds = np.argmax(predictions, axis=2)
    true_labels, pred_labels = [], []

    for pred_seq, label_seq in zip(preds, label_ids):
        true_seq, pred_seq_out = [], []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            true_seq.append(id_to_label[l])
            pred_seq_out.append(id_to_label[p])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_out)

    return true_labels, pred_labels


def _compute_metrics(
    true_labels: list[list[str]],
    pred_labels: list[list[str]],
) -> tuple[str, float, float, float]:
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    report = classification_report(true_labels, pred_labels, digits=4)
    f1 = f1_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels)
    rec = recall_score(true_labels, pred_labels)
    return report, prec, rec, f1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--splits-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing saved train/dev/test splits.",
)
@click.option(
    "--split",
    default="test",
    show_default=True,
    type=click.Choice(["train", "dev", "test"]),
    help="Which split to evaluate on.",
)
@click.option(
    "--model",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Path to trained model checkpoint.",
)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option(
    "--output",
    default=None,
    type=click.Path(dir_okay=False),
    help="Optional JSON output path.",
)
@click.option(
    "--inputs", "-i",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="JSON dataset files — required when --rebuild-splits is set.",
)
@click.option(
    "--rebuild-splits",
    is_flag=True,
    help="Rebuild splits from --inputs before evaluating.",
)
@click.option("--seed", type=int, default=42, show_default=True)
def evaluate(
    splits_dir: str,
    split: str,
    model: str,
    batch_size: int,
    output: str | None,
    inputs: tuple[str, ...],
    rebuild_splits: bool,
    seed: int,
) -> None:
    import json

    label_to_id, id_to_label = _build_label_maps()

    if rebuild_splits:
        if not inputs:
            raise click.UsageError("--inputs is required when --rebuild-splits is set.")
        from conloan_tools.classifier.data import build_and_save_splits, load_conloan
        click.echo("Rebuilding splits...")
        raw_data = load_conloan(list(inputs))
        build_and_save_splits(raw_data, Path(splits_dir), seed=seed)

    dataset = _load_split(splits_dir, split)
    click.echo(f"Evaluating on '{split}' split ({len(dataset)} examples)")

    tokenizer, tokenized = _tokenize(dataset, model, label_to_id)
    raw_preds, raw_labels = _predict(tokenized, model, tokenizer, batch_size)

    true_labels, pred_labels = _align_predictions(raw_preds, raw_labels, id_to_label)
    report, prec, rec, f1 = _compute_metrics(true_labels, pred_labels)

    click.echo(report)
    click.echo(f"Overall  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

    if output:
        results = {
            "split": split,
            "num_examples": len(dataset),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "report": report,
        }
        Path(output).write_text(json.dumps(results, indent=2))
        click.echo(f"Results written to {output}")


if __name__ == "__main__":
    evaluate()
