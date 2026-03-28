from __future__ import annotations

import click
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

from typing import TYPE_CHECKING
from conloan_tools.classifier.data import (
    load_conloan,
    LoanwordEntry,
    LoanLabel,
    tokenize_and_align_labels,
    build_and_save_splits,
)

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict
    from transformers import AutoModelForTokenClassification, AutoTokenizer


# ---------------------------------------------------------------------------
# Splits helpers
# ---------------------------------------------------------------------------

def _splits_are_valid(splits_dir: Path) -> bool:
    required = {"train", "dev", "test", "dataset_info.json"}
    return splits_dir.exists() and required.issubset(
        {p.name for p in splits_dir.iterdir()}
    )


def _load_or_build_splits(
    inputs: list[str],
    splits_dir: Path,
    seed: int,
    rebuild: bool,
) -> DatasetDict:
    if not rebuild and _splits_are_valid(splits_dir):
        from datasets import DatasetDict as _DatasetDict

        click.echo(f"Loading existing splits from {splits_dir}")
        return _DatasetDict.load_from_disk(str(splits_dir))

    if rebuild:
        click.echo("Rebuilding splits (--rebuild-splits set)")
    else:
        click.echo(f"No valid splits found at {splits_dir}, building...")

    raw_data = load_conloan(inputs)
    return build_and_save_splits(raw_data, splits_dir, seed=seed)


def _generate_run_name(model: str, eval_split: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_slug = model.split("/")[-1]
    return f"{ts}_{model_slug}_eval-{eval_split}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DRY_RUN_MODEL = "prajjwal1/bert-tiny"
DEFAULT_MODEL = "distilbert-base-multilingual-cased"


@click.command()
@click.option(
    "--inputs", "-i",
    required=True,
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="JSON dataset files (repeatable).",
)
@click.option(
    "--splits-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory to save/load train/dev/test splits.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory for checkpoints and final model.",
)
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option("--run-name", default=None, help="Run name; auto-generated if omitted.")
@click.option("--epochs", type=int, default=3, show_default=True)
@click.option("--learning-rate", type=float, default=5e-5, show_default=True)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--weight-decay", type=float, default=0.01, show_default=True)
@click.option("--eval-split", default="dev", show_default=True, help="Split key to evaluate on.")
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--rebuild-splits", is_flag=True, help="Force rebuild splits.")
@click.option(
    "--keep-best-only",
    is_flag=True,
    help="Delete intermediate checkpoints, retain only the best epoch.",
)
@click.option(
    "--max-samples",
    type=int,
    default=None,
    help="Cap each split to N samples (smoke-test / dry-run).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help=f"Shorthand: switches to {DRY_RUN_MODEL}, 1 epoch, --max-samples 64.",
)
def train(
    inputs: tuple[str, ...],
    splits_dir: str,
    output_dir: str,
    model: str,
    run_name: str | None,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    eval_split: str,
    seed: int,
    rebuild_splits: bool,
    keep_best_only: bool,
    max_samples: int | None,
    dry_run: bool,
) -> None:
    # --- dry-run overrides ---------------------------------------------------
    if dry_run:
        if model == DEFAULT_MODEL:
            model = DRY_RUN_MODEL
        epochs = 1
        if max_samples is None:
            max_samples = 64
        click.echo(
            f"[dry-run] model={model}, epochs={epochs}, max_samples={max_samples}",
            err=True,
        )

    if run_name is None:
        run_name = _generate_run_name(model, eval_split)
    click.echo(f"Run: {run_name}")

    _splits_dir = Path(splits_dir)
    _output_dir = Path(output_dir) / run_name
    _output_dir.mkdir(parents=True, exist_ok=True)

    # --- heavy imports -------------------------------------------------------
    import torch
    from datasets import DatasetDict
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        TrainingArguments,
    )
    from transformers.trainer import Trainer

    # 1. Labels
    label_to_id = LoanLabel.label_to_id()

    # 2. Splits
    splits = _load_or_build_splits(
        list(inputs), _splits_dir, seed=seed, rebuild=rebuild_splits
    )

    if max_samples is not None:
        splits = DatasetDict(
            {k: v.select(range(min(max_samples, len(v)))) for k, v in splits.items()}
        )

    if eval_split not in splits:
        raise click.BadParameter(
            f"--eval-split '{eval_split}' not found in splits {list(splits.keys())}"
        )

    # 3. Tokenize
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model)
    tokenized_splits = splits.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id),
        batched=True,
    )

    # 4. Model
    clf_model: AutoModelForTokenClassification = (
        AutoModelForTokenClassification.from_pretrained(
            model, num_labels=len(label_to_id)
        )
    )

    # 5. Training arguments
    train_args = TrainingArguments(
        output_dir=str(_output_dir),
        run_name=run_name,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1 if keep_best_only else None,
        use_cpu=not torch.cuda.is_available(),
    )

    # 6. Trainer — test split is held out for eval_classifier.py
    trainer = Trainer(
        model=clf_model,
        args=train_args,
        train_dataset=tokenized_splits["train"],
        eval_dataset=tokenized_splits[eval_split],
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    trainer.train()
    trainer.save_model(str(_output_dir))
    tokenizer.save_pretrained(str(_output_dir))
    click.echo(f"Model saved to {_output_dir}")


if __name__ == "__main__":
    main()
