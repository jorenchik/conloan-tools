from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import click

from conloan_tools.classifier.data import (
    LoanLabel,
    build_and_save_splits,
    load_conloan,
    tokenize_and_align_labels,
)

if TYPE_CHECKING:
    from typing import Callable
    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer, EvalPrediction, Trainer, TrainingArguments

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DRY_RUN_MODEL = "prajjwal1/bert-tiny"
DEFAULT_MODEL = "distilbert-base-multilingual-cased"

DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 5e-5
DEFAULT_WEIGHT_DECAY = 0.01

DRY_RUN_EPOCHS = 1
DRY_RUN_MAX_SAMPLES = 64

#KFOLD_LABELS = ("B-LOAN", "I-LOAN", "O")
#KFOLD_LABELS = ("B-LOAN", "O")
KFOLD_LABELS = ("LOAN",)
KFOLD_METRICS = ("precision", "recall", "f1-score", "support")
_METRIC_HEADER = {
    "precision": "prec",
    "recall": "rec",
    "f1-score": "f1",
    "support": "supp",
}

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def _splits_are_valid(splits_dir: Path) -> bool:
    required = {"train", "test", "dataset_dict.json"}
    return splits_dir.exists() and required.issubset(
        {p.name for p in splits_dir.iterdir()}
    )


def _load_or_build_splits(
    inputs: list[str],
    splits_dir: Path,
    seed: int,
    rebuild: bool,
    dev: bool = False,
) -> "DatasetDict":
    from datasets import DatasetDict

    if not rebuild and _splits_are_valid(splits_dir):
        click.echo(f"Loading existing splits from {splits_dir}")
        return DatasetDict.load_from_disk(str(splits_dir))

    reason = "--rebuild-splits set" if rebuild else f"no valid splits at {splits_dir}"
    click.echo(f"Building splits ({reason})")
    return build_and_save_splits(load_conloan(inputs), splits_dir, seed=seed, dev=dev)


# ---------------------------------------------------------------------------
# Metrics / display
# ---------------------------------------------------------------------------


def _make_compute_metrics(
    id_to_label: dict[int, str],
) -> "Callable[[EvalPrediction], dict]":
    import numpy as np
    from seqeval.metrics import classification_report, f1_score

    def compute_metrics(p: "EvalPrediction") -> dict:
        predictions = np.argmax(p.predictions, axis=2)
        labels = p.label_ids

        true_seqs: list[list[str]] = []
        pred_seqs: list[list[str]] = []
        for pred_seq, label_seq in zip(predictions, labels):
            true_seq: list[str] = []
            pred_seq_out: list[str] = []
            for pred_tok, label_tok in zip(pred_seq, label_seq):
                if label_tok == -100:
                    continue
                true_seq.append(id_to_label[label_tok])
                pred_seq_out.append(id_to_label[pred_tok])
            true_seqs.append(true_seq)
            pred_seqs.append(pred_seq_out)

        report = classification_report(
            true_seqs, pred_seqs, output_dict=True, zero_division=0
        )
        return {
            "f1_macro": f1_score(
                true_seqs, pred_seqs, average="macro", zero_division=0
            ),
            **{
                f"{label}_{metric}": report[label][metric]
                for label in KFOLD_LABELS
                for metric in KFOLD_METRICS
                if label in report
            },
        }

    return compute_metrics


def _eval_key(label: str, metric: str) -> str:
    return f"eval_{label}_{metric}"


def _print_table(
    rows: list[tuple[str, dict]],
    *,
    fold_col_width: int = 6,
) -> None:
    """
    Generic table printer.
    `rows` is a list of (row_label, result_dict) pairs.
    The row_label is printed once per group of KFOLD_LABELS rows.
    """
    col_label = 9
    col_metric = 9

    header = (
        f"{'':{ fold_col_width}}"
        f"{'label':<{col_label}}"
        + "".join(f"{_METRIC_HEADER[m]:>{col_metric}}" for m in KFOLD_METRICS)
    )
    sep = "-" * len(header)
    click.echo(f"\n{sep}\n{header}\n{sep}")

    for row_label, result in rows:
        first = True
        for label in KFOLD_LABELS:
            prefix = row_label if first else ""
            first = False
            row = f"{prefix:<{fold_col_width}}{label:<{col_label}}"
            for metric in KFOLD_METRICS:
                val = result.get(_eval_key(label, metric), float("nan"))
                row += (
                    f"{val:>{col_metric}.0f}"
                    if metric == "support"
                    else f"{val:>{col_metric}.4f}"
                )
            click.echo(row)

        macro = result.get("eval_f1_macro", float("nan"))
        click.echo(
            f"{'':{ fold_col_width}}{'macro f1':<{col_label}}"
            f"{'':>{col_metric}}{'':>{col_metric}}{macro:>{col_metric}.4f}{'':>{col_metric}}"
        )
        click.echo(sep)


def _print_single_eval_table(result: dict, title: str) -> None:
    _print_table([(title, result)], fold_col_width=max(len(title), 6))


def _print_kfold_table(fold_results: list[dict], aggregate: dict) -> None:
    rows = [(str(i + 1), r) for i, r in enumerate(fold_results)]

    for stat in ("mean", "std"):
        stat_result = {
            _eval_key(label, metric): aggregate.get(
                f"{_eval_key(label, metric)}_{stat}", float("nan")
            )
            for label in KFOLD_LABELS
            for metric in KFOLD_METRICS
        }
        rows.append((stat, stat_result))

    _print_table(rows)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _generate_run_name(model: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}_{model.split('/')[-1]}"


def _silence_hf() -> None:
    import os

    import datasets
    import transformers

    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()
    datasets.disable_progress_bar()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _make_training_args(
    output_dir: str,
    run_name: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    weight_decay: float,
    use_cpu: bool = False,
    fp16: bool = False,
    bf16: bool = False,
) -> "TrainingArguments":
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optim="adamw_torch",
        eval_strategy="no",
        save_strategy="no",
        logging_steps=10,
        use_cpu=use_cpu,
        fp16=fp16 and not use_cpu,
        bf16=bf16 and not use_cpu,
    )


def _build_trainer(
    model_name: str,
    label_to_id: dict[str, int],
    train_dataset: "Dataset",
    eval_dataset: "Dataset | None",
    training_args: "TrainingArguments",
    tokenizer: "AutoTokenizer",
) -> "Trainer":
    from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
    from transformers.trainer import Trainer

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_to_id),
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=_make_compute_metrics(LoanLabel.id_to_label()),
    )


def _load_tokenizer(model_name: str) -> "AutoTokenizer":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.is_fast:
        raise ValueError("tokenize_and_align_labels requires a fast tokenizer.")
    return tokenizer


def _cap_dataset(dataset: "Dataset", max_samples: int | None) -> "Dataset":
    if max_samples is None:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def _cap_splits(splits: "DatasetDict", max_samples: int | None) -> "DatasetDict":
    if max_samples is None:
        return splits
    from datasets import DatasetDict

    return DatasetDict(
        {k: _cap_dataset(v, max_samples) for k, v in splits.items()}
    )


def _tokenize_splits(
    splits: "DatasetDict",
    tokenizer: "AutoTokenizer",
    label_to_id: dict[str, int],
    word_level: bool = True,
) -> "DatasetDict":
    return splits.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id, word_level=word_level),
        batched=True,
    )


# ---------------------------------------------------------------------------
# K-fold helpers
# ---------------------------------------------------------------------------


def _kfold_indices(n: int, k: int, seed: int) -> list[tuple[list[int], list[int]]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n).tolist()
    fold_size = n // k
    folds: list[tuple[list[int], list[int]]] = []
    for i in range(k):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < k - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = indices[:val_start] + indices[val_end:]
        folds.append((train_idx, val_idx))
    return folds


# ---------------------------------------------------------------------------
# Shared CLI options
# ---------------------------------------------------------------------------


def common_options(f: "Callable") -> "Callable":
    decorators = [
        click.option(
            "--inputs", "-i",
            required=True, multiple=True,
            type=click.Path(exists=True, dir_okay=False),
            help="JSON dataset files (repeatable).",
        ),
        click.option(
            "--output-dir",
            required=True,
            type=click.Path(file_okay=False),
            help="Directory for checkpoints / results.",
        ),
        click.option("--model", default=DEFAULT_MODEL, show_default=True),
        click.option("--run-name", default=None, help="Auto-generated if omitted."),
        click.option("--seed", type=int, default=42, show_default=True),
        click.option(
            "--max-samples", type=int, default=None,
            help=f"Cap dataset to N samples. Dry-run default: {DRY_RUN_MAX_SAMPLES}.",
        ),
        click.option(
            "--dry-run", is_flag=True,
            help=(
                f"Shorthand: model={DRY_RUN_MODEL}, "
                f"epochs={DRY_RUN_EPOCHS}, "
                f"max-samples={DRY_RUN_MAX_SAMPLES} (unless explicitly set)."
            ),
        ),
        click.option("--quiet", is_flag=True, help="Suppress HuggingFace logs."),
    ]
    for d in reversed(decorators):
        f = d(f)
    return f


def splits_options(f: "Callable") -> "Callable":
    decorators = [
        click.option(
            "--splits-dir",
            required=True,
            type=click.Path(file_okay=False),
            help="Directory to save/load train/test splits.",
        ),
        click.option("--rebuild-splits", is_flag=True, help="Force rebuild splits."),
        click.option("--dev-split", is_flag=True, help="Include a dev split (smaller test)."),
    ]
    for d in reversed(decorators):
        f = d(f)
    return f


def hyperparams(f: "Callable") -> "Callable":
    decorators = [
        click.option(
            "--epochs", type=int, default=None,
            help=f"Training epochs. [default: {DEFAULT_EPOCHS}]",
        ),
        click.option(
            "--token-level", is_flag=True,
            help="Evaluate on all subword tokens; default is word-level (first token only).",
        ),
        click.option("--learning-rate", type=float, default=DEFAULT_LR, show_default=True),
        click.option("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, show_default=True),
        click.option("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, show_default=True),
        click.option("--fp16", is_flag=True, help="Enable fp16 mixed precision (GPU only)."),
        click.option("--bf16", is_flag=True, help="Enable bf16 mixed precision (GPU only, Ampere+)."),
    ]
    for d in reversed(decorators):
        f = d(f)
    return f


def _resolve_dry_run(
    dry_run: bool,
    model: str,
    epochs: int | None,
    max_samples: int | None,
) -> tuple[str, int, int | None]:
    if dry_run:
        if model == DEFAULT_MODEL:
            model = DRY_RUN_MODEL
        epochs = epochs if epochs is not None else DRY_RUN_EPOCHS
        max_samples = max_samples if max_samples is not None else DRY_RUN_MAX_SAMPLES
        click.echo(
            f"[dry-run] model={model}, epochs={epochs}, max_samples={max_samples}",
            err=True,
        )
    else:
        epochs = epochs if epochs is not None else DEFAULT_EPOCHS
    return model, epochs, max_samples


def _evaluate_model(
    model_path: str | Path,
    tokenized_splits: "DatasetDict",
    label_to_id: dict[str, int],
    tokenizer: "AutoTokenizer",
    batch_size: int,
    target_splits: list[str],
    use_cpu: bool,
) -> dict[str, dict]:
    from transformers import (
        AutoModelForTokenClassification,
        DataCollatorForTokenClassification,
        TrainingArguments,
    )
    from transformers.trainer import Trainer

    model_path = Path(model_path)
    clf_model = AutoModelForTokenClassification.from_pretrained(str(model_path))

    args = TrainingArguments(
        output_dir=str(model_path),
        per_device_eval_batch_size=batch_size,
        use_cpu=use_cpu,
    )
    trainer = Trainer(
        model=clf_model,
        args=args,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=_make_compute_metrics(LoanLabel.id_to_label()),
    )

    results: dict[str, dict] = {}
    for s in target_splits:
        click.echo(f"Evaluating on {s} split...")
        metrics = trainer.evaluate(eval_dataset=tokenized_splits[s])
        _print_single_eval_table(metrics, title=s)
        results[s] = metrics

    return results


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@click.command("train")
@common_options
@splits_options
@hyperparams
def train(
    inputs: tuple[str, ...],
    splits_dir: str,
    output_dir: str,
    model: str,
    run_name: str | None,
    seed: int,
    rebuild_splits: bool,
    dev_split: bool,
    max_samples: int | None,
    dry_run: bool,
    epochs: int | None,
    token_level: bool,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    quiet: bool,
    fp16: bool,
    bf16: bool,
) -> None:
    """Train on the train split, evaluate on the test split."""
    import torch

    if quiet:
        _silence_hf()

    model, epochs, max_samples = _resolve_dry_run(dry_run, model, epochs, max_samples)

    run_name = run_name or _generate_run_name(model)
    click.echo(f"Run: {run_name}")

    output_path = Path(output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)

    label_to_id = LoanLabel.label_to_id()
    tokenizer = _load_tokenizer(model)

    splits = _load_or_build_splits(list(inputs), Path(splits_dir), seed, rebuild_splits, dev=dev_split)
    splits = _cap_splits(splits, max_samples)
    tokenized = _tokenize_splits(splits, tokenizer, label_to_id, word_level=not token_level)

    args = _make_training_args(
        output_dir=str(output_path),
        run_name=run_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        weight_decay=weight_decay,
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = _build_trainer(
        model_name=model,
        label_to_id=label_to_id,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        training_args=args,
        tokenizer=tokenizer,
    )

    trainer.train()

    # trainer.save_model(str(output_path))
    # tokenizer.save_pretrained(str(output_path))
    # click.echo(f"Model saved to {output_path}")

    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    click.echo(f"Model saved to {output_path}")

    # Load from disk and evaluate — verifies the saved artifact
    click.echo("Evaluating on test split...")
    results = _evaluate_model(
        model_path=output_path,
        tokenized_splits=tokenized,
        label_to_id=label_to_id,
        tokenizer=tokenizer,
        batch_size=batch_size,
        target_splits=["test"],
        use_cpu=not torch.cuda.is_available(),
    )
    (output_path / "test_metrics.json").write_text(
        json.dumps(results["test"], indent=2)
    )

    # click.echo("Evaluating on test split...")
    # test_metrics = trainer.evaluate()
    # _print_single_eval_table(test_metrics, title="test")
    # (output_path / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))


@click.command("kfold")
@common_options
@hyperparams
@click.option("--k-folds", type=int, default=5, show_default=True, help="Number of folds.")
def kfold(
    inputs: tuple[str, ...],
    output_dir: str,
    model: str,
    run_name: str | None,
    seed: int,
    max_samples: int | None,
    dry_run: bool,
    epochs: int | None,
    token_level: bool,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    k_folds: int,
    quiet: bool,
    fp16: bool,
    bf16: bool,
) -> None:
    """K-fold CV on the full dataset. No model artifact is saved."""
    import numpy as np
    import torch

    if quiet:
        _silence_hf()

    model, epochs, max_samples = _resolve_dry_run(dry_run, model, epochs, max_samples)

    run_name = run_name or _generate_run_name(model)
    click.echo(f"Run: {run_name}")
    click.echo(f"[kfold] k={k_folds}, lr={learning_rate}, wd={weight_decay}, epochs={epochs}")

    output_path = Path(output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)

    label_to_id = LoanLabel.label_to_id()
    tokenizer = _load_tokenizer(model)

    raw_data = load_conloan(list(inputs))
    full_dataset = _cap_dataset(raw_data.shuffle(seed=seed), max_samples)
    tokenized_full = full_dataset.map(
        lambda x: tokenize_and_align_labels(
            x, tokenizer, label_to_id, word_level=not token_level
        ),
        batched=True,
    )

    use_cpu = not torch.cuda.is_available()
    folds = _kfold_indices(len(tokenized_full), k_folds, seed)
    fold_results: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        click.echo(
            f"  Fold {fold_idx + 1}/{k_folds} "
            f"(train={len(train_idx)}, val={len(val_idx)})"
        )
        args = _make_training_args(
            output_dir=str(output_path / f"fold_{fold_idx + 1}"),
            run_name=f"{run_name}_fold{fold_idx + 1}",
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            weight_decay=weight_decay,
            use_cpu=use_cpu,
        )
        trainer = _build_trainer(
            model_name=model,
            label_to_id=label_to_id,
            train_dataset=tokenized_full.select(train_idx),
            eval_dataset=tokenized_full.select(val_idx),
            training_args=args,
            tokenizer=tokenizer,
        )
        trainer.train()
        fold_results.append(trainer.evaluate())

        del trainer
        if not use_cpu:
            torch.cuda.empty_cache()

    all_keys = {k for r in fold_results for k in r}
    aggregate: dict = {}
    for key in all_keys:
        vals = [r[key] for r in fold_results if key in r]
        aggregate[f"{key}_mean"] = float(np.mean(vals))
        aggregate[f"{key}_std"] = float(np.std(vals))

    summary = {
        "hyperparameters": {
            "model": model,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size,
            "k_folds": k_folds,
            "seed": seed,
        },
        "folds": fold_results,
        **aggregate,
    }
    results_path = output_path / "kfold_results.json"
    results_path.write_text(json.dumps(summary, indent=2))

    _print_kfold_table(fold_results, aggregate)
    click.echo(f"\nFull results saved to {results_path}")


@click.command("eval")
@click.option(
    "--model-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--inputs", "-i",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="JSON dataset files. Required if splits don't exist yet.",
)
@click.option(
    "--splits-dir",
    required=True,
    type=click.Path(file_okay=False),
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--rebuild-splits", is_flag=True)
@click.option("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, show_default=True)
@click.option(
    "--split",
    type=click.Choice(["train", "test", "both"]),
    default="test",
    show_default=True,
)
@click.option("--token-level", is_flag=True)
@click.option("--quiet", is_flag=True)
def eval_cmd(
    model_dir: str,
    inputs: tuple[str, ...],
    splits_dir: str,
    seed: int,
    rebuild_splits: bool,
    batch_size: int,
    split: str,
    token_level: bool,
    quiet: bool,
) -> None:
    """Load a saved model and evaluate on train/test splits."""
    import torch

    if quiet:
        _silence_hf()

    splits_path = Path(splits_dir)

    if rebuild_splits or not _splits_are_valid(splits_path):
        if not inputs:
            raise click.UsageError(
                "No valid splits found at --splits-dir. "
                "Provide --inputs to build them."
            )

    label_to_id = LoanLabel.label_to_id()
    tokenizer = _load_tokenizer(model_dir)

    splits = _load_or_build_splits(
        list(inputs), splits_path, seed, rebuild_splits, dev=False
    )
    tokenized = _tokenize_splits(
        splits, tokenizer, label_to_id, word_level=not token_level
    )

    use_cpu = not torch.cuda.is_available()
    target_splits = ["train", "test"] if split == "both" else [split]

    results = _evaluate_model(
        model_path=model_dir,
        tokenized_splits=tokenized,
        label_to_id=label_to_id,
        tokenizer=tokenizer,
        batch_size=batch_size,
        target_splits=target_splits,
        use_cpu=use_cpu,
    )

    out_path = Path(model_dir) / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    click.echo(f"\nResults saved to {out_path}")


@click.command("inspect-tokens")
@click.option(
    "--inputs", "-i",
    required=True, multiple=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option("--token-level", is_flag=True)
@click.option("--max-samples", type=int, default=None)
@click.option("--seed", type=int, default=42, show_default=True)
def inspect_tokens(
    inputs: tuple[str, ...],
    model: str,
    token_level: bool,
    max_samples: int | None,
    seed: int,
) -> None:
    """Print tokenized sentences with aligned labels to stdout."""
    from datasets import DatasetDict

    label_to_id = LoanLabel.label_to_id()
    id_to_label = LoanLabel.id_to_label()
    tokenizer = _load_tokenizer(model)

    dataset = load_conloan(list(inputs))
    dataset = _cap_dataset(dataset.shuffle(seed=seed), max_samples)

    tokenized = _tokenize_splits(
        DatasetDict({"data": dataset}),
        tokenizer,
        label_to_id,
        word_level=not token_level,
    )["data"]

    col_w = 30
    sep = "-" * (col_w + 16)

    for i, (raw_row, tok_row) in enumerate(zip(dataset, tokenized)):
        tokens = tokenizer.convert_ids_to_tokens(tok_row["input_ids"])
        labels = tok_row["labels"]
        click.echo(f"\n[{i}] {raw_row['source_annotated_loanwords']}")
        click.echo(sep)
        click.echo(f"  {'token':<{col_w}} label")
        click.echo(sep)
        pad = tokenizer.pad_token
        pad_start = next(
            (j for j, t in enumerate(tokens) if t == pad),
            len(tokens),
        )
        n_pad = len(tokens) - pad_start
        for tok, label_id in zip(tokens[:pad_start], labels[:pad_start]):
            label_str = "~~" if label_id == -100 else id_to_label[label_id]
            click.echo(f"  {tok:<{col_w}} {label_str}")
        if n_pad:
            click.echo(f"  {'...':<{col_w}} ({n_pad} padding tokens)")
        click.echo(sep)


@click.command("inspect-predictions")
@click.option(
    "--inputs", "-i",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="JSON dataset files. Required if splits don't exist yet.",
)
@click.option(
    "--splits-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Directory to save/load train/test splits.",
)
@click.option(
    "--split",
    type=click.Choice(["train", "test", "both"]),
    default="test",
    show_default=True,
)
@click.option("--rebuild-splits", is_flag=True)
@click.option(
    "--model-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option("--token-level", is_flag=True)
@click.option("--max-samples", type=int, default=None)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--quiet", is_flag=True)
def inspect_predictions(
    inputs: tuple[str, ...],
    splits_dir: str | None,
    split: str,
    rebuild_splits: bool,
    model_dir: str,
    token_level: bool,
    max_samples: int | None,
    seed: int,
    quiet: bool,
) -> None:
    """Run a saved model on samples and print token-level predictions vs labels."""
    import numpy as np
    import torch
    from datasets import DatasetDict
    from transformers import (
        AutoModelForTokenClassification,
        DataCollatorForTokenClassification,
        TrainingArguments,
    )
    from transformers.trainer import Trainer

    if quiet:
        _silence_hf()

    label_to_id = LoanLabel.label_to_id()
    id_to_label = LoanLabel.id_to_label()
    tokenizer = _load_tokenizer(model_dir)

    # --- resolve dataset source ---
    if splits_dir is not None:
        splits_path = Path(splits_dir)
        if rebuild_splits or not _splits_are_valid(splits_path):
            if not inputs:
                raise click.UsageError(
                    "No valid splits found at --splits-dir. "
                    "Provide --inputs to build them."
                )
        source = _load_or_build_splits(
            list(inputs), splits_path, seed, rebuild_splits, dev=False
        )
        target_splits = ["train", "test"] if split == "both" else [split]
        dataset = source[target_splits[0]] if len(target_splits) == 1 else source
        if len(target_splits) > 1:
            from datasets import concatenate_datasets
            dataset = concatenate_datasets([source[s] for s in target_splits])
    else:
        if not inputs:
            raise click.UsageError("Provide either --inputs or --splits-dir.")
        dataset = load_conloan(list(inputs))

    dataset = _cap_dataset(dataset.shuffle(seed=seed), max_samples)

    tokenized = _tokenize_splits(
        DatasetDict({"data": dataset}),
        tokenizer,
        label_to_id,
        word_level=not token_level,
    )["data"]

    use_cpu = not torch.cuda.is_available()
    clf_model = AutoModelForTokenClassification.from_pretrained(model_dir)
    trainer = Trainer(
        model=clf_model,
        args=TrainingArguments(
            output_dir=model_dir,
            per_device_eval_batch_size=16,
            use_cpu=use_cpu,
        ),
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    raw_preds, _, _ = trainer.predict(tokenized)
    pred_ids = np.argmax(raw_preds, axis=-1)  # (N, seq_len)

    col_w = 30
    sep = "-" * (col_w + 32)

    for i, (raw_row, tok_row, sample_preds) in enumerate(
        zip(dataset, tokenized, pred_ids)
    ):
        tokens = tokenizer.convert_ids_to_tokens(tok_row["input_ids"])
        labels = tok_row["labels"]

        pad = tokenizer.pad_token
        pad_start = next(
            (j for j, t in enumerate(tokens) if t == pad),
            len(tokens),
        )
        n_pad = len(tokens) - pad_start

        click.echo(f"\n[{i}] {raw_row['source_annotated_loanwords']}")
        click.echo(sep)
        click.echo(f"  {'token':<{col_w}} {'gold':<12} pred")
        click.echo(sep)

        for tok, label_id, pred_id in zip(
            tokens[:pad_start],
            labels[:pad_start],
            sample_preds[:pad_start],
        ):
            gold_str = "~~" if label_id == -100 else id_to_label[label_id]
            pred_str = "~~" if label_id == -100 else id_to_label[pred_id]
            mismatch = " !" if gold_str != pred_str and gold_str != "~~" else ""
            click.echo(f"  {tok:<{col_w}} {gold_str:<12} {pred_str}{mismatch}")

        if n_pad:
            click.echo(f"  {'...':<{col_w}} ({n_pad} padding tokens)")
        click.echo(sep)
