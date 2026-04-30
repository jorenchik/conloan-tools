from __future__ import annotations

import click

from .schema import SCHEMAS

_DEFAULT_MODEL = "distilbert-base-multilingual-cased"
_DRY_RUN_MODEL = "prajjwal1/bert-tiny"
_DEFAULT_EPOCHS = 10
_DEFAULT_BATCH = 16
_DEFAULT_LR = 5e-5
_DEFAULT_WD = 0.01
_DEFAULT_WARMUP = 0.1
_DEFAULT_MAX_LEN = 128
_DRY_RUN_EPOCHS = 1
_DRY_RUN_SAMPLES = 64


@click.group("classifier")
def classifier() -> None:
    """Conloan classifier utilities."""


@classifier.group("splits")
def splits_group() -> None:
    """Build and inspect train/test splits."""


@splits_group.command("build")
@click.option(
    "--inputs", "-i",
    required=True, multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="JSON dataset files (repeatable).",
)
@click.option(
    "--splits-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory to save splits.",
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--rebuild",
    is_flag=True,
    help="Overwrite existing splits.",
)
def splits_build(
    inputs: tuple[str, ...],
    splits_dir: str,
    seed: int,
    rebuild: bool,
) -> None:
    """Build 80/20 train/test splits from raw JSON files."""
    from pathlib import Path

    from .data import load_conloan
    from .splits import _splits_are_valid, build_and_save_splits

    p = Path(splits_dir)
    if not rebuild and _splits_are_valid(p):
        click.echo(f"Splits already exist at {p}. Use --rebuild to overwrite.")
        return

    dataset = load_conloan(list(inputs))
    build_and_save_splits(dataset, p, seed=seed, source_files=list(inputs))


@splits_group.command("info")
@click.option(
    "--splits-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
def splits_info(splits_dir: str) -> None:
    """Print metadata for an existing splits directory."""
    import json
    from pathlib import Path

    from .splits import splits_info as _info

    meta = _info(Path(splits_dir))
    click.echo(json.dumps(meta, indent=2))


def _hyperparams(f):
    opts = [
        click.option(
            "--epochs", type=int, default=None,
            help=f"Training epochs. [default: {_DEFAULT_EPOCHS}]",
        ),
        click.option(
            "--learning-rate", type=float,
            default=_DEFAULT_LR, show_default=True,
        ),
        click.option(
            "--batch-size", type=int,
            default=_DEFAULT_BATCH, show_default=True,
        ),
        click.option(
            "--weight-decay", type=float,
            default=_DEFAULT_WD, show_default=True,
        ),
        click.option(
            "--warmup-ratio", type=float,
            default=_DEFAULT_WARMUP, show_default=True,
        ),
        click.option(
            "--max-length", type=int,
            default=_DEFAULT_MAX_LEN, show_default=True,
            help="Maximum subword token length.",
        ),
        click.option(
            "--precision",
            type=click.Choice(["fp32", "fp16", "bf16"]),
            default="fp32", show_default=True,
        ),
        click.option(
            "--token-level", is_flag=True,
            help="Label all subword tokens; default is first-subword only.",
        ),
        click.option(
            "--class-weights", is_flag=True,
            help=(
                "Weight loss by inverse class frequency computed from the "
                "training split. Recommended for imbalanced datasets."
            ),
        ),
        click.option(
            "--eval-mode",
            type=click.Choice(["strict", "relaxed"]),
            default="strict", show_default=True,
            help="strict: B-/I- must match exactly; relaxed: entity boundary match only.",
        ),
    ]
    for o in reversed(opts):
        f = o(f)
    return f


def _common(f):
    opts = [
        click.option("--model", default=_DEFAULT_MODEL, show_default=True),
        click.option(
            "--schema",
            type=click.Choice(list(SCHEMAS.keys())),
            default="loan", show_default=True,
            help="Label schema.",
        ),
        click.option("--run-name", default=None, help="Auto-generated if omitted."),
        click.option("--seed", type=int, default=42, show_default=True),
        click.option(
            "--max-samples", type=int, default=None,
            help="Cap dataset to N samples.",
        ),
        click.option(
            "--dry-run", is_flag=True,
            help=(
                f"Smoke-test shorthand: model={_DRY_RUN_MODEL}, "
                f"epochs={_DRY_RUN_EPOCHS}, "
                f"max-samples={_DRY_RUN_SAMPLES}."
            ),
        ),
        click.option("--quiet", is_flag=True, help="Suppress HuggingFace output."),
    ]
    for o in reversed(opts):
        f = o(f)
    return f


def _resolve_dry_run(
    dry_run: bool,
    model: str,
    epochs: int | None,
    max_samples: int | None,
) -> tuple[str, int, int | None]:
    if dry_run:
        if model == _DEFAULT_MODEL:
            model = _DRY_RUN_MODEL
        epochs = epochs if epochs is not None else _DRY_RUN_EPOCHS
        max_samples = max_samples if max_samples is not None else _DRY_RUN_SAMPLES
        click.echo(
            f"[dry-run] model={model}, epochs={epochs}, max_samples={max_samples}",
            err=True,
        )
    else:
        epochs = epochs if epochs is not None else _DEFAULT_EPOCHS
    return model, epochs, max_samples


@classifier.command("train")
@_common
@_hyperparams
@click.option(
    "--splits-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Pre-built splits directory.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Parent directory for run artifacts.",
)
def train_cmd(
    model: str,
    schema: str,
    run_name: str | None,
    seed: int,
    max_samples: int | None,
    dry_run: bool,
    quiet: bool,
    epochs: int | None,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    warmup_ratio: float,
    max_length: int,
    precision: str,
    token_level: bool,
    class_weights: bool,
    eval_mode: str,
    splits_dir: str,
    output_dir: str,
) -> None:
    """Train on pre-built splits and save a model artifact."""
    from pathlib import Path

    from .train import run_train

    model, epochs, max_samples = _resolve_dry_run(dry_run, model, epochs, max_samples)

    # max_samples is applied inside splits — for train we just note it
    # (train operates on pre-built splits, so max_samples caps via select)
    if max_samples is not None:
        click.echo(
            f"[warn] --max-samples has no effect on train — "
            "splits are pre-built. Cap the dataset before building splits.",
            err=True,
        )

    run_train(
        splits_dir=Path(splits_dir),
        output_dir=Path(output_dir),
        model_name=model,
        schema=SCHEMAS[schema],
        run_name=run_name,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        max_length=max_length,
        precision=precision,
        word_level=not token_level,
        quiet=quiet,
        use_class_weights=class_weights,
        eval_mode=eval_mode,
    )


@classifier.command("kfold")
@_common
@_hyperparams
@click.option(
    "--inputs", "-i",
    required=True, multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help="JSON dataset files (repeatable).",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
)
@click.option(
    "--k-folds", type=int, default=5, show_default=True,
    help="Number of folds.",
)
def kfold_cmd(
    inputs: tuple[str, ...],
    model: str,
    schema: str,
    run_name: str | None,
    seed: int,
    max_samples: int | None,
    dry_run: bool,
    quiet: bool,
    epochs: int | None,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    warmup_ratio: float,
    max_length: int,
    precision: str,
    token_level: bool,
    class_weights: bool,
    eval_mode: str,
    output_dir: str,
    k_folds: int,
) -> None:
    """K-fold cross-validation. Produces kfold_results.json, no model saved."""
    from pathlib import Path

    from .kfold import run_kfold

    model, epochs, max_samples = _resolve_dry_run(dry_run, model, epochs, max_samples)

    run_kfold(
        input_files=list(inputs),
        output_dir=Path(output_dir),
        model_name=model,
        schema=SCHEMAS[schema],
        run_name=run_name,
        k=k_folds,
        seed=seed,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        max_length=max_length,
        precision=precision,
        word_level=not token_level,
        max_samples=max_samples,
        quiet=quiet,
        use_class_weights=class_weights,
        eval_mode=eval_mode,
    )


@classifier.command("eval")
@click.option(
    "--model-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Saved model artifact directory.",
)
@click.option(
    "--splits-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Pre-built splits directory.",
)
@click.option(
    "--split",
    type=click.Choice(["train", "test", "both"]),
    default="test", show_default=True,
)
@click.option(
    "--batch-size", type=int,
    default=_DEFAULT_BATCH, show_default=True,
)
@click.option(
    "--eval-mode",
    type=click.Choice(["strict", "relaxed"]),
    default="strict", show_default=True,
    help="strict: B-/I- must match exactly; relaxed: entity boundary match only.",
)
@click.option("--token-level", is_flag=True)
@click.option("--quiet", is_flag=True)
def eval_cmd(
    model_dir: str,
    splits_dir: str,
    split: str,
    batch_size: int,
    eval_mode: str,
    token_level: bool,
    quiet: bool,
) -> None:
    """Evaluate a saved model against pre-built splits."""
    from pathlib import Path

    from .evaluate import run_evaluate

    target = ["train", "test"] if split == "both" else [split]
    run_evaluate(
        model_dir=Path(model_dir),
        splits_dir=Path(splits_dir),
        target_splits=target,
        batch_size=batch_size,
        word_level=not token_level,
        quiet=quiet,
        eval_mode=eval_mode,
    )


@classifier.group("inspect")
def inspect_group() -> None:
    """Debug/development inspection tools."""


@inspect_group.command("tokens")
@click.option(
    "--inputs", "-i",
    required=True, multiple=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--model", default=_DEFAULT_MODEL, show_default=True)
@click.option(
    "--schema",
    type=click.Choice(list(SCHEMAS.keys())),
    default="loan", show_default=True,
)
@click.option("--token-level", is_flag=True)
@click.option("--max-samples", type=int, default=None)
@click.option("--seed", type=int, default=42, show_default=True)
def inspect_tokens_cmd(
    inputs: tuple[str, ...],
    model: str,
    schema: str,
    token_level: bool,
    max_samples: int | None,
    seed: int,
) -> None:
    """Print tokenized sentences with aligned labels."""
    from datasets import DatasetDict

    from .data import load_conloan, tokenize_and_align_labels
    from .train import _load_tokenizer

    schema_obj = SCHEMAS[schema]
    tokenizer = _load_tokenizer(model)

    dataset = load_conloan(list(inputs))
    if max_samples is not None:
        dataset = dataset.shuffle(seed=seed).select(
            range(min(max_samples, len(dataset)))
        )

    tokenized = DatasetDict({"data": dataset}).map(
        lambda x: tokenize_and_align_labels(
            x, tokenizer, schema_obj, word_level=not token_level
        ),
        batched=True,
    )["data"]

    col_w = 30
    sep = "-" * (col_w + 16)
    pad_tok = tokenizer.pad_token

    for i, (raw_row, tok_row) in enumerate(zip(dataset, tokenized)):
        tokens = tokenizer.convert_ids_to_tokens(tok_row["input_ids"])
        labels = tok_row["labels"]
        pad_start = next(
            (j for j, t in enumerate(tokens) if t == pad_tok), len(tokens)
        )
        n_pad = len(tokens) - pad_start

        click.echo(f"\n[{i}] {raw_row['source_annotated_loanwords']}")
        click.echo(f"{sep}\n  {'token':<{col_w}} label\n{sep}")
        for tok, lid in zip(tokens[:pad_start], labels[:pad_start]):
            label_str = "~~" if lid == -100 else schema_obj.id_to_label[lid]
            click.echo(f"  {tok:<{col_w}} {label_str}")
        if n_pad:
            click.echo(f"  {'...':<{col_w}} ({n_pad} padding tokens)")
        click.echo(sep)


@inspect_group.command("predictions")
@click.option(
    "--model-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--inputs", "-i",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--splits-dir",
    default=None,
    type=click.Path(file_okay=False),
)
@click.option(
    "--split",
    type=click.Choice(["train", "test", "both"]),
    default="test", show_default=True,
)
@click.option(
    "--relaxed",
    is_flag=True,
    help="Compare entity spans instead of individual B-/I- tokens.",
)
@click.option("--token-level", is_flag=True)
@click.option("--max-samples", type=int, default=None)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--quiet", is_flag=True)
def inspect_predictions_cmd(
    model_dir: str,
    inputs: tuple[str, ...],
    splits_dir: str | None,
    split: str,
    relaxed: bool,
    token_level: bool,
    max_samples: int | None,
    seed: int,
    quiet: bool,
) -> None:
    """Print token-level gold vs predicted labels for each sample."""
    import numpy as np
    import torch
    from datasets import DatasetDict
    from transformers import (
        AutoModelForTokenClassification,
        DataCollatorForTokenClassification,
        TrainingArguments,
    )
    from transformers.trainer import Trainer

    from .data import load_conloan, tokenize_and_align_labels
    from .evaluate import _load_schema_from_run_config, _load_tokenizer_from_dir
    from .splits import load_splits
    from .train import _silence_hf

    if quiet:
        _silence_hf()

    model_path = __import__("pathlib").Path(model_dir)
    schema = _load_schema_from_run_config(model_path)
    tokenizer = _load_tokenizer_from_dir(model_path)

    if splits_dir is not None:
        from pathlib import Path

        source_splits, _ = load_splits(Path(splits_dir))
        target = ["train", "test"] if split == "both" else [split]
        if len(target) == 1:
            dataset = source_splits[target[0]]
        else:
            from datasets import concatenate_datasets

            dataset = concatenate_datasets([source_splits[s] for s in target])
    elif inputs:
        dataset = load_conloan(list(inputs))
    else:
        raise click.UsageError("Provide --inputs or --splits-dir.")

    if max_samples is not None:
        dataset = dataset.shuffle(seed=seed).select(
            range(min(max_samples, len(dataset)))
        )

    tokenized = DatasetDict({"data": dataset}).map(
        lambda x: tokenize_and_align_labels(
            x, tokenizer, schema, word_level=not token_level
        ),
        batched=True,
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
    pred_ids = np.argmax(raw_preds, axis=-1)

    col_w = 30
    sep = "-" * (col_w + 32)
    pad_tok = tokenizer.pad_token

    for i, (raw_row, tok_row, sample_preds) in enumerate(
        zip(dataset, tokenized, pred_ids)
    ):
        tokens = tokenizer.convert_ids_to_tokens(tok_row["input_ids"])
        labels = tok_row["labels"]
        pad_start = next(
            (j for j, t in enumerate(tokens) if t == pad_tok), len(tokens)
        )
        n_pad = len(tokens) - pad_start

        click.echo(f"\n[{i}] {raw_row['source_annotated_loanwords']}")
        click.echo(f"{sep}\n  {'token':<{col_w}} {'gold':<12} pred\n{sep}")

        if relaxed:
            from seqeval.metrics import sequence_accuracy_score

            true_seq = [
                "~~" if lid == -100 else schema.id_to_label[lid]
                for lid in labels[:pad_start]
            ]
            pred_seq = [
                "~~" if lid == -100 else schema.id_to_label[pid]
                for lid, pid in zip(labels[:pad_start], sample_preds[:pad_start])
            ]
            acc = sequence_accuracy_score([true_seq], [pred_seq])
            click.echo(f"  (relaxed entity accuracy: {acc:.2%})")
            # Fall through to token-level display for reference
            for tok, lid, pid in zip(
                tokens[:pad_start], labels[:pad_start], sample_preds[:pad_start]
            ):
                gold = "~~" if lid == -100 else schema.id_to_label[lid]
                pred = "~~" if lid == -100 else schema.id_to_label[pid]
                flag = " !" if gold != pred and gold != "~~" else ""
                click.echo(f"  {tok:<{col_w}} {gold:<12} {pred}{flag}")
        else:
            for tok, lid, pid in zip(
                tokens[:pad_start], labels[:pad_start], sample_preds[:pad_start]
            ):
                gold = "~~" if lid == -100 else schema.id_to_label[lid]
                pred = "~~" if lid == -100 else schema.id_to_label[pid]
                flag = " !" if gold != pred and gold != "~~" else ""
                click.echo(f"  {tok:<{col_w}} {gold:<12} {pred}{flag}")
        if n_pad:
            click.echo(f"  {'...':<{col_w}} ({n_pad} padding tokens)")
        click.echo(sep)


@inspect_group.command("lengths")
@click.option(
    "--inputs", "-i",
    required=True, multiple=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--model", default=_DEFAULT_MODEL, show_default=True)
@click.option(
    "--schema",
    type=click.Choice(list(SCHEMAS.keys())),
    default="loan", show_default=True,
)
@click.option("--max-length", type=int, default=_DEFAULT_MAX_LEN, show_default=True)
def inspect_lengths_cmd(
    inputs: tuple[str, ...],
    model: str,
    schema: str,
    max_length: int,
) -> None:
    """Print token-length distribution and truncation rate."""
    import numpy as np
    from datasets import DatasetDict

    from .data import load_conloan, tokenize_and_align_labels
    from .train import _load_tokenizer

    schema_obj = SCHEMAS[schema]
    tokenizer = _load_tokenizer(model)
    dataset = load_conloan(list(inputs))

    tokenized = DatasetDict({"data": dataset}).map(
        lambda x: tokenize_and_align_labels(
            x,
            tokenizer,
            schema_obj,
            truncation=False,
            padding="do_not_pad",
        ),
        batched=True,
    )["data"]

    lengths = np.array([len(row["input_ids"]) for row in tokenized])
    truncated = int((lengths > max_length).sum())

    click.echo(
        f"\nToken length distribution (n={len(lengths)}, max_length={max_length}):"
    )
    click.echo(f"  {'min':<8} {int(lengths.min())}")
    click.echo(f"  {'mean':<8} {lengths.mean():.1f}")
    for p in (50, 75, 90, 95, 99, 100):
        click.echo(f"  {'p' + str(p):<8} {int(np.percentile(lengths, p))}")
    click.echo(
        f"\n  Truncated: {truncated}/{len(lengths)} "
        f"({100 * truncated / len(lengths):.1f}%)"
    )
