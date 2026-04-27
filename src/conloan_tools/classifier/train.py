from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import LabelSchema


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_run_name(model: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}_{model.split('/')[-1]}"


def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if not tok.is_fast:
        raise ValueError(f"Model {model_name} does not have a fast tokenizer.")
    return tok


def _make_training_args(
    output_dir: str,
    run_name: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    weight_decay: float,
    warmup_ratio: float,
    train_size: int,
    precision: str,  # "fp32" | "fp16" | "bf16"
    use_cpu: bool,
):
    import math

    from transformers import TrainingArguments

    total_steps = math.ceil(train_size / batch_size) * epochs
    warmup_steps = math.ceil(total_steps * warmup_ratio)

    return TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        optim="adamw_torch",
        eval_strategy="no",
        save_strategy="no",
        logging_steps=10,
        use_cpu=use_cpu,
        fp16=precision == "fp16" and not use_cpu,
        bf16=precision == "bf16" and not use_cpu,
    )


def _make_compute_metrics(schema: "LabelSchema"):
    import numpy as np
    from seqeval.metrics import classification_report, f1_score

    def compute_metrics(p):
        predictions = np.argmax(p.predictions, axis=2)
        labels = p.label_ids

        true_seqs: list[list[str]] = []
        pred_seqs: list[list[str]] = []
        for pred_seq, label_seq in zip(predictions, labels):
            true_seq, pred_seq_out = [], []
            for pred_tok, label_tok in zip(pred_seq, label_seq):
                if label_tok == -100:
                    continue
                true_seq.append(schema.id_to_label[label_tok])
                pred_seq_out.append(schema.id_to_label[pred_tok])
            true_seqs.append(true_seq)
            pred_seqs.append(pred_seq_out)

        report = classification_report(
            true_seqs, pred_seqs, output_dict=True, zero_division=0
        )
        metrics: dict = {
            "f1_macro": f1_score(
                true_seqs, pred_seqs, average="macro", zero_division=0
            )
        }
        for label in schema.report_labels:
            if label not in report:
                continue
            for metric in ("precision", "recall", "f1-score", "support"):
                metrics[f"{label}_{metric}"] = report[label][metric]
        return metrics

    return compute_metrics


def _build_trainer(
    model_name: str,
    schema: "LabelSchema",
    train_dataset,
    eval_dataset,
    training_args,
    tokenizer,
):
    from transformers import (
        AutoModelForTokenClassification,
        DataCollatorForTokenClassification,
    )
    from transformers.trainer import Trainer

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(schema.label_to_id)
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=_make_compute_metrics(schema),
    )


def _tokenize_splits(splits, tokenizer, schema, max_length: int, word_level: bool):
    from .data import tokenize_and_align_labels

    return splits.map(
        lambda x: tokenize_and_align_labels(
            x,
            tokenizer,
            schema,
            max_length=max_length,
            word_level=word_level,
        ),
        batched=True,
    )


def _evaluate_saved_model(
    model_path: Path,
    tokenized_splits,
    schema: "LabelSchema",
    tokenizer,
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
        compute_metrics=_make_compute_metrics(schema),
    )

    results: dict[str, dict] = {}
    for s in target_splits:
        print(f"Evaluating on '{s}' split…")
        results[s] = trainer.evaluate(eval_dataset=tokenized_splits[s])
    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_train(
    splits_dir: Path,
    output_dir: Path,
    model_name: str,
    schema: "LabelSchema",
    run_name: str | None,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    warmup_ratio: float,
    max_length: int,
    precision: str,
    word_level: bool,
    quiet: bool,
) -> Path:
    """Train on the pre-built train split; evaluate and save artifact.

    Returns the run directory path.
    """
    import torch

    from .splits import load_splits

    if quiet:
        _silence_hf()

    run_name = run_name or _generate_run_name(model_name)
    print(f"Run: {run_name}")

    splits, splits_meta = load_splits(splits_dir)
    tokenizer = _load_tokenizer(model_name)
    tokenized = _tokenize_splits(splits, tokenizer, schema, max_length, word_level)

    use_cpu = not torch.cuda.is_available()
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    args = _make_training_args(
        output_dir=str(run_dir),
        run_name=run_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        train_size=len(tokenized["train"]),
        precision=precision,
        use_cpu=use_cpu,
    )
    trainer = _build_trainer(
        model_name=model_name,
        schema=schema,
        train_dataset=tokenized["train"],
        eval_dataset=None,
        training_args=args,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))
    print(f"Model saved to {run_dir}")

    # Verify artifact by reloading from disk before evaluating.
    results = _evaluate_saved_model(
        model_path=run_dir,
        tokenized_splits=tokenized,
        schema=schema,
        tokenizer=tokenizer,
        batch_size=batch_size,
        target_splits=["test"],
        use_cpu=use_cpu,
    )
    (run_dir / "test_metrics.json").write_text(
        json.dumps(results["test"], indent=2)
    )

    run_config = {
        "run_name": run_name,
        "model_name": model_name,
        "schema": schema.to_dict(),
        "hyperparameters": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "max_length": max_length,
            "precision": precision,
            "word_level": word_level,
        },
        "splits_dir": str(splits_dir),
        "splits_meta": splits_meta,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    return run_dir


def _silence_hf() -> None:
    import os

    import datasets
    import transformers

    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()
    datasets.disable_progress_bar()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
