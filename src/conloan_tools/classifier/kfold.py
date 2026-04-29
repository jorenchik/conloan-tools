from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import LabelSchema


def _kfold_indices(
    n: int, k: int, seed: int
) -> list[tuple[list[int], list[int]]]:
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


def run_kfold(
    input_files: list[str],
    output_dir: Path,
    model_name: str,
    schema: "LabelSchema",
    run_name: str | None,
    k: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    warmup_ratio: float,
    max_length: int,
    precision: str,
    word_level: bool,
    max_samples: int | None,
    quiet: bool,
    use_class_weights: bool = False,
) -> Path:
    """K-fold CV — estimates generalisation, produces no model artifact.

    Returns the run directory (contains kfold_results.json only).
    """
    import numpy as np
    import torch

    from .data import load_conloan, tokenize_and_align_labels
    from .train import (
        _build_trainer,
        _generate_run_name,
        _make_class_weights,
        _make_training_args,
        _silence_hf,
        _load_tokenizer,
    )

    if quiet:
        _silence_hf()

    run_name = run_name or _generate_run_name(model_name)
    print(f"Run: {run_name}")
    print(
        f"[kfold] k={k}, lr={learning_rate}, wd={weight_decay}, epochs={epochs}"
    )

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_data = load_conloan(input_files)
    if max_samples is not None:
        # Reshuffle to avoid head bias if sliced.
        raw_data = raw_data.select(range(min(max_samples, len(raw_data))))

    tokenizer = _load_tokenizer(model_name)
    tokenized_full = raw_data.map(
        lambda x: tokenize_and_align_labels(
            x,
            tokenizer,
            schema,
            max_length=max_length,
            word_level=word_level,
        ),
        batched=True,
    )

    use_cpu = not torch.cuda.is_available()
    folds = _kfold_indices(len(tokenized_full), k, seed)
    fold_results: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(
            f"  Fold {fold_idx + 1}/{k} "
            f"(train={len(train_idx)}, val={len(val_idx)})"
        )
        args = _make_training_args(
            output_dir=str(run_dir / f"fold_{fold_idx + 1}"),
            run_name=f"{run_name}_fold{fold_idx + 1}",
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            train_size=len(train_idx),
            precision=precision,
            use_cpu=use_cpu,
        )
        train_split = tokenized_full.select(train_idx)
        class_weights = (
            _make_class_weights(train_split, schema)
            if use_class_weights
            else None
        )
        trainer = _build_trainer(
            model_name=model_name,
            schema=schema,
            train_dataset=train_split,
            eval_dataset=tokenized_full.select(val_idx),
            training_args=args,
            tokenizer=tokenizer,
            class_weights=class_weights,
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

    results = {
        "run_name": run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": {
            "model": model_name,
            "schema": schema.name,
            "k": k,
            "seed": seed,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "max_length": max_length,
            "precision": precision,
            "word_level": word_level,
            "use_class_weights": use_class_weights,
        },
        "folds": fold_results,
        "aggregate": aggregate,
    }
    out_path = run_dir / "kfold_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")
    return run_dir

