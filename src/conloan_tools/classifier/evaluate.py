from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import LabelSchema


def _load_schema_from_run_config(model_dir: Path) -> "LabelSchema":
    from .schema import LabelSchema

    cfg_path = model_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"run_config.json not found in {model_dir}. "
            "Cannot determine label schema."
        )
    cfg = json.loads(cfg_path.read_text())
    return LabelSchema.from_dict(cfg["schema"])


def _load_tokenizer_from_dir(model_dir: Path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    if not tok.is_fast:
        raise ValueError(f"Tokenizer at {model_dir} is not a fast tokenizer.")
    return tok


def run_evaluate(
    model_dir: Path,
    splits_dir: Path,
    target_splits: list[str],
    batch_size: int,
    word_level: bool,
    quiet: bool,
    eval_mode: str = "strict",
) -> dict[str, dict]:
    """Load a saved model, evaluate on requested splits, write eval_results.json."""
    import torch

    from .data import tokenize_and_align_labels
    from .splits import load_splits
    from .train import _make_compute_metrics, _silence_hf

    if quiet:
        _silence_hf()

    schema = _load_schema_from_run_config(model_dir)
    tokenizer = _load_tokenizer_from_dir(model_dir)

    splits, _ = load_splits(splits_dir)
    tokenized = splits.map(
        lambda x: tokenize_and_align_labels(
            x, tokenizer, schema, word_level=word_level
        ),
        batched=True,
    )

    from transformers import (
        AutoModelForTokenClassification,
        DataCollatorForTokenClassification,
        TrainingArguments,
    )
    from transformers.trainer import Trainer

    use_cpu = not torch.cuda.is_available()
    clf_model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    args = TrainingArguments(
        output_dir=str(model_dir),
        per_device_eval_batch_size=batch_size,
        use_cpu=use_cpu,
    )
    trainer = Trainer(
        model=clf_model,
        args=args,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=_make_compute_metrics(schema, eval_mode=eval_mode),
    )

    results: dict[str, dict] = {}
    for s in target_splits:
        print(f"Evaluating on '{s}' split…")
        results[s] = trainer.evaluate(eval_dataset=tokenized[s])

    out_path = model_dir / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    return results
