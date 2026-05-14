from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import LabelSchema


def _load_schema_from_run_config(model: str | Path) -> "LabelSchema":
    from .schema import LabelSchema

    p = Path(model)
    if p.is_dir():
        cfg_path = p / "run_config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            return LabelSchema.from_dict(cfg["schema"])

    # HF model ID or local dir without run_config.json: infer from model config.
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(str(model))
    label2id = {k: int(v) for k, v in label2id.items()}
    id_to_label = {v: k for k, v in label2id.items()}
    entity_types = tuple(k[2:] for k in label2id if k.startswith("B-"))
    primary_label = entity_types[0] if entity_types else "LOAN"
    return LabelSchema.from_dict({
        "name": "inferred",
        "label_to_id": label2id,
        "id_to_label": {str(k): v for k, v in id_to_label.items()},
        "report_labels": list(entity_types),
        "primary_label": primary_label,
    })


def _load_tokenizer_from_dir(model: str | Path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model))
    if not tok.is_fast:
        raise ValueError(f"Tokenizer at {model!r} is not a fast tokenizer.")
    return tok


def run_evaluate(
    model: str | Path,
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

    schema = _load_schema_from_run_config(model)
    tokenizer = _load_tokenizer_from_dir(model)

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

    model_p = Path(model)
    out_dir = model_p if model_p.is_dir() else Path(".")
    use_cpu = not torch.cuda.is_available()
    clf_model = AutoModelForTokenClassification.from_pretrained(str(model))
    args = TrainingArguments(
        output_dir=str(out_dir),
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

    out_path = out_dir / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {out_path}")
    return results
