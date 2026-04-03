from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import click
from tqdm import tqdm

if TYPE_CHECKING:
    import stanza
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

# ----- Public types ------

@dataclass
class NERModel:
    tokenizer: AutoTokenizer
    model: AutoModelForTokenClassification
    id2label: dict[int, str]
    device: str
    torch_dtype: torch.dtype = None


@dataclass
class NERResult:
    text: str
    words: list[str]
    label_ids: list[int]
    confidences: list[float]
    probs: torch.Tensor | None = None  # shape (W, num_labels), optional

# ----- Internal ------

def _stanza_tokenizer(lang: str = "lv", use_gpu: bool = False) -> stanza.Pipeline:
    import stanza as _stanza

    return _stanza.Pipeline(
        lang,
        processors="tokenize",
        verbose=False,
        use_gpu=use_gpu,
    )


def _normalize_label(label: str) -> str:
    return label[2:] if label.startswith(("B-", "I-")) else label

# ----- Public interface ------

def get_id2label(model: NERModel) -> dict[int, str]:
    """Return the normalized id→label mapping (B-/I- prefixes stripped)."""
    return model.id2label


def get_logits(
    model: NERModel,
    batch: list[tuple[str, list[str]]],
) -> list[tuple[list[str], torch.Tensor]]:
    """Run model and return word-aligned logits. Shape per entry: (W, num_labels)."""
    import torch

    batched_words = [b[1] for b in batch]

    encoded = model.tokenizer(
        batched_words,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        logits: torch.Tensor = model.model(**encoded).logits

    entries: list[tuple[list[str], torch.Tensor]] = []
    for idx, words in enumerate(batched_words):
        word_logits: list[torch.Tensor] = []
        prev_word_idx: int | None = None

        for sub_idx, word_idx in enumerate(encoded.word_ids(batch_index=idx)):
            if word_idx is None or word_idx == prev_word_idx:
                continue
            word_logits.append(logits[idx, sub_idx])
            prev_word_idx = word_idx

        if not word_logits:
            continue
        entries.append((words, torch.stack(word_logits)))

    return entries


def get_softmax(
    entries: list[tuple[list[str], torch.Tensor]],
    label_biases: dict[str, float] | None = None,
    id2label: dict[int, str] | None = None,
) -> list[tuple[list[str], torch.Tensor]]:
    """Apply optional per-label logit bias then softmax. Shape preserved: (W, num_labels)."""
    import torch

    results = []
    for words, logit_tensor in entries:
        t = logit_tensor.clone()
        if label_biases and id2label:
            for label_id, label_name in id2label.items():
                if label_name in label_biases:
                    t[:, label_id] += label_biases[label_name]
        results.append((words, torch.softmax(t, dim=-1)))

    return results


def get_argmax(
    entries: list[tuple[list[str], torch.Tensor]],
    min_confidence_fn: (
        typing.Callable[[str], float | None] | None
    ) = None,
    o_label_id: int | None = None,
    id2label: dict[int, str] | None = None,
    keep_probs: bool = False,
) -> list[NERResult]:
    """Argmax over prob/logit tensor. Optionally fall back to O below per-label threshold."""
    import torch

    results: list[NERResult] = []
    for words, tensor in entries:
        conf_values = tensor.max(dim=-1).values.tolist()
        ids = torch.argmax(tensor, dim=-1).tolist()
        if min_confidence_fn is not None and o_label_id is not None and id2label is not None:
            ids = [
                lid if (
                    (threshold := min_confidence_fn(id2label[lid])) is None
                    or c >= threshold
                ) else o_label_id
                for lid, c in zip(ids, conf_values)
            ]
        results.append(NERResult(
            text=" ".join(words),
            words=words,
            label_ids=ids,
            confidences=conf_values,
            probs=tensor if keep_probs else None,
        ))

    return results


def infer_ner_pretokenized(
    model: NERModel,
    batch: list[tuple[str, list[str]]],
    label_biases: dict[str, float] | None = None,
    min_confidences: list[tuple[str | None, float]] | None = None,
    keep_probs: bool = False,
) -> list[NERResult]:
    """Thin wrapper: get_logits → get_softmax → get_argmax. No biasing."""
    o_label_id = next(k for k, v in model.id2label.items() if v == "O")
    return get_argmax(
        get_softmax(
            get_logits(model, batch),
            label_biases=label_biases,
            id2label=model.id2label,
        ),
        min_confidence_fn=(
            (lambda label: _resolve_label_float(min_confidences, label))
            if min_confidences
            else None
        ),
        o_label_id=o_label_id,
        id2label=model.id2label,
        keep_probs=keep_probs,
    )


def infer_ner_batch(
    tokenizer: stanza.Pipeline,
    model: NERModel,
    inputs: Iterable[str],
    batch_size: int = 32,
    label_biases: dict[str, float] | None = None,
    min_confidences: list[tuple[str | None, float]] | None = None,
    keep_probs: bool = False,
) -> list[NERResult]:
    results: list[NERResult] = []
    batch: list[tuple[str, list[str]]] = []
    auto_id = 0

    def flush() -> None:
        results.extend(infer_ner_pretokenized(
            model,
            batch,
            label_biases=label_biases,
            min_confidences=min_confidences,
            keep_probs=keep_probs,
        ))
        batch.clear()

    for text in tqdm(inputs, desc="Processing", unit="text"):
        words = [
            t.text
            for s in tokenizer(text).sentences
            for t in s.tokens
        ]
        batch.append((str(auto_id), words))
        auto_id += 1
        if len(batch) == batch_size:
            flush()

    if batch:
        flush()

    return results


def build_ner_model(
    model_name: str = "Babelscape/wikineural-multilingual-ner",
    device: str | None = None,
    dtype: str = "auto", # "auto" | "fp32" | "fp16" | "bf16"
) -> NERModel:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _DTYPE_MAP = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype == "auto":
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
    elif dtype in _DTYPE_MAP:
        torch_dtype = _DTYPE_MAP[dtype]
    else:
        raise ValueError(f"dtype must be auto/fp32/fp16/bf16, got {dtype!r}")

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
    model: AutoModelForTokenClassification = (
        AutoModelForTokenClassification.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype,
        ).to(device)
    )

    return NERModel(
        tokenizer=tokenizer,
        model=model,
        torch_dtype=torch_dtype,
        id2label={k: _normalize_label(v) for k, v in model.config.id2label.items()},
        device=device,
    )


class LabelFloatParam(click.ParamType):
    """Accepts 'LABEL FLOAT' (bias) or 'FLOAT' / 'LABEL FLOAT' (confidence)."""

    name = "label_float"

    def __init__(self, require_label: bool = False) -> None:
        self.require_label = require_label

    def convert(
        self, value: str, param: click.Parameter | None, ctx: click.Context | None
    ) -> tuple[str | None, float]:
        if isinstance(value, tuple):
            return value
        parts = value.split()
        if len(parts) == 1 and not self.require_label:
            try:
                return (None, float(parts[0]))
            except ValueError:
                self.fail(f"expected FLOAT, got {parts[0]!r}", param, ctx)
        elif len(parts) == 2:
            try:
                return (parts[0], float(parts[1]))
            except ValueError:
                self.fail(f"expected LABEL FLOAT, got {value!r}", param, ctx)
        else:
            fmt = "LABEL FLOAT" if self.require_label else "FLOAT or LABEL FLOAT"
            self.fail(f"expected {fmt}, got {value!r}", param, ctx)


def _resolve_label_float(
    pairs: list[tuple[str | None, float]],
    label: str,
) -> float | None:
    """Return per-label value if present, else global (None-keyed), else None."""
    per_label = {k: v for k, v in pairs if k is not None}
    global_val = next((v for k, v in pairs if k is None), None)
    return per_label.get(label, global_val)

# ----- CLI ------

@click.command()
@click.option(
    "--label-bias",
    "label_biases",
    multiple=True,
    type=LabelFloatParam(require_label=True),
    help="Per-label logit bias as 'LABEL FLOAT'. Repeatable. E.g. --label-bias MISC -3.0",
)
@click.option(
    "--min-confidence",
    "min_confidences",
    multiple=True,
    type=LabelFloatParam(require_label=False),
    help=(
        "Confidence threshold. 'FLOAT' sets global, 'LABEL FLOAT' sets per-label. "
        "Repeatable. Per-label overrides global."
    ),
)
@click.option(
    "--input",
    "input_texts",
    required=True,
    help="Input text(s) to run NER on. Repeatable.",
)
@click.option(
    "--model",
    "model_name",
    default="Babelscape/wikineural-multilingual-ner",
    show_default=True,
    help="HuggingFace model name.",
)
@click.option(
    "--lang",
    default="lv",
    show_default=True,
    help="Stanza tokenizer language.",
)
@click.option(
    "--batch-size",
    default=32,
    show_default=True,
    help="Inference batch size.",
)
@click.option(
    "--repeat",
    default=1,
    show_default=True,
    help="Repeat the input texts N times (stress testing).",
)
@click.option(
    "--show-softmax",
    is_flag=True,
    default=False,
    help="Print full softmax distribution for every word.",
)
@click.option(
    "--device",
    default=None,
    type=click.Choice(["cpu", "cuda"]),
    help="Device to run on. Auto-detected if omitted.",
)
@click.option(
    "--perf-only",
    is_flag=True,
    default=False,
    help="Suppress per-result output; print only benchmark stats.",
)
@click.option(
    "--dtype",
    type=click.Choice(["auto", "fp32", "fp16", "bf16"]),
    default="auto",
    show_default=True,
    help="Model weight dtype.",
)
def benchmark(
    input_texts: str,
    model_name: str,
    lang: str,
    batch_size: int,
    repeat: int,
    show_softmax: bool,
    device: str | None,
    perf_only: bool,
    label_biases: tuple[tuple[str, float], ...],
    min_confidences: tuple[tuple[str | None, float], ...],
) -> None:
    """Run a NER benchmark over provided texts."""
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    texts = None
    with open(input_texts, 'r') as f:
        texts = [l for l in f.read().split('\n') if l.strip()]

    expanded: list[str] = list(texts) * repeat

    click.echo(f"Initializing models on {device}...")
    _tokenizer: stanza.Pipeline = _stanza_tokenizer(
        lang=lang, use_gpu=(device == "cuda")
    )
    _model: NERModel = build_ner_model(model_name=model_name, device=device, dtype=dtype)

    biases: dict[str, float] | None = (
        {label: val for label, val in label_biases} if label_biases else None
    )
    confidences: list[tuple[str | None, float]] = list(min_confidences)

    click.echo(f"Starting benchmark for {len(expanded)} sample(s)...")
    start: float = time.time()
    results: list[NERResult] = infer_ner_batch(
        _tokenizer,
        _model,
        iter(expanded),
        batch_size=batch_size,
        label_biases=biases,
        min_confidences=confidences if confidences else None,
        keep_probs=show_softmax,
    )
    duration: float = time.time() - start

    if not perf_only:
        for res in results:
            click.echo(f"\nText: {res.text}")
            if show_softmax and res.probs is not None:
                raw_labels = list(_model.model.config.id2label.values())
                col_w = max(len(l) for l in raw_labels) + 4  # +4 = 2 brackets + 2 padding
                header = f"  {'word':<20}" + "".join(f"{l:>{col_w}}" for l in raw_labels)
                click.echo(header)
                for word, row in zip(res.words, res.probs.tolist()):
                    best = max(range(len(row)), key=lambda i: row[i])
                    num_w = col_w - 2  # space for the two brackets
                    vals = "".join(
                        f"[{v:>{col_w - 2}.3f}]" if i == best else f" {v:>{col_w - 2}.3f} "
                        for i, v in enumerate(row)
                    )
                    click.echo(f"  {word:<20}{vals}")
            else:
                entities = [
                    (w, _model.id2label[lid], conf)
                    for w, lid, conf in zip(res.words, res.label_ids, res.confidences)
                    if _model.id2label[lid] != "O"
                ]
                if entities:
                    for word, label, conf in entities:
                        click.echo(f"  {word:<20} -> {label:<10} ({conf:.3f})")
                else:
                    click.echo("  No entities found.")

    click.echo("\n" + "=" * 40)
    click.echo(f"Total Time:  {duration:.2f}s")
    click.echo(f"Throughput:  {len(expanded) / duration:.2f} texts/sec")
    click.echo(f"Device:      {device}")
    click.echo("=" * 40)


@click.group("ner")
def ner() -> None:
    """Named entity recognition model."""


ner.add_command(benchmark)

if __name__ == "__main__":
    ner()
