import json
import sys
from pathlib import Path

import click
import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_h5(path: str) -> h5py.File:
    p = Path(path)
    if not p.exists():
        click.echo(f"Error: file not found: {path}", err=True)
        sys.exit(1)
    return h5py.File(p, "r")


def _attrs(f: h5py.File) -> dict:
    result = {}
    for k, v in f.attrs.items():
        try:
            result[k] = json.loads(v)
        except (json.JSONDecodeError, TypeError):
            result[k] = v
    return result


def _is_logits(f: h5py.File) -> bool:
    return f["scores"]["data"].ndim == 2


def _is_ner(attrs: dict) -> bool:
    return attrs.get("type") == "ner"


def _id2label(attrs: dict) -> dict[int, str] | None:
    raw = attrs.get("id2label")
    if raw is None:
        return None
    # keys come back as strings from JSON
    return {int(k): v for k, v in raw.items()}


def _find_sentence_by_cpos(
    cpos_ds: h5py.Dataset, target: int
) -> tuple[int, int]:
    """
    Binary search over cpos array.
    Returns (spos_idx, cpos_value) for the sentence that *contains* target.
    i.e. the last sentence whose cpos <= target.
    """
    lo, hi = 0, cpos_ds.shape[0] - 1
    result = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        val = int(cpos_ds[mid])
        if val <= target:
            result = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return result, int(cpos_ds[result])


def _get_sentence_scores(
    f: h5py.File, row: int
) -> np.ndarray:
    offset = int(f["index"]["cpos"][row])
    count  = int(f["index"]["count"][row])
    return f["scores"]["data"][offset : offset + count]


def _fmt_scores_1d(
    arr: np.ndarray,
    head: int,
    dtype_name: str,
) -> str:
    shown = arr[:head]
    suffix = f"  ... (+{len(arr) - head} more)" if len(arr) > head else ""
    vals = "  ".join(f"{v:.4f}" if np.issubdtype(arr.dtype, np.floating) else str(v) for v in shown)
    return f"[{vals}]{suffix}  dtype={dtype_name}  shape={arr.shape}"


def _fmt_scores_2d(
    arr: np.ndarray,
    id2label: dict[int, str] | None,
    head_tokens: int,
    head_labels: int,
) -> list[str]:
    """Return one line per token (up to head_tokens)."""
    lines = []
    for i, row in enumerate(arr[:head_tokens]):
        argmax = int(np.argmax(row))
        label  = id2label[argmax] if id2label else str(argmax)
        top    = row[:head_labels]
        vals   = "  ".join(f"{v:+.4f}" for v in top)
        suffix = f"  ... (+{len(row) - head_labels} more)" if len(row) > head_labels else ""
        lines.append(f"  token[{i:>4}]  argmax={label:<12} logits=[{vals}]{suffix}")
    if len(arr) > head_tokens:
        lines.append(f"  ... (+{len(arr) - head_tokens} more tokens)")
    return lines


def _fmt_scores_labels(
    arr: np.ndarray,
    id2label: dict[int, str] | None,
    head_tokens: int,
) -> list[str]:
    """Return one line per token for NER labels mode (uint8 IDs)."""
    lines = []
    for i, lid in enumerate(arr[:head_tokens]):
        label = id2label[int(lid)] if id2label else str(int(lid))
        lines.append(f"  token[{i:>4}]  label={label}")
    if len(arr) > head_tokens:
        lines.append(f"  ... (+{len(arr) - head_tokens} more tokens)")
    return lines


def _print_sentence_scores(
    arr: np.ndarray,
    attrs: dict,
    f: h5py.File,
    head_tokens: int,
    head_labels: int,
) -> None:
    """Dispatch score display based on index type."""
    id2label = _id2label(attrs)
    if _is_logits(f):
        click.echo("scores (logits):")
        for line in _fmt_scores_2d(arr, id2label, head_tokens, head_labels):
            click.echo(line)
    elif _is_ner(attrs):
        click.echo("scores (NER labels):")
        for line in _fmt_scores_labels(arr, id2label, head_tokens):
            click.echo(line)
    else:
        click.echo(
            "scores : "
            + _fmt_scores_1d(arr, head_tokens, str(arr.dtype))
        )


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group("inspect-index")
def inspect_index():
    """Inspect and validate HDF5 corpus index files."""


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@inspect_index.command("info")
@click.argument("path")
def cmd_info(path: str) -> None:
    """Dump file attributes and dataset shapes."""
    with _open_h5(path) as f:
        attrs = _attrs(f)

        click.echo("=== Attributes ===")
        for k, v in attrs.items():
            if isinstance(v, dict):
                click.echo(f"  {k}:")
                for ik, iv in v.items():
                    click.echo(f"    {ik}: {iv}")
            else:
                click.echo(f"  {k}: {v}")

        click.echo("\n=== Datasets ===")
        def _visit(name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset):
                click.echo(
                    f"  /{name:<30}  shape={str(obj.shape):<20}  dtype={obj.dtype}"
                )
        f.visititems(_visit)


# ---------------------------------------------------------------------------
# sent
# ---------------------------------------------------------------------------


@inspect_index.command("sent")
@click.argument("path")
@click.option(
    "--spos", "lookup_spos", type=int, default=None,
    help="Look up sentence by ordinal.",
)
@click.option(
    "--cpos", "lookup_cpos", type=int, default=None,
    help="Look up sentence containing this token position.",
)
@click.option(
    "--head-tokens", default=8, show_default=True,
    help="Max tokens to display.",
)
@click.option(
    "--head-labels", default=5, show_default=True,
    help="Max logit columns to display per token (logits mode only).",
)
def cmd_sent(
    path: str,
    lookup_spos: int | None,
    lookup_cpos: int | None,
    head_tokens: int,
    head_labels: int,
) -> None:
    """Look up one sentence by --spos or --cpos and show its scores."""
    if (lookup_spos is None) == (lookup_cpos is None):
        click.echo("Error: provide exactly one of --spos or --cpos.", err=True)
        sys.exit(1)

    with _open_h5(path) as f:
        attrs    = _attrs(f)
        cpos_ds  = f["index"]["cpos"]
        count_ds = f["index"]["count"]
        n_sents  = cpos_ds.shape[0]

        if lookup_spos is not None:
            if lookup_spos < 0 or lookup_spos >= n_sents:
                click.echo(
                    f"Error: spos {lookup_spos} out of range [0, {n_sents - 1}].",
                    err=True,
                )
                sys.exit(1)
            row      = lookup_spos
            cpos_val = int(cpos_ds[row])
        else:
            row, cpos_val = _find_sentence_by_cpos(cpos_ds, lookup_cpos)

        count = int(count_ds[row])
        spos_val = (
            int(f["index"]["spos"][row])
            if "spos" in f["index"]
            else row
        )

        click.echo(f"spos   : {spos_val}")
        click.echo(f"cpos   : {cpos_val}")
        click.echo(f"count  : {count} tokens")

        arr = _get_sentence_scores(f, row)
        _print_sentence_scores(arr, attrs, f, head_tokens, head_labels)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@inspect_index.command("validate")
@click.argument("path")
def cmd_validate(path: str) -> None:
    """
    Check internal consistency of an index file.

      - cpos[i+1] == cpos[i] + count[i] for all i
      - spos is strictly monotone (if present)
      - scores/data.shape[0] == sum(count)
    """
    errors = 0

    def _err(msg: str) -> None:
        nonlocal errors
        click.echo(f"  [FAIL] {msg}")
        errors += 1

    with _open_h5(path) as f:
        cpos_ds  = f["index"]["cpos"]
        count_ds = f["index"]["count"]
        scores_ds = f["scores"]["data"]
        n_sents  = cpos_ds.shape[0]

        click.echo(f"Sentences : {n_sents:,}")
        click.echo(f"Tokens    : {scores_ds.shape[0]:,}")
        click.echo("Checking...")

        # Load index arrays in one shot — they're small even for large corpora
        cpos  = cpos_ds[:]
        count = count_ds[:]

        # 1. cpos continuity
        expected = cpos[0]
        for i in range(n_sents):
            if cpos[i] != expected:
                _err(
                    f"cpos discontinuity at spos={i}: "
                    f"expected {expected}, got {cpos[i]}"
                )
                # re-sync so we don't cascade errors
                expected = cpos[i]
            expected += count[i]

        # 2. total token count
        total = int(count.sum())
        actual = scores_ds.shape[0]
        if total != actual:
            _err(f"sum(count)={total:,} != scores/data.shape[0]={actual:,}")

        # 3. spos monotonicity
        if "spos" in f["index"]:
            spos = f["index"]["spos"][:]
            bad  = np.where(np.diff(spos.astype(np.int64)) != 1)[0]
            for b in bad[:10]:  # cap noise
                _err(
                    f"spos not strictly +1 at row {b}: "
                    f"{spos[b]} -> {spos[b + 1]}"
                )
        else:
            click.echo("  spos: not stored (skipped)")

    if errors == 0:
        click.echo("  [OK] all checks passed.")
    else:
        click.echo(f"\n  {errors} error(s) found.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------


@inspect_index.command("sample")
@click.argument("path")
@click.option("--n", default=5, show_default=True, help="Number of sentences.")
@click.option("--seed", default=42, show_default=True)
@click.option("--head-tokens", default=8, show_default=True)
@click.option("--head-labels", default=5, show_default=True)
def cmd_sample(
    path: str, n: int, seed: int, head_tokens: int, head_labels: int
) -> None:
    """Print N random sentences with their scores."""
    with _open_h5(path) as f:
        attrs   = _attrs(f)
        n_sents = f["index"]["cpos"].shape[0]
        rng     = np.random.default_rng(seed)
        rows    = sorted(rng.choice(n_sents, size=min(n, n_sents), replace=False))

        for row in rows:
            cpos_val = int(f["index"]["cpos"][row])
            count    = int(f["index"]["count"][row])
            spos_val = (
                int(f["index"]["spos"][row])
                if "spos" in f["index"]
                else row
            )
            arr = _get_sentence_scores(f, row)

            click.echo(f"\n--- spos={spos_val}  cpos={cpos_val}  count={count} ---")
            _print_sentence_scores(arr, attrs, f, head_tokens, head_labels)


# ---------------------------------------------------------------------------
# hist
# ---------------------------------------------------------------------------


@inspect_index.command("hist")
@click.argument("path")
@click.option("--bins", default=20, show_default=True)
@click.option(
    "--max-sentences", default=50_000, show_default=True,
    help="Cap sentences sampled for histogram (0 = all).",
)
def cmd_hist(path: str, bins: int, max_sentences: int) -> None:
    """
    Print ASCII histogram of score values.

    Surprisal        : histogram of raw float scores.
    NER labels       : frequency count per label ID.
    NER logits       : frequency count of per-token argmax label.
    """
    with _open_h5(path) as f:
        attrs     = _attrs(f)
        scores_ds = f["scores"]["data"]
        n_tokens  = scores_ds.shape[0]
        logits    = _is_logits(f)
        ner       = _is_ner(attrs)
        id2label  = _id2label(attrs)

        cap = max_sentences
        if cap and cap > 0:
            count_ds = f["index"]["count"][:]
            n_cap    = min(cap, len(count_ds))
            n_tok    = int(count_ds[:n_cap].sum())
            data_raw = scores_ds[:n_tok]
            click.echo(
                f"Sampling first {n_cap:,} sentences ({n_tok:,} / {n_tokens:,} tokens)"
            )
        else:
            data_raw = scores_ds[:]

        if logits:
            data = np.argmax(data_raw, axis=1).astype(np.int32)
            click.echo("Mode: argmax label index over logits\n")
        elif ner:
            data = data_raw[:].astype(np.int32)
            click.echo("Mode: NER label IDs\n")
        else:
            data = data_raw[:].astype(np.float32)
            click.echo(f"Mode: raw scores  mean={data.mean():.4f}  std={data.std():.4f}\n")

        if logits or ner:
            n_classes = len(id2label) if id2label else int(data.max()) + 1
            counts    = np.bincount(data, minlength=n_classes)
            edges     = np.arange(n_classes + 1, dtype=np.float32)
        else:
            counts, edges = np.histogram(data, bins=bins)

        max_count = counts.max()
        bar_width = 50

        for i, c in enumerate(counts):
            bar = "█" * int(c / max_count * bar_width)
            if (logits or ner) and id2label:
                label = id2label.get(i, str(i))
                tag   = f"{label:<14}"
            else:
                lo, hi = edges[i], edges[i + 1]
                tag = f"{lo:>8.4f} – {hi:<8.4f}"
            click.echo(f"  {tag}  {bar:<{bar_width}}  {c:>8,}")
