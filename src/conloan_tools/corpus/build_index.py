import datetime
import json
import os
import re
from pathlib import Path

import click
import h5py
import numpy as np

from conloan_tools.corpus import corpus
from conloan_tools.corpus.query import is_clean_word
from conloan_tools.wb.wb import WittenBellCharLM
from conloan_tools.ner.ner import NERModel, infer_ner_pretokenized, get_logits
from tqdm import tqdm

IDX_FLUSH_EVERY = 10_000
SENT_ID_RE = re.compile(r'id="([^"]+)"')

_GZIP_LEVEL   = 4
_CHUNK_TOKENS = 8_192
_CHUNK_SENTS  = 1_024


def _iter_surprisal_scores(
    lm: WittenBellCharLM,
    input_path: str,
    reduction: str,
    limit_lines: int | None,
    limit_sentences: int | None,
    limit_mb: float | None,
):
    for _, tokens, cpos in iter_vert_sentences(
        input_path,
        limit_lines=limit_lines,
        limit_sentences=limit_sentences,
        limit_mb=limit_mb,
    ):
        scores = [
            lm.compute_score(tok, reduction)
            if is_clean_word(tok, allow_ner=False)
            else 0.0
            for tok in tokens
        ]
        yield cpos, np.array(scores, dtype=np.float16)


def _iter_ner_scores(
    model: "NERModel",
    input_path: str,
    ner_output: str,
    batch_size: int,
    scores_dtype: type,
    num_labels: int | None,
    limit_lines: int | None,
    limit_sentences: int | None,
    limit_mb: float | None,
):
    import torch

    def _null_arr(n_tokens: int) -> np.ndarray:
        if ner_output == "logits":
            return np.zeros((n_tokens, num_labels), dtype=scores_dtype)
        else:
            return np.zeros((n_tokens,), dtype=np.uint8)

    batch: list[tuple[str, list[str], int]] = []

    def _process_batch() -> list[tuple[int, np.ndarray]]:
        logit_entries = get_logits(model, batch)
        # use id() rather than fragile list-equality or iterator alignment
        id_to_result: dict[int, np.ndarray] = {}

        if ner_output == "logits":
            for words, t in logit_entries:
                rows = (
                    t.cpu().numpy().astype(scores_dtype)
                    if scores_dtype == np.float16
                    else t.to(torch.float32).cpu().numpy()
                )
                arr = np.array(
                    [
                        row if is_clean_word(w, allow_ner=True) else np.zeros_like(row)
                        for w, row in zip(words, rows)
                    ],
                    dtype=scores_dtype,
                )
                id_to_result[id(words)] = arr
        else:
            # infer_ner_pretokenized needs (sent_id, words, cpos) triples
            surviving = [
                (None, words, cpos) for _, words, cpos in batch
                if id(words) in {id(w) for w, _ in logit_entries}
            ]
            results = infer_ner_pretokenized(model, surviving)
            for (words, _), result in zip(logit_entries, results):
                arr = np.array(
                    [
                        lid if is_clean_word(w, allow_ner=True) else 0
                        for w, lid in zip(words, result.label_ids)
                    ],
                    dtype=np.uint8,
                )
                id_to_result[id(words)] = arr

        return [
            (cpos, id_to_result.get(id(words), _null_arr(len(words))))
            for _, words, cpos in batch
        ]


    for sent_id, tokens, cpos in iter_vert_sentences(
        input_path,
        limit_lines=limit_lines,
        limit_sentences=limit_sentences,
        limit_mb=limit_mb,
    ):
        batch.append((sent_id, tokens, cpos))
        if len(batch) >= batch_size:
            yield from _process_batch()
            batch.clear()

    if batch:
        yield from _process_batch()


def _build_index_from_iter(
    score_iter,
    f: h5py.File,
    store_spos: bool,
    flush_every: int,
) -> None:
    cpos_buf:   list[int]        = []
    count_buf:  list[int]        = []
    spos_buf:   list[int] | None = [] if store_spos else None
    scores_buf: list[np.ndarray] = []
    spos = 0

    for cpos, arr in score_iter:
        cpos_buf.append(cpos)
        count_buf.append(arr.shape[0])
        if spos_buf is not None:
            spos_buf.append(spos)
        scores_buf.append(arr)
        spos += 1

        if len(cpos_buf) >= flush_every:
            _hdf5_flush(f, cpos_buf, count_buf, spos_buf, scores_buf, store_spos)

    _hdf5_flush(f, cpos_buf, count_buf, spos_buf, scores_buf, store_spos)


def iter_vert_sentences(
    path: str,
    limit_lines: int | None = None,
    limit_sentences: int | None = None,
    limit_mb: float | None = None,
):
    auto_id = 0
    current_id: str | None = None
    tokens: list[str] = []
    sentence_start_cpos = 0
    cpos_counter = 0  # global, advances for every token line

    processed_count = 0
    sentences_count = 0
    bytes_processed = 0
    byte_limit = limit_mb * 1024 * 1024 if limit_mb is not None else None
    total_bytes = os.path.getsize(path)

    with open(path, "rb") as f_bin, tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        desc="Processing Vert",
    ) as pbar:
        for line_bytes in f_bin:
            if limit_lines is not None and processed_count >= limit_lines:
                break
            if byte_limit is not None and bytes_processed >= byte_limit:
                break

            line_len = len(line_bytes)
            bytes_processed += line_len
            pbar.update(line_len)
            processed_count += 1

            if processed_count % 1000 == 0:
                pbar.set_postfix(lines=processed_count, sents=sentences_count)

            stripped = line_bytes.decode("utf-8").rstrip("\n")

            if stripped.startswith("<s") and (
                len(stripped) == 2 or stripped[2] in (" ", ">")
            ):
                if current_id is not None:
                    yield current_id, tokens, sentence_start_cpos
                    sentences_count += 1
                    if limit_sentences is not None and sentences_count >= limit_sentences:
                        return

                m = SENT_ID_RE.search(stripped)
                current_id = m.group(1) if m else str(auto_id)
                auto_id += 1
                tokens = []
                sentence_start_cpos = cpos_counter  # record here

            elif stripped.startswith("</s") and (
                len(stripped) == 3 or stripped[3] in (" ", ">")
            ):
                if current_id is not None:
                    yield current_id, tokens, sentence_start_cpos
                    sentences_count += 1
                    current_id = None
                    tokens = []
                    if limit_sentences is not None and sentences_count >= limit_sentences:
                        return

            elif stripped.startswith("<") and "\t" not in stripped:
                pass

            else:
                if not stripped:
                    continue
                if current_id is not None and not tokens:
                    sentence_start_cpos = cpos_counter  # first token of sentence
                cpos_counter += 1  # always advance, even outside <s>
                if current_id is not None:
                    parts = stripped.split("\t", 1)
                    tokens.append(parts[0])


# HDF5 layout (both surprisal and NER):
#
#   /                           ← file-level attrs carry all metadata
#   /index/cpos   uint64        ← absolute token offset of sentence start
#   /index/count  uint32        ← token count for that sentence
#   /index/spos   uint64        ← sentence ordinal (optional, store_spos flag)
#   /scores/data  dtype         ← flat token scores
#                                  surprisal : (total_tokens,)        float16
#                                  NER labels: (total_tokens,)        uint8
#                                  NER logits: (total_tokens, n_labels) float32


def _ds_opts(chunks: tuple) -> dict:
    return dict(
        compression="gzip",
        compression_opts=_GZIP_LEVEL,
        chunks=chunks,
    )


def _create_hdf5(
    path: Path,
    scores_dtype: np.dtype,
    meta: dict,
    store_spos: bool,
    num_labels: int | None = None,
) -> h5py.File:
    """
    Create a new HDF5 index file.

    num_labels=None  → scores/data is 1-D  (surprisal or NER labels mode)
    num_labels=int   → scores/data is 2-D  (NER logits mode)
    """
    f = h5py.File(path, "w")

    for k, v in meta.items():
        f.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v

    idx = f.create_group("index")
    so = _ds_opts((_CHUNK_SENTS,))
    idx.create_dataset("cpos",  shape=(0,), maxshape=(None,), dtype=np.uint64, **so)
    idx.create_dataset("count", shape=(0,), maxshape=(None,), dtype=np.uint32, **so)
    if store_spos:
        idx.create_dataset(
            "spos", shape=(0,), maxshape=(None,), dtype=np.uint64, **so
        )

    scores = f.create_group("scores")
    if num_labels is None:
        scores.create_dataset(
            "data",
            shape=(0,),
            maxshape=(None,),
            dtype=scores_dtype,
            **_ds_opts((_CHUNK_TOKENS,)),
        )
    else:
        scores.create_dataset(
            "data",
            shape=(0, num_labels),
            maxshape=(None, num_labels),
            dtype=scores_dtype,
            **_ds_opts((_CHUNK_TOKENS, num_labels)),
        )

    return f


def _hdf5_flush(
    f: h5py.File,
    cpos_buf: list[int],
    count_buf: list[int],
    spos_buf: list[int] | None,
    scores_buf: list[np.ndarray],
    store_spos: bool,
) -> None:
    if not cpos_buf:
        return

    idx       = f["index"]
    scores_ds = f["scores"]["data"]

    def _append(ds: h5py.Dataset, arr: np.ndarray) -> None:
        old = ds.shape[0]
        ds.resize(old + arr.shape[0], axis=0)
        ds[old:] = arr

    _append(idx["cpos"],  np.array(cpos_buf,  dtype=np.uint64))
    _append(idx["count"], np.array(count_buf, dtype=np.uint32))
    if store_spos and spos_buf is not None:
        _append(idx["spos"], np.array(spos_buf, dtype=np.uint64))
    _append(scores_ds, np.concatenate(scores_buf, axis=0))

    cpos_buf.clear()
    count_buf.clear()
    if spos_buf is not None:
        spos_buf.clear()
    scores_buf.clear()


def build_surprisal_index(
    lm: WittenBellCharLM,
    input_path: str,
    output_dir: str,
    name: str,
    reduction: str = "max",
    store_spos: bool = True,
    flush_every: int = IDX_FLUSH_EVERY,
    limit_lines: int | None = None,
    limit_sentences: int | None = None,
    limit_mb: float | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    h5_path = out / f"{name}.h5"

    meta = {
        "type":       "surprisal",
        "input":      str(input_path),
        "reduction":  reduction,
        "n":          lm.n,
        "model":      "WittenBellCharLM",
        "date":       datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dtype":      "float16",
        "store_spos": store_spos,
    }

    f = _create_hdf5(h5_path, np.float16, meta, store_spos, num_labels=None)
    try:
        _build_index_from_iter(
            _iter_surprisal_scores(
                lm, input_path, reduction,
                num_labels if ner_output == "logits" else None,
                limit_lines, limit_sentences, limit_mb,
            ),
            f, store_spos, flush_every,
        )
    finally:
        f.close()

    return h5_path


def build_ner_index(
    model: "NERModel",
    input_path: str,
    output_dir: str,
    name: str,
    ner_output: str = "logits",
    batch_size: int = 32,
    store_spos: bool = True,
    flush_every: int = IDX_FLUSH_EVERY,
    limit_lines: int | None = None,
    limit_sentences: int | None = None,
    limit_mb: float | None = None,
) -> Path:
    import torch

    if ner_output not in ("logits", "labels"):
        raise ValueError(f"ner_output must be 'logits' or 'labels', got {ner_output!r}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    h5_path = out / f"{name}.h5"

    id2label: dict[int, str] = {int(k): v for k, v in model.id2label.items()}
    num_labels = len(id2label)

    _TORCH_TO_NP: dict[torch.dtype, type] = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.bfloat16: np.float32,
    }
    if ner_output == "logits":
        torch_dtype = model.torch_dtype or next(model.model.parameters()).dtype
        scores_dtype = _TORCH_TO_NP.get(torch_dtype)
        if scores_dtype is None:
            raise ValueError(f"Unsupported model dtype {torch_dtype}")
        num_labels_arg = num_labels
    else:
        scores_dtype = np.uint8
        num_labels_arg = None

    meta = {
        "type":          "ner",
        "ner_output":    ner_output,
        "model":         getattr(model.model.config, "_name_or_path", "unknown"),
        "input":         str(input_path),
        "date":          datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dtype":         str(scores_dtype) if ner_output == "logits" else "uint8",
        "torch_dtype":   str(model.torch_dtype) if model.torch_dtype else "unknown",
        "bf16_promoted": ner_output == "logits" and model.torch_dtype == torch.bfloat16,
        "num_labels":    num_labels,
        "id2label":      id2label,
        "store_spos":    store_spos,
    }

    f = _create_hdf5(h5_path, scores_dtype, meta, store_spos, num_labels_arg)
    try:
        _build_index_from_iter(
            _iter_ner_scores(
                model, input_path, ner_output, batch_size, scores_dtype,
                num_labels if ner_output == "logits" else None,
                limit_lines, limit_sentences, limit_mb,
            ),
            f, store_spos, flush_every,
        )
    finally:
        f.close()

    return h5_path


@click.group("build-index")
def build_index():
    """Build validate HDF5 corpus index files."""

@build_index.command("surprisal")
@click.option("--wb-pkl",          required=True, help="Witten-Bell model (.pkl)")
@click.option("--input",           "input_path", required=True, help=".vert corpus")
@click.option("--output-dir",      required=True, help="Output directory")
@click.option("--name",            required=True, help="Base name for output file")
@click.option("--n",               default=3, show_default=True)
@click.option("--reduction",       type=click.Choice(["max", "mean"]), default="mean", show_default=True)
@click.option("--no-spos",         is_flag=True, default=False, help="Omit spos dataset")
@click.option("--limit-lines",     type=int)
@click.option("--limit-sentences", type=int)
@click.option("--limit-mb",        type=float)
def build_surprisal_index_command(
    wb_pkl, input_path, output_dir, name, n,
    reduction, no_spos, limit_lines, limit_sentences, limit_mb,
):
    """Score every token in a .vert corpus with surprisal. Streams to HDF5."""
    lm = WittenBellCharLM.load(wb_pkl)
    h5_path = build_surprisal_index(
        lm=lm,
        input_path=input_path,
        output_dir=output_dir,
        name=name,
        reduction=reduction,
        store_spos=not no_spos,
        limit_lines=limit_lines,
        limit_sentences=limit_sentences,
        limit_mb=limit_mb,
    )
    click.echo("[✓] Finished", err=True)
    click.echo(f"    {h5_path} ({h5_path.stat().st_size:,} B)", err=True)


@build_index.command("ner")
@click.option("--model-name",      default="Babelscape/wikineural-multilingual-ner", show_default=True)
@click.option("--input",           "input_path", required=True, help=".vert corpus")
@click.option("--output-dir",      required=True, help="Output directory")
@click.option("--name",            required=True, help="Base name for output file")
@click.option("--lang",            default="lv", show_default=True)
@click.option("--batch-size",      default=32, show_default=True)
@click.option("--device",          default=None, type=click.Choice(["cpu", "cuda"]))
@click.option("--ner-output",      type=click.Choice(["logits", "labels"]), default="logits", show_default=True)
@click.option("--no-spos",         is_flag=True, default=False, help="Omit spos dataset")
@click.option("--limit-lines",     type=int)
@click.option("--limit-sentences", type=int)
@click.option("--limit-mb",        type=float)
@click.option(
    "--dtype",
    type=click.Choice(["auto", "fp32", "fp16", "bf16"]),
    default="auto",
    show_default=True,
    help="Model weight dtype. bf16 logits are stored as fp32.",
)
def build_ner_index_command(
    model_name, input_path, output_dir, name, lang,
    batch_size, device, ner_output, dtype, no_spos,
    limit_lines, limit_sentences, limit_mb,
) -> None:
    from conloan_tools.ner.ner import build_ner_model

    click.echo(f"Device {device}")
    _model = build_ner_model(model_name=model_name, device=device, dtype=dtype)
    if ner_output == "logits" and dtype == "bf16":
        click.echo("[!] bf16 model: logits will be stored as fp32 (HDF5 limitation)", err=True)
    h5_path = build_ner_index(
        model=_model,
        input_path=input_path,
        output_dir=output_dir,
        name=name,
        ner_output=ner_output,
        batch_size=batch_size,
        store_spos=not no_spos,
        limit_lines=limit_lines,
        limit_sentences=limit_sentences,
        limit_mb=limit_mb,
    )
    click.echo("[✓] Finished", err=True)
    click.echo(f"    {h5_path} ({h5_path.stat().st_size:,} B)", err=True)


if __name__ == "__main__":
    corpus()
