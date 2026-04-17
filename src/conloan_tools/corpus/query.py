from contextlib import contextmanager
import sys
import subprocess
import os
import tempfile
import click
import json
from scipy.special import softmax
from typing import List, Optional, Tuple, Literal, Iterator
import re
from pathlib import Path
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
import h5py
import importlib

from conloan_tools.corpus import corpus
from .scoring import (
    Token,
    CQPResult,
    QueryProfile,
    ScoringConfig,
    ScoredResult,
    load_scoring_config,
    score_sentence,
    build_loanword_mask,
    build_named_entity_mask,
)


DEFAULT_CQP_BIN = "cqp"
DEFAULT_LOOKUP  = 200
DEFAULT_RESULTS = 20

# ------ Data types -------

@dataclass(slots=True)
class IndexRecord:
    offset: int
    count: int


class IndexRecordList:
    """Mimics list[IndexRecord] but backed by numpy arrays."""
    def __init__(self, offsets: np.ndarray, counts: np.ndarray):
        self._offsets = offsets
        self._counts  = counts

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return IndexRecordList(self._offsets[i], self._counts[i])
        return IndexRecord(int(self._offsets[i]), int(self._counts[i]))

    def __iter__(self):
        for i in range(len(self)):
            yield IndexRecord(int(self._offsets[i]), int(self._counts[i]))


def _build_records(count: np.ndarray) -> IndexRecordList:
    offsets = np.concatenate([[0], np.cumsum(count[:-1])])
    return IndexRecordList(offsets, count)


@dataclass
class CodeSwitchRun:
    sent_idx: int
    token_indices: list[int]
    token_scores: list[float]
    tokens: list[Token]
    metrics: ScoredResult 

# ------ Internal -------

@dataclass
class CandidateRecord:
    mode: str
    sentence: str
    score_total: float
    score_lw: float
    score_cs: float
    score_ne: float
    score_length: float
    score_alpha: float
    cqp_id: int
    cpos: int
    spos: int | None
    seed: int | None
    filtered: bool
    filter_reason: str | None
    matched_lemmas: list[str]
    tag_map: dict[str, str]  # e.g. {"L1": "computer", "CS1": "boot"}
    token_surprisals: dict[str, float] = field(default_factory=dict)


def _write_record(record: CandidateRecord, fh) -> None:
    import dataclasses
    fh.write(json.dumps(dataclasses.asdict(record), ensure_ascii=False) + "\n")


def _collapse_spans(
    tokens: list[tuple[str, bool]],
    tag: str,
) -> str:
    parts = []
    span: list[str] = []
    counter = 1

    def _flush():
        nonlocal counter
        if span:
            parts.append(f"<{tag}{counter}>{' '.join(span)}</{tag}{counter}>")
            span.clear()
            counter += 1

    for word, tagged in tokens:
        if tagged:
            span.append(word)
        else:
            _flush()
            parts.append(word)
    _flush()

    return " ".join(parts)


def _collapse_ne_spans(
    tokens: list[tuple[str, str | None]],
) -> str:
    parts = []
    span: list[str] = []
    current_label: str | None = None
    counter = 1

    def _flush():
        nonlocal counter
        if span:
            parts.append(f"<NE{counter}>{' '.join(span)}</NE{counter}>")
            span.clear()
            counter += 1

    for word, label in tokens:
        if label is not None:
            if label != current_label:
                _flush()
            span.append(word)
            current_label = label
        else:
            _flush()
            current_label = None
            parts.append(word)
    _flush()

    return " ".join(parts)


def _tag_code_switch_sentence(run: CodeSwitchRun) -> str:
    index_set = set(run.token_indices)
    tokens = [(t.word, i in index_set) for i, t in enumerate(run.tokens)]
    return _collapse_spans(tokens, "CS")


def tag_all_loanwords(parsed_result, lemma_set_lower, primary_lemma):
    """Tag all loanwords in sentence with L1, L2, L3… (each token individually)."""
    if parsed_result is None:
        return None

    loanword_positions = []
    for i, token in enumerate(parsed_result.tokens):
        token_lemma = token.lemma.lower()
        if token_lemma in lemma_set_lower:
            is_primary = token_lemma == primary_lemma.lower()
            loanword_positions.append((i, token.word, is_primary))

    if not loanword_positions:
        return None

    loanword_positions.sort(key=lambda x: (not x[2], x[0]))

    tag_map = {}
    for tag_num, (pos, _, _) in enumerate(loanword_positions, start=1):
        tag_map[pos] = tag_num

    parts = []
    for i, t in enumerate(parsed_result.tokens):
        if i in tag_map:
            n = tag_map[i]
            parts.append(f"<L{n}>{t.word}</L{n}>")
        else:
            parts.append(t.word)

    return " ".join(parts)


def _tag_ner_sentence(
    parsed: CQPResult,
    sent_idx: int,
    ner_labels: np.ndarray,
    ner_records: list[IndexRecord],
    id2label: dict[int, str],
    want: set[int],
) -> str:
    record = ner_records[sent_idx]
    chunk = ner_labels[record.offset : record.offset + record.count]
    tokens = [
        (
            t.word,
            id2label[int(chunk[i])]
            if i < len(chunk) and int(chunk[i]) in want
            else None,
        )
        for i, t in enumerate(parsed.tokens)
    ]
    return _collapse_ne_spans(tokens)


def _result_to_record(
    mode: str,
    sentence: str,
    scored: ScoredResult,
    matched_lemmas: list[str],
    tag_map: dict[str, str],
    token_surprisals: dict[str, float],
    cpos: int,
    spos: int | None,
    seed: int | None,
) -> CandidateRecord:
    return CandidateRecord(
        mode=mode,
        sentence=sentence,
        score_total=scored.score_total,
        score_lw=scored.score_loanword,
        score_cs=scored.score_code_switch,
        score_ne=scored.score_named_entity,
        score_length=scored.score_length,
        score_alpha=scored.score_alpha,
        cqp_id=scored.cqp_id,
        cpos=cpos,
        spos=spos,
        seed=seed,
        filtered=scored.filtered,
        filter_reason=scored.filter_reason,
        matched_lemmas=matched_lemmas,
        tag_map=tag_map,
        token_surprisals=token_surprisals,

    )


def _make_lingua_detector():
    from lingua import LanguageDetectorBuilder
    return LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()


def _get_lingua_language(lang_code: str):
    """Get Lingua Language enum from ISO 639-1 code, or None if invalid."""
    from lingua import Language
    try:
        return Language.from_iso_code_639_1(lang_code.lower())
    except (ValueError, AttributeError):
        return None


@dataclass
class MaskSources:
    surprisal_scores: Optional[np.ndarray] = None
    surprisal_records: Optional[list[IndexRecord]] = None
    surprisal_cpos: Optional[np.ndarray] = None
    surprisal_reduction: str = "mean"
    surprisal_threshold: float = 0.0
    lingua_context_threshold: float = 0.0
    lingua_span_threshold: float = 0.0
    ner_labels: Optional[np.ndarray] = None
    ner_records: Optional[list[IndexRecord]] = None
    ner_cpos: Optional[np.ndarray] = None
    ner_id2label: Optional[dict[int, str]] = None
    ner_exclude: Optional[set[int]] = None
    lw_lemmas: Optional[set[str]] = None
    ner_confidence: Optional[np.ndarray] = None
    ner_confidence_thresholds: dict[str, float] = field(default_factory=dict)
    ner_ignore_labels: set[str] = field(default_factory=set)


def _assert_index_alignment(
    a: list[IndexRecord], b: list[IndexRecord]
) -> None:
    if len(a) != len(b):
        raise click.UsageError(
            f"Index alignment mismatch: {len(a)} vs {len(b)} sentences."
        )


def _lookup_sent_idx(
    cpos: int,
    cpos_array: np.ndarray,          # keep as ndarray
    records: list[IndexRecord],
) -> int | None:
    i = int(np.searchsorted(cpos_array, cpos, side="right")) - 1
    if i < 0:
        return None
    rec = records[i]
    if cpos_array[i] <= cpos < cpos_array[i] + rec.count:
        return i
    return None


def _resolve_registry(
    registry_dir: Optional[str],
) -> tuple[str, str]:
    if registry_dir:
        return registry_dir, f"{registry_dir}/registry"
    env = os.environ.get("CORPUS_REGISTRY")
    if env:
        return os.path.dirname(env), env
    raise click.UsageError(
        "--registry-dir not provided and CORPUS_REGISTRY not set."
    )

def _run_cqp_command(
    corpus: str, 
    commands: List[str], 
    cqp_bin: str, 
    registry_dir: str, 
    registry: str
) -> str:
    """Internal helper to execute CQP commands."""

    # These are session-level settings that apply to all commands
    session_setup = [
        f"{corpus};",
        "set Context s;",
        "set PrintMode sgml;",
        "show -pos -lemma;",
        "show +pos +lemma;",
        "set PrintOptions noheader;",
    ]
    
    # Create a batch script with all commands
    script_lines = session_setup + commands
    # click.echo(f"[DEBUG] CQP script:\n{chr(10).join(script_lines)}", err=True)
    
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".cqp"
    ) as tf:
        tf.write("\n".join(script_lines))
        temp_file_path = tf.name

    try:
        process = subprocess.Popen(
            [cqp_bin, "-r", registry, "-S", "-f", temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=registry_dir,
        )
        stdout, stderr = process.communicate()
        if stderr.strip():
            click.echo(f"[CQP stderr]: {stderr.strip()}", err=True)
        if process.returncode != 0:
            raise click.ClickException(f"CQP Error: {stderr}")
        return stdout
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def _scan_runs_vectorized(
    flat: np.ndarray,
    offsets: np.ndarray,   # int64, one per sentence
    counts: np.ndarray,    # int32, one per sentence
    threshold: float,
    min_consecutive: int,
    max_consecutive: int | None = None,
    allowed: set[int] | None = None,
) -> list[tuple[int, list[int], list[float]]]:
    """
    Returns (sent_idx, token_indices, token_scores) for the best run per
    sentence, only for sentences that have at least one qualifying run.
    """
    n_sents = len(offsets)

    # 1. Boolean mask over the full flat slice
    above = (flat > threshold)

    # 2. Build sentence-id per token so we can reject cross-boundary runs.
    sent_ids = np.repeat(np.arange(n_sents, dtype=np.int32), counts)

    # 3. Find run starts / ends with a single diff pass.
    padded = np.empty(len(above) + 2, dtype=np.int8)
    padded[0] = 0
    padded[1:-1] = above.view(np.int8)
    padded[-1] = 0
    diff = np.diff(padded)
    run_starts = np.where(diff == 1)[0]   # inclusive
    run_ends   = np.where(diff == -1)[0]  # exclusive

    if run_starts.size == 0:
        return []

    run_lengths = run_ends - run_starts

    # 4. Length gate
    valid = run_lengths >= min_consecutive
    if max_consecutive is not None:
        valid &= (run_lengths <= max_consecutive)
    if not valid.any():
        return []
    run_starts  = run_starts[valid]
    run_ends    = run_ends[valid]
    run_lengths = run_lengths[valid]

    # 5. Discard runs that cross sentence boundaries (rare but possible).
    same_sent = sent_ids[run_starts] == sent_ids[run_ends - 1]
    run_starts  = run_starts[same_sent]
    run_ends    = run_ends[same_sent]
    run_lengths = run_lengths[same_sent]

    if run_starts.size == 0:
        return []

    run_sent_ids = sent_ids[run_starts]

    # 6. Keep only the longest run per sentence.
    #    Process in length-descending order; first hit per sentence wins.
    order = np.argsort(-run_lengths)
    run_starts   = run_starts[order]
    run_ends     = run_ends[order]
    run_lengths  = run_lengths[order]
    run_sent_ids = run_sent_ids[order]

    seen: set[int] = set()
    output: list[tuple[int, list[int], list[float]]] = []
    for s, e, sid in zip(run_starts, run_ends, run_sent_ids):
        sid = int(sid)
        if sid in seen:
            continue
        seen.add(sid)
        # Token indices are relative to sentence start
        sent_start = int(offsets[sid])
        indices = list(range(int(s) - sent_start, int(e) - sent_start))
        scores  = flat[s:e].tolist()
        output.append((sid, indices, scores))

    # Re-sort by sent_idx for stable downstream behaviour
    output.sort(key=lambda x: x[0])
    if allowed is not None:
        output = [o for o in output if o[0] in allowed]
    return output


def _chunked_read(ds, out: np.ndarray, desc: str) -> None:
    """Read an h5py dataset into a pre-allocated array with a progress bar."""
    n = ds.shape[0]
    chunk = ds.chunks[0] if ds.chunks else _CHUNK_TOKENS
    click.echo(f"[*] {desc}: {n:,} tokens in chunks of {chunk:,}", err=True)
    with tqdm(total=n, unit="tok", desc=desc, leave=False) as pbar:
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            out[start:end] = ds[start:end]
            pbar.update(end - start)


def _load_scores(h5_path: Path, reduction: str = "mean") -> np.ndarray:
    click.echo(f"[*] Opening scores index: {h5_path}", err=True)
    with h5py.File(h5_path, "r") as f:
        available = list(f["scores"].keys())
        if reduction not in available:
            raise click.UsageError(
                f"Reduction '{reduction}' not in index. Available: {available}"
            )
        ds = f["scores"][reduction]
        out = np.empty(ds.shape[0], dtype=np.float32)
        _chunked_read(ds, out, f"Loading scores {h5_path.name}")
    return out


def _load_index_records(
    h5_path: Path,
) -> tuple[list[IndexRecord], np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        cpos  = f["index"]["cpos"][:]   # small, no chunking needed
        count = f["index"]["count"][:]
    return _build_records(count), cpos


def _load_ner_labels(
    h5_path: Path,
) -> tuple[np.ndarray, Optional[np.ndarray], list[IndexRecord], dict[int, str], np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        ner_output   = f.attrs.get("ner_output", "labels")
        cpos         = f["index"]["cpos"][:]
        count        = f["index"]["count"][:]
        raw_id2label = json.loads(f.attrs["id2label"])
        ds           = f["scores"]["data"]
        click.echo(
            f"[*] NER index: ner_output={ner_output}  "
            f"shape={ds.shape}  dtype={ds.dtype}",
            err=True,
        )

        if ner_output == "logits":
            if ds.ndim != 2:
                raise click.UsageError(
                    f"{h5_path.name}: expected 2-D logits, got {ds.shape}."
                )
            n_tokens, n_labels = ds.shape
            labels = np.empty(n_tokens, dtype=np.uint8)
            confidence = np.empty(n_tokens, dtype=np.float32)
            chunk = ds.chunks[0] if ds.chunks else _CHUNK_TOKENS
            with tqdm(total=n_tokens, unit="tok", desc=f"Loading NER logits {h5_path.name}", leave=False) as pbar:
                for start in range(0, n_tokens, chunk):
                    end = min(start + chunk, n_tokens)
                    chunk_logits = ds[start:end]
                    probs = softmax(chunk_logits.astype(np.float32), axis=-1)
                    labels[start:end]     = np.argmax(probs, axis=-1).astype(np.uint8)
                    confidence[start:end] = probs.max(axis=-1)
                    pbar.update(end - start)
        elif ner_output == "labels":
            labels = np.empty(ds.shape[0], dtype=np.uint8)
            _chunked_read(ds, labels, f"Loading NER labels {h5_path.name}")
            confidence = None
        else:
            raise click.UsageError(
                f"{h5_path.name}: unknown ner_output='{ner_output}'."
            )

    click.echo(
        f"[*] NER index: building records", err=True,
    )
    id2label = {int(k): v for k, v in raw_id2label.items()}
    records  = _build_records(count)

    click.echo(
        f"[*] NER labels loaded: {len(records):,} sentences  "
        f"{labels.nbytes / 1024**2:.1f} MB  "
        f"labels={sorted(set(id2label.values()))}",
        err=True,
    )
    return labels, confidence, records, id2label, cpos


def _apply_ner_confidence_filter(
    labels: np.ndarray,
    confidence: np.ndarray,
    o_id: int,
    id2label: dict[int, str],
    thresholds: dict[str, float],
) -> np.ndarray:
    """
    Return a copy of `labels` where any run of non-O tokens whose first token
    has confidence below the label-specific threshold is replaced with o_id.
    Labels not present in `thresholds` are left untouched.
    """
    if not thresholds:
        return labels

    out = labels.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i] == o_id:
            i += 1
            continue
        run_start = i
        while i < n and out[i] != o_id:
            i += 1
        run_end = i  # exclusive
        label_name = id2label.get(int(out[run_start]), "O")
        thr = thresholds.get(label_name)
        if thr is not None and confidence[run_start] < thr:
            out[run_start:run_end] = o_id
    return out

def _parse_label_thresholds(raw: tuple[str, ...]) -> dict[str, float]:
    """Parse ('MISC:0.8', 'ORG:0.6') into {'MISC': 0.8, 'ORG': 0.6}."""
    result = {}
    for entry in raw:
        try:
            label, val = entry.rsplit(":", 1)
            result[label.strip()] = float(val)
        except ValueError:
            raise click.UsageError(
                f"Invalid --ner-confidence-threshold: {entry!r}. "
                "Expected format LABEL:VALUE (e.g. MISC:0.8)."
            )
    return result


def _render_code_switch_results(
    results: list[CodeSwitchRun],
    threshold: float,
) -> None:
    click.echo(f"\nTop results for code-switch sequences (threshold={threshold:.2f}):")
    click.echo("-" * 60)

    for run in results:
        res = run.metrics
        index_set = set(run.token_indices)

        status = f"[FILTERED: {res.filter_reason}]" if res.filtered else ""
        click.echo(
            f"Score: {res.score_total:.4f}  "
            f"(len={res.score_length:.2f}"
            f"  lw={res.score_loanword:.2f}"
            f"  cs={res.score_code_switch:.2f}"
            f"  alpha={res.score_alpha:.2f}  "
            f"  ne={res.score_named_entity:.2f})"
            f"  | Pos: {run.sent_idx}  ID: {res.cqp_id}  {status}"
        )

        surp_parts = "  ".join(
            f"{tag}={word!r} surprisal={rec.token_surprisals.get(tag, 0.0):.3f}"
            for tag, word in rec.tag_map.items()
            if tag.startswith("CS")
        )
        click.echo(f"  ↳ surprisal: [{surp_parts}]")
        click.echo("-" * 60)

    click.echo(f"({len(results)} sequences shown)")

# ------ Public -------

def load_mask_sources(
    surprisal_h5: Optional[Path] = None,
    surprisal_threshold: float = 0.0,
    surprisal_reduction: str = "mean",
    ner_h5: Optional[Path] = None,
    ner_confidence_thresholds: dict[str, float] = {},
    loanword_file: Optional[Path] = None,
    ner_ignore_labels: set[str] = frozenset(),
    lingua_context_threshold: float = 0.0,
    lingua_span_threshold: float = 0.0,
) -> MaskSources:
    """Load all optional index files into a MaskSources bundle."""
    src = MaskSources(
        surprisal_threshold=surprisal_threshold,
        surprisal_reduction=surprisal_reduction,
        ner_confidence_thresholds=ner_confidence_thresholds,
        ner_ignore_labels=set(ner_ignore_labels),
        lingua_context_threshold=lingua_context_threshold,
        lingua_span_threshold=lingua_span_threshold,
    )

    if surprisal_h5 is not None:
        click.echo(
            f"[*] Loading surprisal index: {surprisal_h5}  reduction={surprisal_reduction}",
            err=True,
        )
        src.surprisal_scores = _load_scores(surprisal_h5, reduction=surprisal_reduction)
        src.surprisal_records, src.surprisal_cpos = _load_index_records(surprisal_h5)

    if ner_h5 is not None:
        click.echo(f"[*] Loading NER index: {ner_h5}", err=True)
        src.ner_labels, src.ner_confidence, src.ner_records, src.ner_id2label, src.ner_cpos = (
            _load_ner_labels(ner_h5)
        )
        if ner_confidence_thresholds:
            if src.ner_confidence is None:
                raise click.UsageError(
                    "--ner-confidence-threshold requires logits-mode NER index "
                    "(ner_output=logits); got pre-computed labels."
                )
            o_id = next(k for k, v in src.ner_id2label.items() if v == "O")
            click.echo(
                f"[*] Applying per-label NER confidence filter: {ner_confidence_thresholds}",
                err=True,
            )
            src.ner_labels = _apply_ner_confidence_filter(
                src.ner_labels, src.ner_confidence, o_id,
                src.ner_id2label, ner_confidence_thresholds,
            )

    if surprisal_h5 is not None and ner_h5 is not None:
        _assert_index_alignment(src.surprisal_records, src.ner_records)
        click.echo("[*] Index alignment check passed", err=True)

    if loanword_file is not None:
        click.echo(f"[*] Loading loanword list: {loanword_file.name}", err=True)
        with open(loanword_file, encoding="utf-8") as f:
            src.lw_lemmas = {line.strip().lower() for line in f if line.strip()}

    return src


def build_masks(
    parsed: CQPResult,
    sent_idx: int,
    src: MaskSources,
    lw_lemma_set: Optional[set[str]] = None,
) -> tuple[list[int], list[int], list[int]]:
    n = len(parsed.tokens)

    combined_lw = (src.lw_lemmas or set()) | (lw_lemma_set or set())
    lw_mask = (
        build_loanword_mask(parsed, combined_lw)
        if combined_lw
        else [0] * n
    )

    if src.surprisal_scores is not None and src.surprisal_records is not None:
        rec = src.surprisal_records[sent_idx]
        sent_scores = src.surprisal_scores[rec.offset : rec.offset + rec.count]
        cs_mask = [1 if v > src.surprisal_threshold else 0 for v in sent_scores]
        cs_mask = cs_mask[:n]
    else:
        cs_mask = [0] * n

    if (
        src.ner_labels is not None
        and src.ner_records is not None
        and src.ner_id2label is not None
    ):
        rec = src.ner_records[sent_idx]
        sent_labels = src.ner_labels[rec.offset : rec.offset + rec.count]

        def _is_ne(label_id: int) -> bool:
            label = src.ner_id2label.get(label_id, "O")
            if label == "O":
                return False
            if label in src.ner_ignore_labels:
                return False
            if src.ner_exclude and label_id in src.ner_exclude:
                return False
            return True

        ne_mask = [int(_is_ne(int(l))) for l in sent_labels[:n]]
    else:
        ne_mask = build_named_entity_mask(parsed)

    return lw_mask, cs_mask, ne_mask


def is_clean_word_old(w: str) -> bool:
    """Determine if a token is a likely word rather than scientific/technical noise."""

    # 1. Must be longer than 2 characters
    if len(w) <= 2: 
        return False
    # 2. Must contain at least one alphabetic character
    if not any(c.isalpha() for c in w):
        return False
    # 3. Disregard strings with digits or math symbols (p-values, 10/94, etc)
    if re.search(r'[0-9=><±/]', w): 
        return False
    # 4. Disregard technical artifacts (like //zx)
    if "//" in w or "_" in w: 
        return False
    # 5. Disregard phonetic alphabet tokens (bracketed notation)
    if (w.startswith("[") and w.endswith("]")) or (
        w.startswith("/") and w.endswith("/")
    ):
        return False
    # 6. IPA Extensions and Phonetic Blocks (Unicode U+0250–U+02AF)
    # This catches "naked" phonetic symbols like ʃ, ʊ, ʌ, etc.
    if re.search(r"[\u0250-\u02AF]", w):
        return False
    # 7. Abbreviation: All caps (e.g., NASA, HTML)
    if sum(1 for c in w if c.isupper()) >= 2:
        return False
    # 8. Abbreviation: Internal or trailing periods (e.g., i.e., U.S., Dr.)
    if "." in w:
        return False

    return True

def is_clean_word(w: str, allow_ner: bool = True) -> bool:
    """
    Determine if a token is likely a word/entity rather than technical noise.
    Set allow_ner=True to prevent filtering out common entity patterns.
    """

    # Technical artifacts.
    if "//" in w or "_" in w: 
        return False
    # [..] and /.../ - technical.
    if (w.startswith("[") and w.endswith("]")) or (w.startswith("/") and w.endswith("/")):
        return False
    # Phonetic alphabet
    if re.search(r"[\u0250-\u02AF]", w):
        return False
    # At least one alphabetic character.
    if not any(c.isalpha() for c in w):
        return False
    # Single-chars is noise.
    if len(w) < 2:
        return False
    # Digits/Math.
    if re.search(r'[=><±]', w): 
        return False
    if re.search(r'[0-9/]', w): 
        return False

    if not allow_ner:
        # Strict Length: Discard "EU", "LR", "A."
        if len(w) < 3: 
            return False
        # Strict Case: Discard "NASA", "HTML"
        if sum(1 for c in w if c.isupper()) >= 2:
            return False
        # Strict Punctuation: Discard "U.S.", "Dr."
        if "." in w:
            return False

    return True


_LINE_RE  = re.compile(r"<LINE>(.*?)</LINE>", re.DOTALL)
_MNUM_RE  = re.compile(r"<MATCHNUM>(\d+)</MATCHNUM>")
_TOK_ITER = re.compile(r"<MATCH>|</MATCH>|<TOKEN>(.*?)</TOKEN>")


def _parse_sgml_line(ordinal: int, line_content: str) -> Optional[CQPResult]:
    m = _MNUM_RE.search(line_content)
    cqp_id = int(m.group(1)) if m else ordinal

    content_m = re.search(r"<CONTENT>(.*?)</CONTENT>", line_content, re.DOTALL)
    content = content_m.group(1) if content_m else line_content

    parsed_tokens: list[Token] = []
    match_index = -1
    in_match = False
    current_index = 0

    for tok_m in _TOK_ITER.finditer(content):
        tag = tok_m.group(0)
        if tag == "<MATCH>":
            in_match = True
            if match_index == -1:
                match_index = current_index
            continue
        if tag == "</MATCH>":
            in_match = False
            continue
        raw_tok = tok_m.group(1)
        if not raw_tok:
            continue
        parts = raw_tok.rsplit("/", 2)
        if len(parts) == 3:
            w, pos, lemma = parts
        elif len(parts) == 2:
            w, pos = parts
            lemma = w
        else:
            w, pos, lemma = raw_tok, "UNK", raw_tok
        if not w:
            continue
        parsed_tokens.append(Token(word=w, pos=pos, lemma=lemma))
        current_index += 1

    if not parsed_tokens:
        return None
    return CQPResult(cqp_id=cqp_id, tokens=parsed_tokens, match_index=match_index)


def parse_cwb_output(raw: str) -> Iterator[CQPResult]:
    matches = list(_LINE_RE.finditer(raw))
    for ordinal, m in enumerate(matches):
        parsed = _parse_sgml_line(ordinal, m.group(1))
        if parsed:
            yield parsed


def _build_spos_commands(indices: List[int]) -> List[str]:
    """Select sentences by s-attribute index (spos)."""
    commands = ["Results = <s> [];"]
    for idx in indices:
        commands.append(f"cat Results {idx} {idx};")
    return commands


def _build_cpos_commands(indices: List[int]) -> List[str]:
    """Select sentences by corpus position (cpos), expanded to sentence context."""
    # Build disjunction: [_.pos = 0 | _.pos = 42 | ...]
    disjunction = " | ".join(f"_.pos = {idx}" for idx in indices)
    commands = [
        "set Context s;",
        f"Results = [{disjunction}];",
        "cat Results;",
    ]
    return commands


def fetch_corpus_sentences(
    corpus: str,
    indices: List[int],
    mode: Literal["spos", "cpos"] = "cpos",
    cqp_bin: Optional[str] = None,
    registry_dir: Optional[str] = None,
) -> str:
    """
    Retrieves sentences from a corpus by index in a single CQP session.

    Args:
        corpus: Corpus name
        indices: List of indices to select
        mode: "spos" for sentence positions (s-attribute index),
              "cpos" for corpus positions (token-level)
        cqp_bin: Optional path to CQP binary
        registry_dir: Optional registry directory

    Returns:
        Concatenated CQP output for all requested sentences
    """
    if not indices:
        return ""
    if cqp_bin is None:
        cqp_bin = DEFAULT_CQP_BIN
    reg_dir, registry = _resolve_registry(registry_dir)

    if mode == "spos":
        commands = _build_spos_commands(indices)
    elif mode == "cpos":
        commands = _build_cpos_commands(indices)
    else:
        raise ValueError(f"Invalid mode '{mode}'. Expected 'spos' or 'cpos'.")

    return _run_cqp_command(corpus, commands, cqp_bin, reg_dir, registry)


def query_cqp_batch(
    corpus: str,
    queries: List[Tuple[str, Optional[int]]],
    cqp_bin: Optional[str] = None,
    registry_dir: Optional[str] = None,
    seed: int = 42,
) -> List[str]:
    """
    Executes multiple CQP queries in a single session.
    """
    if cqp_bin is None:
        cqp_bin = DEFAULT_CQP_BIN
    reg_dir, registry = _resolve_registry(registry_dir)
    
    outputs = []
    for query, limit in queries:
        commands = [f"Results = {query};"]
        if limit and limit > 0:
            commands.append(f"randomize {seed};")
            commands.append(f"reduce Results to {limit};")
        commands.append("cat Results;")
        output = _run_cqp_command(corpus, commands, cqp_bin, reg_dir, registry)
        outputs.append(output)
    
    return outputs


def build_or_query(lemmas: List[str]) -> str:
    """Build a CQP OR query from a list of lemmas."""
    escaped = [re.sub(r'([\[\](){}.*+?^$|\\])', r'\\\1', l) for l in lemmas]
    return f'[lemma="{"|".join(escaped)}"]'


def _lookup_sent_indices_batch(
    cpos_values: np.ndarray,
    cpos_array: np.ndarray,
    records: list[IndexRecord],
) -> np.ndarray:
    """
    Vectorized version of _lookup_sent_idx for a batch of cpos values.
    Returns an int32 array of sent_idx, with -1 for misses.
    """
    counts = np.array([r.count for r in records], dtype=np.int32)
    indices = np.searchsorted(cpos_array, cpos_values, side="right") - 1
    result  = np.full(len(cpos_values), -1, dtype=np.int32)

    valid = indices >= 0
    idx_v = indices[valid]
    cpos_v = cpos_values[valid]

    in_range = (cpos_array[idx_v] <= cpos_v) & (
        cpos_v < cpos_array[idx_v] + counts[idx_v]
    )
    result_valid = np.where(valid)[0]
    result[result_valid[in_range]] = idx_v[in_range]
    return result


def _detect_langs(
    detector,
    parsed: CQPResult,
    span_mask: list[int],
    context_threshold: float = 0.0,
    span_threshold: float = 0.0,
) -> tuple[str | None, str | None]:
    """
    Returns (context_lang, span_lang) using Lingua.
    context_lang: detected language of non-span tokens.
    span_lang:    detected language of span tokens.
    Either is None if detection fails or the text is empty.
    """
    if detector is None:
        return None, None

    span_set     = {i for i, v in enumerate(span_mask) if v}
    context_text = " ".join(t.word for i, t in enumerate(parsed.tokens) if i not in span_set)
    span_text    = " ".join(t.word for i, t in enumerate(parsed.tokens) if i in span_set)

    def _detect(text: str, threshold: float) -> str | None:
        if not text.strip():
            return None
        confidences = detector.compute_language_confidence_values(text)
        if not confidences:
            return None
        top = confidences[0]
        if top.value < threshold:
            return None
        return top.language.iso_code_639_1.name.lower()

    return _detect(context_text, context_threshold), _detect(span_text, span_threshold)


def query_by_lemmas(
    corpus_name: str,
    lemmas: List[str],
    lookup: int = DEFAULT_LOOKUP,
    cqp_bin: str = DEFAULT_CQP_BIN,
    registry_dir: str = None,
    scoring_config: str = None,
    deduplicate: bool = True,
    verbose: bool = False,
    mask_src: Optional[MaskSources] = None,
    lingua_lang: str | None = None,
) -> List[ScoredResult]:

    detector = _make_lingua_detector() if lingua_lang else None

    if lingua_lang:
        lang = _get_lingua_language(lingua_lang)
        if lang is None:
            raise click.UsageError(
                f"Unrecognized language code: {lingua_lang!r}. "
                f"Provide a valid ISO 639-1 code (e.g., 'lv', 'en', 'de')."
            )

    click.echo("[*] Querying CQP...", err=True)
    raw_output = query_cqp_batch(
        corpus_name,
        [(build_or_query(lemmas), lookup)],
        cqp_bin,
        registry_dir,
    )
    cfg = load_scoring_config(scoring_config, profile=QueryProfile.LEMMAS)
    if (cfg.filter_require_context_lang or cfg.filter_require_span_lang) and not lingua_lang:
        raise click.UsageError(
            "This scoring profile requires --lingua-lang "
            "(filter_require_context_lang or filter_require_span_lang is set)."
        )
    lemma_set = {l.lower() for l in lemmas}

    if mask_src is None:
        mask_src = MaskSources()

    if mask_src.surprisal_records and mask_src.ner_records:
        _assert_index_alignment(mask_src.surprisal_records, mask_src.ner_records)

    ref_records  = mask_src.surprisal_records or mask_src.ner_records
    ref_cpos_arr = (
        mask_src.surprisal_cpos if mask_src.surprisal_records else mask_src.ner_cpos
    )

    parsed_list = list(parse_cwb_output(raw_output[0]))

    # Resolve all sent_idx values in one vectorized call
    if ref_cpos_arr is not None and ref_records is not None:
        cpos_values = np.array([p.cqp_id for p in parsed_list], dtype=np.int64)
        sent_indices = _lookup_sent_indices_batch(cpos_values, ref_cpos_arr, ref_records)
    else:
        sent_indices = np.array([p.cqp_id for p in parsed_list], dtype=np.int32)

    seen_texts: dict[tuple, ScoredResult] = {}
    scored: List[ScoredResult] = []

    for parsed, sent_idx in tqdm(
        zip(parsed_list, sent_indices),
        total=len(parsed_list),
        disable=not verbose,
    ):
        sent_idx = int(sent_idx)
        if sent_idx == -1:
            continue

        lw_mask, cs_mask, ne_mask = build_masks(
            parsed, sent_idx, mask_src, lw_lemma_set=lemma_set
        )

        detected_lang, span_lang = _detect_langs(
            detector, parsed, lw_mask,
            context_threshold=mask_src.lingua_context_threshold,
            span_threshold=mask_src.lingua_span_threshold,
        )

        result = score_sentence(
            parsed,
            loanword_mask=lw_mask,
            code_switch_mask=cs_mask,
            named_entity_mask=ne_mask,
            detected_lang=detected_lang,
            span_lang=span_lang,
            lingua_lang=lingua_lang,
            cfg=cfg,
        )

        if deduplicate:
            key = tuple((t.word, t.pos, t.lemma) for t in result.tokens)
            existing = seen_texts.get(key)
            if existing is None or result.score_total > existing.score_total:
                seen_texts[key] = result
        else:
            scored.append(result)

    if deduplicate:
        scored = list(seen_texts.values())

    scored.sort(key=lambda r: r.score_total, reverse=True)
    return scored


def scan_anomaly_candidates(
    scores: np.ndarray,
    index_records: list[IndexRecord],
    threshold: float,
    min_consecutive: int,
    max_consecutive: int | None = None,
    lookup: int | None = None,
    allowed=None,
    desc: str = "Scanning",
) -> list[tuple[int, list[int], list[float]]]:
    """
    Vectorized scan. Returns (sent_idx, token_indices, token_scores)
    for the best run per sentence, sorted by run length descending.
    """
    records = index_records[:lookup] if lookup else index_records
    if not records:
        return []

    CHUNK = 50_000
    candidates: list[tuple[int, list[int], list[float]]] = []
    n = len(records)
    with tqdm(total=n, desc=desc, unit="sent") as pbar:
        for chunk_start in range(0, n, CHUNK):
            chunk_end = min(chunk_start + CHUNK, n)
            chunk_records = records[chunk_start:chunk_end]
            offsets = np.array([r.offset for r in chunk_records], dtype=np.int64)
            counts  = np.array([r.count  for r in chunk_records], dtype=np.int32)
            chunk_flat_start = int(offsets[0])
            end_idx = int(offsets[-1]) + int(counts[-1])
            flat = scores[chunk_flat_start:end_idx]
            offsets = offsets - chunk_flat_start
            chunk_allowed = (
                None
                if allowed is None
                else {i - chunk_start for i in allowed if chunk_start <= i < chunk_end}
            )
            chunk_hits = _scan_runs_vectorized(
                flat, offsets, counts, threshold, min_consecutive, max_consecutive,
                allowed=chunk_allowed,
            )
            candidates.extend(
                (sid + chunk_start, indices, token_scores)
                for sid, indices, token_scores in chunk_hits
            )
            pbar.update(chunk_end - chunk_start)

    # Sort by run length descending (best first)
    candidates.sort(key=lambda x: len(x[1]), reverse=True)
    return candidates


def _ner_matching_sentences(
    ner_labels: np.ndarray,
    records: list[IndexRecord],
    want: set[int],
    exclude: set[int] | None = None,
    allowed: set[int] | None = None,
    desc: str = "Scanning NER",
) -> list[int]:
    """
    Vectorized: return sorted sentence indices that contain any label in `want`
    and NO token whose label is in `exclude` (if provided).
    """
    if not records:
        return []

    CHUNK = 50_000
    want_arr = np.array(sorted(want), dtype=ner_labels.dtype)
    excl_arr = (
        np.array(sorted(exclude), dtype=ner_labels.dtype) if exclude else None
    )
    hit_sents:  set[int] = set()
    excl_sents: set[int] = set()
    n = len(records)
    with tqdm(total=n, desc=desc, unit="sent") as pbar:
        for chunk_start in range(0, n, CHUNK):
            chunk_end = min(chunk_start + CHUNK, n)
            chunk_records = records[chunk_start:chunk_end]
            offsets  = np.array([r.offset for r in chunk_records], dtype=np.int64)
            counts   = np.array([r.count  for r in chunk_records], dtype=np.int32)
            chunk_flat_start = int(offsets[0])
            end_idx  = int(offsets[-1]) + int(counts[-1])
            flat     = ner_labels[chunk_flat_start:end_idx]
            sent_ids = np.repeat(
                np.arange(chunk_start, chunk_end, dtype=np.int32), counts
            )
            token_hits = np.isin(flat, want_arr)
            if token_hits.any():
                hit_sents |= set(np.unique(sent_ids[token_hits]).tolist())
            if excl_arr is not None:
                excl_hits = np.isin(flat, excl_arr)
                if excl_hits.any():
                    excl_sents |= set(np.unique(sent_ids[excl_hits]).tolist())
            pbar.update(chunk_end - chunk_start)

    hit_sents -= excl_sents
    if allowed is not None:
        hit_sents &= allowed
    return sorted(hit_sents)


def find_code_switch_sequences(
    scores: np.ndarray,
    index_records: list[IndexRecord],
    threshold: float,
    min_consecutive: int,
    max_consecutive: int,
    corpus: str,
    cfg: ScoringConfig,
    cqp_bin: str = DEFAULT_CQP_BIN,
    registry_dir: Optional[str] = None,
    lookup: int | None = None,
    mask_src: Optional[MaskSources] = None,
    allowed = None,
    lingua_lang: str | None = None,
    ne_exclude_span_only: bool = False,
) -> list[CodeSwitchRun]:
    """Score all qualifying runs; caller is responsible for capping display."""

    detector = _make_lingua_detector() if lingua_lang else None

    if lingua_lang:
        lang = _get_lingua_language(lingua_lang)
        if lang is None:
           raise click.UsageError(
               f"Unrecognized language code: {lingua_lang!r}. "
               f"Provide a valid ISO 639-1 code (e.g., 'lv', 'en', 'de')."
           )

    if mask_src is None:
        mask_src = MaskSources()

    if mask_src.ner_records is not None:
        _assert_index_alignment(index_records, mask_src.ner_records)


    candidates = scan_anomaly_candidates(
        scores=scores,
        index_records=index_records,
        threshold=threshold,
        min_consecutive=min_consecutive,
        lookup=lookup,
        allowed=allowed,
    )
    if not candidates:
        return []

    unique_sent_indices = sorted({sent_idx for sent_idx, _, _ in candidates})
    raw_output = fetch_corpus_sentences(
        corpus=corpus,
        indices=unique_sent_indices,
        mode="spos",
        cqp_bin=cqp_bin,
        registry_dir=registry_dir,
    )

    sentence_map: dict[int, CQPResult] = {}
    for parsed, sent_idx in zip(parse_cwb_output(raw_output), unique_sent_indices):
        sentence_map[sent_idx] = parsed

    results: list[CodeSwitchRun] = []
    for sent_idx, token_indices, token_scores in tqdm(
        candidates, desc="Scoring candidates", unit="sent"
    ):
        parsed = sentence_map.get(sent_idx)
        if not parsed:
            continue

        n = len(parsed.tokens)
        valid_indices = [i for i in token_indices if i < n]
        valid_scores = [s for i, s in zip(token_indices, token_scores) if i < n]

        if len(valid_indices) < min_consecutive:
            continue

        cs_mask = [1 if i in set(valid_indices) else 0 for i in range(n)]
        lw_mask, _, ne_mask = build_masks(parsed, sent_idx, mask_src)

        if ne_exclude_span_only:
            span_set = set(valid_indices)
            ne_mask = [v if i in span_set else 0 for i, v in enumerate(ne_mask)]

        detected_lang, span_lang = _detect_langs(
            detector, parsed, cs_mask,
            context_threshold=mask_src.lingua_context_threshold,
            span_threshold=mask_src.lingua_span_threshold,
        )

        metrics = score_sentence(
            parsed,
            loanword_mask=lw_mask,
            code_switch_mask=cs_mask,
            named_entity_mask=ne_mask,
            detected_lang=detected_lang,
            cfg=cfg,
            span_lang=span_lang,
            lingua_lang=lingua_lang,
        )
        results.append(
            CodeSwitchRun(
                sent_idx=sent_idx,
                token_indices=valid_indices,
                token_scores=valid_scores,
                tokens=parsed.tokens,
                metrics=metrics,
            )
        )

    results.sort(key=lambda r: r.metrics.score_total, reverse=True)
    return results


def mask_source_options(f):
    """Decorator that adds --surprisal-h5, --surprisal-threshold,
    --ner-h5, and --loanwords to a command."""
    f = click.option(
        "--ner-confidence-threshold",
        "ner_confidence_thresholds",
        multiple=True,
        default=[],
        help=(
            "Label-specific confidence threshold: LABEL:VALUE "
            "(e.g. MISC:0.8). Repeatable. Requires logits-mode NER index."
        ),
    )(f)
    f = click.option(
        "--loanwords",
        "loanword_file",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="One lemma per line; builds loanword mask.",
    )(f)
    f = click.option(
        "--ner-h5",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="NER HDF5 index; builds NE mask. Falls back to heuristic if omitted.",
    )(f)
    f = click.option(
        "--surprisal-threshold",
        type=float,
        default=0.0,
        show_default=True,
        help="Per-token surprisal threshold for code-switch mask.",
    )(f)
    f = click.option(
        "--surprisal-h5",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="Surprisal HDF5 index; builds code-switch mask. Omit for all-zero.",
    )(f)
    f = click.option(
        "--ignore-label",
        "ner_ignore_labels",
        multiple=True,
        default=[],
        help="Treat these NER labels as O. Repeatable: --ignore-label MISC --ignore-label ORG",
    )(f)
    f = click.option(
        "--lingua-lang",
        "lingua_lang",
        type=str,
        default=None,
        help="Required context language code (e.g. 'lv'). Enables Lingua detection.",
    )(f)
    f = click.option(
        "--surprisal-reduction",
        type=click.Choice(["mean", "dm_mad"]),
        default="mean",
        show_default=True,
        help="Which surprisal score column to use as the code-switch signal.",
    )(f)
    f = click.option(
        "--lingua-context-threshold",
        "lingua_context_threshold",
        type=float,
        default=0.0,
        help="Lingua confidence threshold for context (non-span) language (0.0-1.0).",
    )(f)
    f = click.option(
        "--lingua-span-threshold",
        "lingua_span_threshold",
        type=float,
        default=0.0,
        help="Lingua confidence threshold for span language (0.0-1.0).",
    )(f)
    return f

# ------ CLI -------

def scoring_config_option(f):
    return click.option(
        "--scoring-config",
        type=click.Path(exists=True, dir_okay=False),
        default=None,
        help="TOML file overriding default scoring parameters.",
    )(f)


@click.group("query")
def query_group():
    """Query corpus."""


@contextmanager
def _output_context(output: Path | None):
    if output is None:
        yield sys.stdout
    else:
        with open(output, "w", encoding="utf-8") as fh:
            yield fh


def _load_candidates(path: str) -> list[CandidateRecord]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading candidates", unit="rec"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records.append(CandidateRecord(**d))
    return records


def _sentence_index_set(
    n_total: int,
    sequential: bool,
    seed: int,
    lookup: int | None,
) -> set[int] | None:
    """
    Returns a set of allowed sentence indices, or None meaning 'all'.
    Sequential: first `lookup` sentences. Random: `lookup` random indices.
    """
    if lookup is None:
        return None
    if sequential:
        return set(range(min(lookup, n_total)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=min(lookup, n_total), replace=False)
    return set(idx.tolist())


@query_group.command("pretty-print")
@click.argument(
    "candidates",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--results",
    type=int,
    default=0,
    show_default=True,
    help="Number of results to show. 0 = all.",
)
@click.option(
    "--mode",
    type=click.Choice(["lemmas", "ner", "code_switch", "all"]),
    default="all",
    show_default=True,
    help="Filter by mode.",
)
def pretty_print_command(candidates, results, mode):
    """Pretty-print a JSONL candidates file."""
    pool = _load_candidates(candidates)
    if mode != "all":
        pool = [r for r in pool if r.mode == mode]
    shown = pool[:results] if results > 0 else pool
    _render_candidates(shown)
    click.echo(f"({len(shown)} of {len(pool)} records shown)", err=True)


def _render_candidates(records: list[CandidateRecord]) -> None:
    for rec in records:
        status = (
            f" [FILTERED: {rec.filter_reason}]" if rec.filtered else ""
        )
        click.echo(
            f"Score: {rec.score_total:.4f}  "
            f"(len={rec.score_length:.2f}  lw={rec.score_lw:.2f}  "
            f"cs={rec.score_cs:.2f}  alpha={rec.score_alpha:.2f}  "
            f"ne={rec.score_ne:.2f})  "
            f"| mode={rec.mode}  cqp_id={rec.cqp_id}"
            f"  spos={rec.spos}{status}"
        )
        click.echo(rec.sentence)
        if rec.tag_map:
            click.echo(f"  ↳ tags: {rec.tag_map}")
        if rec.token_surprisals:
            surp_parts = "  ".join(
                f"{tag}={word!r} surprisal={rec.token_surprisals.get(tag, 0.0):.3f}"
                for tag, word in rec.tag_map.items()
                if tag.startswith("CS")
            )
            click.echo(f"  ↳ surprisal: [{surp_parts}]")
        click.echo("-" * 60)


@query_group.command("code-switch")
@click.argument("corpus_name")
@click.option("--threshold", type=float, required=True)
@click.option("--min-consecutive", type=int, default=2, show_default=True)
@click.option("--max-consecutive", type=int, default=None, help="Max length of code-switch span.")
@mask_source_options
@click.option(
    "--lookup",
    type=int,
    default=None,
    help="Scan only the first N sentences. Omit to scan all.",
)
@click.option(
    "--results",
    type=int,
    default=DEFAULT_RESULTS,
    show_default=True,
    help="Number of top results to emit. 0 = all.",
)
@click.option("--cqp-bin", default=DEFAULT_CQP_BIN, show_default=True)
@click.option("--registry-dir", default=None)
@click.option("--seed", type=int, default=42, show_default=True)
@scoring_config_option
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSONL to file. Omit to write to stdout.",
)
@click.option(
    "--sequential",
    is_flag=True,
    default=False,
    help="Scan in index order. Default: random by seed."
)
@click.option(
    "--ne-exclude-span-only",
    "ne_exclude_span_only",
    is_flag=True,
    default=False,
    help=(
        "Apply NE-based filtering/penalty only to tokens within the "
        "code-switch span, not the whole sentence."
    ),
)
def query_code_switch(
    corpus_name,
    surprisal_h5,
    surprisal_threshold,
    threshold,
    min_consecutive,
    max_consecutive,
    ner_h5,
    loanword_file,
    ner_ignore_labels,
    ner_confidence_thresholds,
    lingua_lang,
    lingua_context_threshold,
    lingua_span_threshold,
    surprisal_reduction,
    lookup,
    results,
    cqp_bin,
    registry_dir,
    seed,
    scoring_config,
    output,
    sequential,
    ne_exclude_span_only,
):
    """Find code-switch sequences and emit JSONL records."""
    cfg = load_scoring_config(scoring_config, profile=QueryProfile.CODE_SWITCH)
    if (cfg.filter_require_context_lang or cfg.filter_require_span_lang) and not lingua_lang:
        raise click.UsageError(
            "This scoring profile requires --lingua-lang "
            "(filter_require_context_lang or filter_require_span_lang is set)."
        )

    mask_src = load_mask_sources(
        surprisal_h5=surprisal_h5,
        surprisal_threshold=surprisal_threshold,
        surprisal_reduction=surprisal_reduction,
        ner_h5=ner_h5,
        ner_confidence_thresholds=_parse_label_thresholds(ner_confidence_thresholds),
        loanword_file=loanword_file,
        ner_ignore_labels=set(ner_ignore_labels),
        lingua_context_threshold=lingua_context_threshold,
        lingua_span_threshold=lingua_span_threshold,
    )

    if mask_src.surprisal_scores is None:
        raise click.UsageError("--surprisal-h5 is required for code-switch mode.")

    scores = mask_src.surprisal_scores
    index_records = mask_src.surprisal_records
    allowed = _sentence_index_set(len(index_records), sequential, seed, lookup)

    click.echo(
        f"[*] Scanning in {'sequential' if sequential else f'random (seed={seed})'} order"
        + (f", {lookup} sentences" if lookup else ""),
        err=True,
    )
    found = find_code_switch_sequences(
        scores=scores,
        index_records=index_records,
        threshold=threshold,
        min_consecutive=min_consecutive,
        max_consecutive=max_consecutive,
        corpus=corpus_name,
        cfg=cfg,
        cqp_bin=cqp_bin,
        registry_dir=registry_dir,
        lookup=None,
        mask_src=mask_src,
        allowed=allowed,
        lingua_lang=lingua_lang,
        ne_exclude_span_only=ne_exclude_span_only,
    )

    click.echo(f"[*] Found and scored {len(found)} candidate sequences", err=True)
    shown = found if results == 0 else found[:results]

    with _output_context(output) as fh:
        for run in shown:
            sentence = _tag_code_switch_sentence(run)
            matched = [
                run.tokens[i].lemma
                for i in run.token_indices
                if i < len(run.tokens)
            ]
            tag_map = {
                f"CS{i+1}": run.tokens[idx].lemma
                for i, idx in enumerate(run.token_indices)
                if idx < len(run.tokens)
            }
            token_surprisals = {
                f"CS{i+1}": s
                for i, (idx, s) in enumerate(
                    zip(run.token_indices, run.token_scores)
                )
                if idx < len(run.tokens)
            }
            rec = _result_to_record(
                mode="code_switch",
                sentence=sentence,
                scored=run.metrics,
                matched_lemmas=matched,
                tag_map=tag_map,
                token_surprisals=token_surprisals,
                cpos=run.metrics.cqp_id,
                spos=run.sent_idx,
                seed=seed,
            )
            _write_record(rec, fh)

    if output:
        click.echo(f"[✓] Wrote {len(shown)} records to {output}", err=True)


@query_group.command("lemmas")
@click.argument("corpus_name")
@click.argument("lemmas", required=False, default=None)
@click.option(
    "--lemmas-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Newline-separated file of lemmas. Alternative to LEMMAS argument.",
)
@click.option(
    "--lookup",
    type=int,
    default=DEFAULT_LOOKUP,
    show_default=True,
    help="Number of corpus rows to fetch and process.",
)
@click.option(
    "--results",
    type=int,
    default=DEFAULT_RESULTS,
    show_default=True,
    help="Number of top results to emit.",
)
@click.option("--cqp-bin", default=DEFAULT_CQP_BIN, show_default=True)
@click.option("--registry-dir", default=None)
@click.option("--seed", type=int, default=42, show_default=True)
@scoring_config_option
@mask_source_options
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSONL to file. Omit to write to stdout.",
)
def query_lemmas_command(
    corpus_name,
    lemmas,
    lemmas_file,
    lookup,
    results,
    cqp_bin,
    registry_dir,
    seed,
    scoring_config,
    surprisal_h5,
    surprisal_threshold,
    ner_h5,
    ner_confidence_thresholds,
    loanword_file,
    ner_ignore_labels,
    surprisal_reduction,
    lingua_lang,
    lingua_context_threshold,
    lingua_span_threshold,
    output,
):
    """Query corpus by lemma(s) and emit JSONL records."""
    mask_src = load_mask_sources(
        surprisal_h5=surprisal_h5,
        surprisal_threshold=surprisal_threshold,
        surprisal_reduction=surprisal_reduction,
        ner_h5=ner_h5,
        ner_confidence_thresholds=_parse_label_thresholds(ner_confidence_thresholds),
        loanword_file=loanword_file,
        ner_ignore_labels=ner_ignore_labels,
        lingua_context_threshold=lingua_context_threshold,
        lingua_span_threshold=lingua_span_threshold,
    )

    if lemmas_file is not None:
        with open(lemmas_file, encoding="utf-8") as f:
            lemma_list = [l.strip() for l in f if l.strip()]
    elif lemmas is not None:
        lemma_list = lemmas.split(",")
    else:
        raise click.UsageError("Provide either LEMMAS argument or --lemmas-file.")
    lemma_set_lower = {l.lower(): l for l in lemma_list}

    found = query_by_lemmas(
        corpus_name=corpus_name,
        lemmas=lemma_list,
        lookup=lookup,
        cqp_bin=cqp_bin,
        registry_dir=registry_dir,
        scoring_config=scoring_config,
        verbose=True,
        mask_src=mask_src,
        lingua_lang=lingua_lang,
    )

    shown = found[:results] if results > 0 else found

    with _output_context(output) as fh:
        for r in shown:
            matched = [
                t.lemma for t in r.tokens
                if t.lemma.lower() in lemma_set_lower
            ]
            sentence = tag_all_loanwords(r, lemma_set_lower, lemma_list[0])
            tag_map = {
                f"L{i+1}": lemma_set_lower.get(m.lower(), m)
                for i, m in enumerate(matched)
            }
            rec = _result_to_record(
                mode="lemmas",
                sentence=sentence or " ".join(t.word for t in r.tokens),
                scored=r,
                matched_lemmas=matched,
                tag_map=tag_map,
                token_surprisals={},
                cpos=r.cqp_id,
                spos=None,
                seed=seed,
            )
            _write_record(rec, fh)

    if output:
        click.echo(f"[✓] Wrote {len(shown)} records to {output}", err=True)


@query_group.command("position")
@click.argument("corpus_name")
@click.argument("range_str")
@click.option("--cqp-bin", default=DEFAULT_CQP_BIN)
@click.option("--registry-dir", default=None)
def sentence_slice(corpus_name, range_str, cqp_bin, registry_dir):
    """Get the nth sentence or a range (start:stop) from the corpus."""
    if ":" in range_str:
        try:
            start, stop = map(int, range_str.split(":"))
        except ValueError:
            raise click.UsageError("Range must be 'start:stop' (e.g. 0:10)")
    else:
        start = stop = int(range_str)

    raw_output = fetch_corpus_sentences(corpus_name, indices=[i for i in range(start,stop+1)], mode="spos", cqp_bin=cqp_bin, registry_dir=registry_dir)
    
    for parsed in parse_cwb_output(raw_output):
        # Simple text reconstruction for display
        text = " ".join([t.word for t in parsed.tokens])
        click.echo(f"[{parsed.cqp_id}] {text}")


@query_group.command("ner-entities")
@click.argument("corpus_name")
@click.argument(
    "ner_h5",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--label", "labels", multiple=True, help="Include only these NER labels.")
@click.option("--exclude-label", "excl_labels", multiple=True, help="Drop sentences containing these NER labels.")
@click.option(
    "--lookup",
    type=int,
    default=None,
    help="Scan only the first N sentences. Omit to scan all.",
)
@click.option(
    "--results",
    type=int,
    default=DEFAULT_RESULTS,
    show_default=True,
    help="Number of top results to emit. 0 = all.",
)
@click.option("--cqp-bin", default=DEFAULT_CQP_BIN, show_default=True)
@click.option("--registry-dir", default=None)
@click.option("--seed", type=int, default=42, show_default=True)
@scoring_config_option
@mask_source_options
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write JSONL to file. Omit to write to stdout.",
)
@click.option(
    "--sequential",
    is_flag=True,
    default=False,
    help="Scan in index order. Default: random by seed."
)
def query_ner_entities(
    corpus_name, ner_h5, labels, excl_labels,
    lookup, results, cqp_bin, registry_dir, seed, scoring_config,
    surprisal_h5, surprisal_threshold, ner_ignore_labels, loanword_file,
    ner_confidence_thresholds, lingua_lang, lingua_context_threshold,
    lingua_span_threshold, surprisal_reduction, output, sequential,
):
    """Find named entity sentences and emit JSONL records."""
    cfg = load_scoring_config(scoring_config, profile=QueryProfile.NER)
    if (cfg.filter_require_context_lang or cfg.filter_require_span_lang) and not lingua_lang:
        raise click.UsageError(
            "This scoring profile requires --lingua-lang "
            "(filter_require_context_lang or filter_require_span_lang is set)."
        )
    detector = _make_lingua_detector() if lingua_lang else None

    if lingua_lang:
        lang = _get_lingua_language(lingua_lang)
        if lang is None:
            raise click.UsageError(
                f"Unrecognized language code: {lingua_lang!r}. "
                f"Provide a valid ISO 639-1 code (e.g., 'lv', 'en', 'de')."
            )

    mask_src = load_mask_sources(
        surprisal_h5=surprisal_h5,
        surprisal_threshold=surprisal_threshold,
        surprisal_reduction=surprisal_reduction,
        loanword_file=loanword_file,
        ner_confidence_thresholds=_parse_label_thresholds(ner_confidence_thresholds),
        ner_ignore_labels=ner_ignore_labels,
        lingua_context_threshold=lingua_context_threshold,
        lingua_span_threshold=lingua_span_threshold,
    )

    click.echo("[*] Loading NER index...", err=True)
    ner_labels, ner_confidence, ner_records, id2label, ner_cpos = _load_ner_labels(ner_h5)
    mask_src.ner_labels   = ner_labels
    mask_src.ner_confidence = ner_confidence
    mask_src.ner_records  = ner_records
    mask_src.ner_id2label = id2label
    mask_src.ner_cpos     = ner_cpos

    thresholds = _parse_label_thresholds(ner_confidence_thresholds)
    if thresholds:
        if ner_confidence is None:
            raise click.UsageError(
                "--ner-confidence-threshold requires logits-mode NER index "
                "(ner_output=logits); got pre-computed labels."
            )
        o_id = next(k for k, v in id2label.items() if v == "O")
        click.echo(
            f"[*] Applying per-label NER confidence filter: {thresholds}",
            err=True,
        )
        mask_src.ner_labels = _apply_ner_confidence_filter(
            ner_labels, ner_confidence, o_id, id2label, thresholds
        )

    if surprisal_h5 is not None and mask_src.surprisal_records is not None:
        _assert_index_alignment(mask_src.surprisal_records, ner_records)

    o_ids = {k for k, v in id2label.items() if v == "O"}

    if labels:
        want = {k for k, v in id2label.items() if v in set(labels)}
        if not want:
            available = sorted(set(id2label.values()) - {"O"})
            raise click.UsageError(
                f"None of {list(labels)!r} found in index. "
                f"Available labels: {available}"
            )
    else:
        want = set(id2label.keys()) - o_ids

    exclude: set[int] | None = None
    if excl_labels:
        exclude = {k for k, v in id2label.items() if v in set(excl_labels)}
        bad = set(excl_labels) - set(id2label.values())
        if bad:
            raise click.UsageError(
                f"Unknown --exclude-label value(s): {sorted(bad)}. "
                f"Available: {sorted(set(id2label.values()))}"
            )
        overlap = want & exclude
        if overlap:
            raise click.UsageError(
                f"Labels appear in both --label and --exclude-label: "
                f"{sorted(id2label[k] for k in overlap)}"
            )

    mask_src.ner_exclude = exclude

    click.echo(
        f"[*] Scanning for: {sorted(id2label[k] for k in want)}", err=True
    )
    if exclude:
        click.echo(
            f"[*] Excluding sentences with: "
            f"{sorted(id2label[k] for k in exclude)}",
            err=True,
        )

    allowed = _sentence_index_set(len(ner_records), sequential, seed, lookup)
    click.echo(
        f"[*] Scanning in {'sequential' if sequential else f'random (seed={seed})'} order"
        + (f", {lookup} sentences" if lookup else ""),
        err=True,
    )

    click.echo(
        f"[*] Scanning {len(ner_records)} sentences for matching labels...",
        err=True,
    )
    matching = _ner_matching_sentences(mask_src.ner_labels, ner_records, want, exclude, allowed=allowed)
    click.echo(f"[*] Found {len(matching)} candidate sentence(s)", err=True)

    if not matching:
        return

    click.echo("[*] Fetching sentences from corpus...", err=True)
    raw_output = fetch_corpus_sentences(
        corpus=corpus_name,
        indices=matching,
        mode="spos",
        cqp_bin=cqp_bin,
        registry_dir=registry_dir,
    )

    sentence_map: dict[int, CQPResult] = {}
    for parsed, sent_idx in zip(parse_cwb_output(raw_output), matching):
        sentence_map[sent_idx] = parsed

    scored_results: list[tuple[ScoredResult, str, int]] = []
    for sent_idx in tqdm(matching, desc="Scoring", unit="sent"):
        parsed = sentence_map[sent_idx]
        lw_mask, cs_mask, ne_mask = build_masks(parsed, sent_idx, mask_src)
        detected_lang, span_lang = _detect_langs(
            detector, parsed, ne_mask,
            context_threshold=lingua_context_threshold,
            span_threshold=lingua_span_threshold,
        )

        scored = score_sentence(
            parsed,
            loanword_mask=lw_mask,
            code_switch_mask=cs_mask,
            named_entity_mask=ne_mask,
            detected_lang=detected_lang,
            lingua_lang=lingua_lang,
            span_lang=span_lang,
            cfg=cfg,
        )
        sentence = _tag_ner_sentence(
            parsed, sent_idx, mask_src.ner_labels, ner_records, id2label, want
        )
        scored_results.append((scored, sentence, sent_idx))

    scored_results.sort(key=lambda x: x[0].score_total, reverse=True)
    click.echo(f"[*] Scored {len(scored_results)} sentence(s)", err=True)

    shown = scored_results if results == 0 else scored_results[:results]

    with _output_context(output) as fh:
        for scored, sentence, sent_idx in shown:
            record = ner_records[sent_idx]
            chunk = mask_src.ner_labels[record.offset : record.offset + record.count]
            matched = [
                id2label[int(chunk[i])]
                for i, t in enumerate(sentence_map[sent_idx].tokens)
                if i < len(chunk) and int(chunk[i]) in want
            ]
            tag_map = {f"NE{i+1}": v for i, v in enumerate(matched)}
            rec = _result_to_record(
                mode="ner",
                sentence=sentence,
                scored=scored,
                matched_lemmas=matched,
                tag_map=tag_map,
                token_surprisals={},
                cpos=scored.cqp_id,
                spos=sent_idx,
                seed=seed,
            )
            _write_record(rec, fh)

    if output:
        click.echo(f"[✓] Wrote {len(shown)} records to {output}", err=True)


if __name__ == "__main__":
    corpus()
