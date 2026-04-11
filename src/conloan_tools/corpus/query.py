import subprocess
import os
import tempfile
import click
import json
from typing import List, Optional, Tuple, Literal, Iterator
import re
from pathlib import Path
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

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

@dataclass
class IndexRecord:
    offset: int
    count: int


@dataclass
class CodeSwitchRun:
    sent_idx: int
    token_indices: list[int]
    token_scores: list[float]
    tokens: list[Token]
    metrics: ScoredResult 

# ------ Internal -------

@dataclass
class MaskSources:
    surprisal_scores: Optional[np.ndarray] = None
    surprisal_records: Optional[list[IndexRecord]] = None
    surprisal_cpos: Optional[np.ndarray] = None
    surprisal_threshold: float = 0.0
    ner_labels: Optional[np.ndarray] = None
    ner_records: Optional[list[IndexRecord]] = None
    ner_cpos: Optional[np.ndarray] = None
    ner_id2label: Optional[dict[int, str]] = None
    ner_exclude: Optional[set[int]] = None
    lw_lemmas: Optional[set[str]] = None
    ner_ignore_misc: bool = False


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
    return output


def _load_scores(h5_path: Path) -> np.ndarray:
    """Load flat token scores from HDF5 /scores/data, cast to float32."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        return f["scores"]["data"][:].astype(np.float32)


def _load_index_records(h5_path: Path) -> tuple[list[IndexRecord], np.ndarray]:
    """Returns (records, cpos_array). Records use flat offsets for score slicing;
    cpos_array retains raw corpus positions for sentence lookup."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        cpos  = f["index"]["cpos"][:]
        count = f["index"]["count"][:]
    offsets = np.concatenate([[0], np.cumsum(count[:-1])])
    records = [
        IndexRecord(offset=int(o), count=int(n))
        for o, n in zip(offsets, count)
    ]
    return records, cpos


def _load_ner_labels(h5_path: Path) -> tuple[np.ndarray, list[IndexRecord], dict[int, str]]:
    """
    Load NER label index from HDF5.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        ner_output = f.attrs.get("ner_output", "labels")
        raw = f["scores"]["data"][:]
        if ner_output == "logits":
            if raw.ndim != 2:
                raise click.UsageError(
                    f"{h5_path.name}: expected 2-D logits array, "
                    f"got shape {raw.shape}."
                )
            labels = np.argmax(raw, axis=-1).astype(np.uint8)
        elif ner_output == "labels":
            labels = raw.astype(np.uint8)
        else:
            raise click.UsageError(
                f"{h5_path.name}: unknown ner_output='{ner_output}'."
            )
        cpos = f["index"]["cpos"][:]
        count  = f["index"]["count"][:]
        raw_id2label = json.loads(f.attrs["id2label"])

    id2label = {int(k): v for k, v in raw_id2label.items()}
    offsets = np.concatenate([[0], np.cumsum(count[:-1])])
    records  = [
        IndexRecord(offset=int(o), count=int(n))
        for o, n in zip(offsets, count)
    ]
    return labels, records, id2label, cpos


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
            f"  alpha={scored.score_alpha:.2f}  "
            f"  ne={res.score_named_entity:.2f})"
            f"  | Pos: {run.sent_idx}  ID: {res.cqp_id}  {status}"
        )

        parts = [
            f"[{t.word}]" if i in index_set else t.word
            for i, t in enumerate(run.tokens)
        ]
        click.echo(" ".join(parts))

        run_details = "  ".join(
            f"{run.tokens[i].word}({s:.2f})"
            for i, s in zip(run.token_indices, run.token_scores)
        )
        click.echo(f"  ↳ run: {run_details}")
        click.echo("-" * 60)

    click.echo(f"({len(results)} sequences shown)")

# ------ Public -------

def load_mask_sources(
    surprisal_h5: Optional[Path] = None,
    surprisal_threshold: float = 0.0,
    ner_h5: Optional[Path] = None,
    loanword_file: Optional[Path] = None,
    ner_ignore_misc: bool = False,
) -> MaskSources:
    """Load all optional index files into a MaskSources bundle."""
    src = MaskSources(
        surprisal_threshold=surprisal_threshold,
        ner_ignore_misc=ner_ignore_misc,
    )

    if surprisal_h5 is not None:
        click.echo(f"[*] Loading surprisal index: {surprisal_h5.name}", err=True)
        src.surprisal_scores = _load_scores(surprisal_h5)
        src.surprisal_records, src.surprisal_cpos = _load_index_records(surprisal_h5)

    if ner_h5 is not None:
        click.echo(f"[*] Loading NER index: {ner_h5.name}", err=True)
        src.ner_labels, src.ner_records, src.ner_id2label, src.ner_cpos = _load_ner_labels(ner_h5)

    if surprisal_h5 is not None and ner_h5 is not None:
        _assert_index_alignment(src.surprisal_records, src.ner_records)

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
            if src.ner_ignore_misc and label == "MISC":
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
    """Yield one CQPResult per <LINE> in SGML output."""
    for ordinal, m in enumerate(_LINE_RE.finditer(raw)):
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
) -> List[ScoredResult]:
    raw_output = query_cqp_batch(
        corpus_name,
        [(build_or_query(lemmas), lookup)],
        cqp_bin,
        registry_dir,
    )
    cfg = load_scoring_config(scoring_config, profile=QueryProfile.LEMMAS)
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
        result = score_sentence(
            parsed,
            loanword_mask=lw_mask,
            code_switch_mask=cs_mask,
            named_entity_mask=ne_mask,
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
    lookup: int | None = None,
) -> list[tuple[int, list[int], list[float]]]:
    """
    Vectorized scan. Returns (sent_idx, token_indices, token_scores)
    for the best run per sentence, sorted by run length descending.
    """
    records = index_records[:lookup] if lookup else index_records
    if not records:
        return []

    offsets = np.array([r.offset for r in records], dtype=np.int64)
    counts  = np.array([r.count  for r in records], dtype=np.int32)
    end_idx = int(offsets[-1]) + int(counts[-1])

    flat = scores[:end_idx]

    candidates = _scan_runs_vectorized(
        flat, offsets, counts, threshold, min_consecutive
    )

    # Sort by run length descending (best first)
    candidates.sort(key=lambda x: len(x[1]), reverse=True)
    return candidates


def _ner_matching_sentences(
    ner_labels: np.ndarray,
    records: list[IndexRecord],
    want: set[int],
    exclude: set[int] | None = None,
) -> list[int]:
    """
    Vectorized: return sorted sentence indices that contain any label in `want`
    and NO token whose label is in `exclude` (if provided).
    """
    if not records:
        return []

    offsets = np.array([r.offset for r in records], dtype=np.int64)
    counts  = np.array([r.count  for r in records], dtype=np.int32)
    end_idx = int(offsets[-1]) + int(counts[-1])

    flat     = ner_labels[:end_idx]
    sent_ids = np.repeat(np.arange(len(records), dtype=np.int32), counts)

    want_arr   = np.array(sorted(want), dtype=flat.dtype)
    token_hits = np.isin(flat, want_arr)
    if not token_hits.any():
        return []

    hit_sents = set(np.unique(sent_ids[token_hits]).tolist())

    if exclude:
        excl_arr  = np.array(sorted(exclude), dtype=flat.dtype)
        excl_hits = np.isin(flat, excl_arr)
        if excl_hits.any():
            excl_sents = set(np.unique(sent_ids[excl_hits]).tolist())
            hit_sents -= excl_sents

    return sorted(hit_sents)


def find_code_switch_sequences(
    scores: np.ndarray,
    index_records: list[IndexRecord],
    threshold: float,
    min_consecutive: int,
    corpus: str,
    cfg: ScoringConfig,
    cqp_bin: str = DEFAULT_CQP_BIN,
    registry_dir: Optional[str] = None,
    lookup: int | None = None,        # renamed from limit_sentences
    mask_src: Optional[MaskSources] = None,
) -> list[CodeSwitchRun]:
    """Score all qualifying runs; caller is responsible for capping display."""
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
    for sent_idx, token_indices, token_scores in candidates:
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

        metrics = score_sentence(
            parsed,
            loanword_mask=lw_mask,
            code_switch_mask=cs_mask,
            named_entity_mask=ne_mask,
            cfg=cfg,
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
        "--ignore-misc",
        "ner_ignore_misc",
        is_flag=True,
        default=False,
        help="Treat MISC NER labels as O (exclude from named-entity mask).",
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


@query_group.command("code-switch")
@click.argument("corpus_name")
@click.argument(
    "surprisal_h5",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--threshold", type=float, required=True)
@click.option("--min-consecutive", type=int, default=2, show_default=True)
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
    help="Number of top results to display. 0 = show all.",
)
@click.option("--cqp-bin", default=DEFAULT_CQP_BIN, show_default=True)
@click.option("--registry-dir", default=None)
@scoring_config_option
@click.option(
    "--ner-h5",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="NER HDF5 index; builds NE mask. Falls back to heuristic if omitted.",
)
@click.option(
    "--loanwords",
    "loanword_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="One lemma per line; builds loanword mask.",
)
@click.option(
    "--ignore-misc",
    "ner_ignore_misc",
    is_flag=True,
    default=False,
    help="Treat MISC NER labels as O (exclude from named-entity mask).",
)
def query_code_switch(
    corpus_name,
    surprisal_h5,
    threshold,
    min_consecutive,
    lookup,
    results,
    cqp_bin,
    registry_dir,
    scoring_config,
    ner_h5,
    loanword_file,
    ner_ignore_misc,
):
    """Find code-switch sequences using a surprisal HDF5 index."""
    cfg = load_scoring_config(scoring_config, profile=QueryProfile.CODE_SWITCH)

    mask_src = load_mask_sources(
        ner_h5=ner_h5,
        loanword_file=loanword_file,
        ner_ignore_misc=ner_ignore_misc,
    )

    click.echo("[*] Loading surprisal index...", err=True)
    scores = _load_scores(surprisal_h5)
    index_records, _ = _load_index_records(surprisal_h5)

    found = find_code_switch_sequences(
        scores=scores,
        index_records=index_records,
        threshold=threshold,
        min_consecutive=min_consecutive,
        corpus=corpus_name,
        cfg=cfg,
        cqp_bin=cqp_bin,
        registry_dir=registry_dir,
        lookup=lookup,
        mask_src=mask_src,
    )

    click.echo(f"[*] Found and scored {len(found)} candidate sequences", err=True)
    shown = found if results == 0 else found[:results]
    if shown:
        _render_code_switch_results(shown, threshold)


@query_group.command("lemmas")
@click.argument("corpus_name")
@click.argument("lemmas")
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
    help="Number of top results to display.",
)
@click.option("--cqp-bin", default=DEFAULT_CQP_BIN, show_default=True)
@click.option("--registry-dir", default=None)
@scoring_config_option
@mask_source_options
def query_lemmas_command(
    corpus_name,
    lemmas,
    lookup,
    results,
    cqp_bin,
    registry_dir,
    scoring_config,
    surprisal_h5,
    surprisal_threshold,
    ner_h5,
    loanword_file,
    ner_ignore_misc,
):
    """Query corpus by lemma(s) and score results."""
    mask_src = load_mask_sources(
        surprisal_h5=surprisal_h5,
        surprisal_threshold=surprisal_threshold,
        ner_h5=ner_h5,
        loanword_file=loanword_file,
        ner_ignore_misc=ner_ignore_misc,
    )
    found = query_by_lemmas(
        corpus_name=corpus_name,
        lemmas=lemmas.split(","),
        lookup=lookup,
        cqp_bin=cqp_bin,
        registry_dir=registry_dir,
        scoring_config=scoring_config,
        verbose=True,
        mask_src=mask_src,
    )

    click.echo(f"\nTop {results} results for '{lemmas}':")
    click.echo("-" * 60)
    for r in found[:results]:
        click.echo(
            f"Score: {r.score_total:.4f}  "
            f"(len={r.score_length:.2f}  lw={r.score_loanword:.2f}  "
            f"cs={r.score_code_switch:.2f}  alpha={r.score_alpha:.2f}  "
            f"ne={r.score_named_entity:.2f})  | ID: {r.cqp_id}"
            f"{' [FILTERED: ' + r.filter_reason + ']' if r.filtered else ''}"
        )
        click.echo(" ".join(t.word for t in r.tokens))
        click.echo("-" * 60)


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
@click.option("--label",         "labels",  multiple=True, help="Include only these NER labels.")
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
    help="Number of top results to display. 0 = show all.",
)
@click.option("--cqp-bin", default=DEFAULT_CQP_BIN, show_default=True)
@click.option("--registry-dir", default=None)
@scoring_config_option
@click.option(
    "--surprisal-h5",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
)
@click.option("--surprisal-threshold", type=float, default=0.0, show_default=True)
@click.option(
    "--loanwords",
    "loanword_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
)
@click.option("--ignore-misc", "ner_ignore_misc", is_flag=True, default=False)
def query_ner_entities(
    corpus_name, ner_h5, labels, excl_labels,
    lookup, results, cqp_bin, registry_dir, scoring_config,
    surprisal_h5, surprisal_threshold, loanword_file, ner_ignore_misc,
):
    """Find and score sentences containing named entities."""
    cfg = load_scoring_config(scoring_config, profile=QueryProfile.NER)

    mask_src = load_mask_sources(
        surprisal_h5=surprisal_h5,
        surprisal_threshold=surprisal_threshold,
        loanword_file=loanword_file,
        ner_ignore_misc=ner_ignore_misc,
    )

    click.echo("[*] Loading NER index...", err=True)
    ner_labels, ner_records, id2label, ner_cpos = _load_ner_labels(ner_h5)
    mask_src.ner_labels   = ner_labels
    mask_src.ner_records  = ner_records
    mask_src.ner_id2label = id2label
    mask_src.ner_cpos     = ner_cpos

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
        f"[*] Scanning for:  {sorted(id2label[k] for k in want)}", err=True
    )
    if exclude:
        click.echo(
            f"[*] Excluding sentences with: {sorted(id2label[k] for k in exclude)}",
            err=True,
        )

    records = ner_records[:lookup] if lookup else ner_records
    matching = _ner_matching_sentences(ner_labels, records, want, exclude)

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

    scored_results: list[tuple[ScoredResult, str, int]] = []
    for parsed, sent_idx in tqdm(
        zip(parse_cwb_output(raw_output), matching),
        total=len(matching),
        desc="Scoring",
        unit="sent",
    ):
        lw_mask, cs_mask, ne_mask = build_masks(parsed, sent_idx, mask_src)
        scored = score_sentence(
            parsed,
            loanword_mask=lw_mask,
            code_switch_mask=cs_mask,
            named_entity_mask=ne_mask,
            cfg=cfg,
        )

        record = ner_records[sent_idx]
        chunk  = ner_labels[record.offset : record.offset + record.count]
        parts  = [
            f"[{t.word}/{id2label[int(chunk[i])]}]"
            if i < len(chunk) and int(chunk[i]) in want
            else t.word
            for i, t in enumerate(parsed.tokens)
        ]
        scored_results.append((scored, " ".join(parts), sent_idx))

    scored_results.sort(key=lambda x: x[0].score_total, reverse=True)
    click.echo(f"[*] Scored {len(scored_results)} sentence(s)", err=True)

    shown = scored_results if results == 0 else scored_results[:results]

    click.echo("-" * 60)
    for scored, parts_str, sent_idx in shown:
        status = f" [FILTERED: {scored.filter_reason}]" if scored.filtered else ""
        click.echo(
            f"Score: {scored.score_total:.4f}  "
            f"(len={scored.score_length:.2f}  lw={scored.score_loanword:.2f}  "
            f"cs={scored.score_code_switch:.2f}  "
            f"alpha={scored.score_alpha:.2f}  "
            f"ne={scored.score_named_entity:.2f})  "
            f"| ID: {scored.cqp_id}{status}"
        )
        click.echo(f"[{sent_idx}] " + parts_str)
        click.echo("-" * 60)

    click.echo(f"({len(shown)} of {len(scored_results)} sentences shown)")


if __name__ == "__main__":
    corpus()
