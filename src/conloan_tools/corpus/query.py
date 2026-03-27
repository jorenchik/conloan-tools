import subprocess
import os
import tempfile
import click
import json
from typing import List, Optional, Tuple, Literal
import re
from pathlib import Path
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

from conloan_tools.corpus import corpus
from .scoring import (
    Token,
    CQPResult,
    ScoringConfig,
    ScoredResult,
    load_scoring_config,
    score_sentence,
    build_loanword_mask,
    build_named_entity_mask,
)

DEFAULT_CQP_BIN = "cqp"
DEFAULT_RESULTS = 200

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
        "set PrintMode ascii;",
        "show -pos -lemma;",
        "show +pos +lemma;",
        "set PrintOptions noheader;",
    ]
    
    # Create a batch script with all commands
    script_lines = session_setup + commands
    
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
        if process.returncode != 0:
            raise click.ClickException(f"CQP Error: {stderr}")
        return stdout
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def _find_consecutive_runs(
    sent_scores: np.ndarray,
    threshold: float,
    min_consecutive: int,
) -> list[dict]:
    """
    Return all runs of consecutive tokens above `threshold` that are at
    least `min_consecutive` long.  Each entry is:
        {start, end, length, indices, scores}
    """
    runs: list[dict] = []
    current_start = 0
    current_run = 0

    for i, score in enumerate(sent_scores):
        if score > threshold:
            if current_run == 0:
                current_start = i
            current_run += 1
        else:
            if current_run >= min_consecutive:
                runs.append({
                    "start": current_start,
                    "end": i - 1,
                    "length": current_run,
                    "indices": list(range(current_start, i)),
                    "scores": sent_scores[current_start:i].tolist(),
                })
            current_run = 0

    # Flush trailing run
    if current_run >= min_consecutive:
        end = current_start + current_run
        runs.append({
            "start": current_start,
            "end": end - 1,
            "length": current_run,
            "indices": list(range(current_start, end)),
            "scores": sent_scores[current_start:end].tolist(),
        })

    return runs


def _load_scores(bin_path: Path) -> np.ndarray:
    return np.fromfile(bin_path, dtype=np.float32)


def _load_index_records(idx_path: Path) -> list[IndexRecord]:
    records: list[IndexRecord] = []
    for line in idx_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        records.append(IndexRecord(offset=rec["offset"], count=rec["count"]))
    return records


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
            f"  clean={res.score_cleanliness:.2f}"
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


def parse_cqp_line(line: str) -> Optional[CQPResult]:
    try:
        id_part, content_part = line.split(":", 1)
        cqp_id = int(id_part.strip())
    except ValueError:
        return None

    raw_tokens = content_part.strip().split()
    parsed_tokens: List[Token] = []
    match_index = -1
    current_index = 0

    for rt in raw_tokens:
        if rt in ("<g", "<s>", "</s>") or rt.startswith("</"):
            continue
        is_match_start = rt.startswith("<")
        if is_match_start:
            rt = rt[1:]
            if match_index == -1:
                match_index = current_index
        if rt.endswith(">"):
            rt = rt[:-1]
        if not rt:
            continue
        parts = rt.rsplit("/", 2)
        if len(parts) == 3:
            w, p, lemma = parts
        elif len(parts) == 2:
            w, p = parts
            lemma = w
        else:
            w = rt
            p = "UNK"
            lemma = w
        if w in ("/>", "<g/>", "<g", "") or w.startswith("</"):
            continue
        parsed_tokens.append(Token(word=w, pos=p, lemma=lemma))
        current_index += 1

    return CQPResult(
        cqp_id=cqp_id,
        tokens=parsed_tokens,
        match_index=match_index,
    )


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


def query_cqp_batch(
    corpus: str,
    queries: List[Tuple[str, Optional[int]]],
    cqp_bin: Optional[str] = None,
    registry_dir: Optional[str] = None,
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
            commands.append(f"reduce Results to {limit};")
        commands.append("cat Results;")
        output = _run_cqp_command(corpus, commands, cqp_bin, reg_dir, registry)
        outputs.append(output)
    
    return outputs


def build_or_query(lemmas: List[str]) -> str:
    """Build a CQP OR query from a list of lemmas."""
    escaped = [re.escape(l) for l in lemmas]
    return f'[lemma="{"|".join(escaped)}"]'


def query_by_lemmas(
    corpus_name: str,
    lemmas: List[str],
    limit: int = DEFAULT_RESULTS,
    cqp_bin: str = DEFAULT_CQP_BIN,
    registry_dir: str = None,
    scoring_config: str = None,
    deduplicate: bool = True,
    verbose: bool = False,
) -> List[ScoredResult]:

    """Library function: Performs query and scoring without printing."""
    # -- QUERY.
    raw_output = query_cqp_batch(
        corpus_name, 
        [(build_or_query(lemmas), limit)],
        cqp_bin,
        registry_dir
    )
    seen_texts: dict[tuple, ScoredResult] = {}
    scored: List[ScoredResult] = []

    # -- SCORE.
    cfg = load_scoring_config(scoring_config)
    lemma_set = {l.lower() for l in lemmas}
    
    lines = raw_output[0].splitlines()
    iterator = tqdm(lines, disable=not verbose)

    for line in iterator:
        if not line.strip():
            continue
        parsed = parse_cqp_line(line)
        if not parsed:
            continue
        
        lw_mask = build_loanword_mask(parsed, lemma_set)
        ne_mask = build_named_entity_mask(parsed)
        result = score_sentence(
            parsed,
            loanword_mask=lw_mask,
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
    max_results: int = 100,
) -> list[tuple[int, list[int], list[float]]]:
    """
    Scan pre-loaded scores against pre-parsed index records.

    Returns a list of (sentence_index, token_indices, token_scores) for the
    longest qualifying run per sentence, sorted by run length descending and
    capped at `max_results`.
    """
    candidates: list[tuple[int, list[dict]]] = []

    for sent_idx, record in enumerate(index_records):
        sent_scores = scores[record.offset : record.offset + record.count]
        runs = _find_consecutive_runs(sent_scores, threshold, min_consecutive)
        if runs:
            candidates.append((sent_idx, runs))

    candidates.sort(
        key=lambda x: max(r["length"] for r in x[1]),
        reverse=True,
    )

    if max_results > 0:
        candidates = candidates[:max_results]

    output: list[tuple[int, list[int], list[float]]] = []
    for sent_idx, runs in candidates:
        best = max(runs, key=lambda r: r["length"])
        output.append((sent_idx, best["indices"], best["scores"]))

    return output


def find_code_switch_sequences(
    scores: np.ndarray,
    index_records: list[IndexRecord],
    threshold: float,
    min_consecutive: int,
    corpus: str,
    cfg: ScoringConfig,
    max_results: int = DEFAULT_RESULTS,
    cqp_bin: str = DEFAULT_CQP_BIN,
    registry_dir: Optional[str] = None,
) -> list[CodeSwitchRun]:
    """
    Full code-switch detection pipeline.

    Args:
        scores:          Pre-loaded float32 token scores (entire corpus).
        index_records:   Pre-parsed index file records mapping sentences to
                         offsets in `scores`.
        threshold:       Per-token surprisal threshold.
        min_consecutive: Minimum consecutive tokens above threshold.
        corpus:          CWB corpus name.
        cfg:             Scoring configuration.
        max_results:     Cap on candidate sentences before CQP retrieval.
        cqp_bin:         Path to the CQP binary.
        registry_dir:    CWB registry directory.  Falls back to
                         CORPUS_REGISTRY env var when None.

    Returns:
        Scored and sorted list of CodeSwitchRun, best score first.
    """
    candidates = scan_anomaly_candidates(
        scores=scores,
        index_records=index_records,
        threshold=threshold,
        min_consecutive=min_consecutive,
        max_results=max_results,
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
    for line, sent_idx in zip(
        raw_output.strip().splitlines(), unique_sent_indices
    ):
        parsed = parse_cqp_line(line)
        if parsed:
            sentence_map[sent_idx] = parsed

    results: list[CodeSwitchRun] = []

    for sent_idx, token_indices, token_scores in candidates:

        parsed = sentence_map.get(sent_idx)
        if not parsed:
            continue

        valid_indices: list[int] = []
        valid_scores: list[float] = []
        for idx, score in zip(token_indices, token_scores):
            if idx < len(parsed.tokens):
                valid_indices.append(idx)
                valid_scores.append(score)

        if len(valid_indices) < min_consecutive:
            continue

        cs_mask = [i in set(valid_indices) for i in range(len(parsed.tokens))]
        ne_mask = build_named_entity_mask(parsed)
        metrics = score_sentence(
            parsed,
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
@click.argument("input_prefix")
@click.option(
    "--threshold",
    type=float,
    required=True,
    help="Per-token LM surprisal threshold.",
)
@click.option(
    "--min-consecutive",
    type=int,
    default=2,
    show_default=True,
    help="Minimum consecutive tokens above threshold.",
)
@click.option(
    "--max-results",
    type=int,
    default=DEFAULT_RESULTS,
    show_default=True,
    help="Maximum sequences to retrieve and score.",
)
@click.option("--cqp-bin", default=DEFAULT_CQP_BIN, show_default=True)
@click.option(
    "--registry-dir",
    default=None,
    help="CWB registry dir. Falls back to CORPUS_REGISTRY env var.",
)
@scoring_config_option
def query_code_switch(
    corpus_name: str,
    input_prefix: str,
    threshold: float,
    min_consecutive: int,
    max_results: int,
    cqp_bin: str,
    registry_dir: Optional[str],
    scoring_config: Optional[str],
) -> None:
    """Find code-switch sequences in CORPUS_NAME using a pre-scored index.

    INPUT_PREFIX: base path for .bin/.idx files (e.g. 'output' → output.bin/output.idx)
    """
    bin_path = Path(input_prefix).with_suffix(".bin")
    idx_path = Path(input_prefix).with_suffix(".idx")

    if not bin_path.exists():
        raise click.FileError(str(bin_path), hint="Binary score file not found.")
    if not idx_path.exists():
        raise click.FileError(str(idx_path), hint="Index file not found.")

    click.echo("[*] Loading scores and index into memory...")
    scores = _load_scores(bin_path)
    index_records = _load_index_records(idx_path)
    cfg = load_scoring_config(scoring_config)

    click.echo(
        f"[*] Scanning {bin_path.name} for ≥{min_consecutive} "
        f"consecutive tokens > {threshold:.2f}"
    )

    results = find_code_switch_sequences(
        scores=scores,
        index_records=index_records,
        threshold=threshold,
        min_consecutive=min_consecutive,
        corpus=corpus_name,
        cfg=cfg,
        max_results=max_results,
        cqp_bin=cqp_bin,
        registry_dir=registry_dir,
    )

    click.echo(f"[*] Found {len(results)} candidate sequences")
    if not results:
        return

    _render_code_switch_results(results, threshold)


@query_group.command("lemmas")
@click.argument("corpus_name")
@click.argument("lemmas")
@click.option(
    "--limit",
    type=int,
    default=DEFAULT_RESULTS,
    show_default=True,
    help="Max results.",
)
@click.option(
    "--cqp-bin",
    default=DEFAULT_CQP_BIN,
    show_default=True,
    help="Path to cqp binary.",
)
@click.option(
    "--registry-dir",
    default=None,
    help="Path to cwb directory.",
)
@scoring_config_option
def query_lemmas_command(corpus_name, lemmas, limit, cqp_bin, registry_dir, scoring_config):
    """CLI wrapper: Handles UI and printing."""
    click.echo(f"Scoring results...")
    
    # Call logic function
    results = query_by_lemmas(
        corpus_name=corpus_name,
        lemmas=lemmas.split(","),
        limit=limit,
        cqp_bin=cqp_bin,
        registry_dir=registry_dir,
        scoring_config=scoring_config,
        verbose=True,  # Enable progress bar for CLI
    )

    # -- SHOW RESULTS.
    click.echo(f"Top 5 Results for '{lemmas}':")
    click.echo("-" * 60)
    for r in results[:5]:
        click.echo(
            f"Score: {r.score_total:.4f}  "
            f"(len={r.score_length:.2f}  lw={r.score_loanword:.2f}  "
            f"cs={r.score_code_switch:.2f}  clean={r.score_cleanliness:.2f}  "
            f"ne={r.score_named_entity:.2f})  | ID: {r.cqp_id}"
            f"{' [FILTERED: ' + r.filter_reason + ']' if r.filtered else ''}"
        )
        click.echo(" ".join([t.word for t in r.tokens]))
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
    
    for line in raw_output.splitlines():
        parsed = parse_cqp_line(line)
        if parsed:
            # Simple text reconstruction for display
            text = " ".join([t.word for t in parsed.tokens])
            click.echo(f"[{parsed.cqp_id}] {text}")

if __name__ == "__main__":
    corpus()
