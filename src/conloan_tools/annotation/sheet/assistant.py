"""assistant - Interactive CLI for loanword annotation management."""

import re
import hashlib
import click
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict
from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from tqdm import tqdm

from conloan_tools.stz.lemmatize import Lemmatizer
from .validate_sheet import (
    validate_row,
    validate_file,
    MODE_CONFIG,
    REQUIRED_COLUMNS,
    ValidationError,
    RowResult,
)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TokenEntry:
    """A single token (word) from a sentence."""
    file: str
    row_index: int
    prefix: str              # tag prefix or "o"
    tag_id: Optional[str]    # None if prefix is "o"
    surface: str
    lemma_key: str


@dataclass
class CorpusEntry:
    """A labeled span from a row."""
    file: str
    row_index: int
    tag_id: str
    prefix: str
    surface: str
    lemma_key: str
    paired_tag_id: str
    paired_surface: str
    paired_lemma_key: str
    is_replaced: bool


@dataclass
class CorpusIndex:
    """The complete in-memory corpus index."""
    entries: List[CorpusEntry] = field(default_factory=list)
    tokens: List[TokenEntry] = field(default_factory=list)
    unlabeled_tokens: Dict[Tuple[str, int], List[Tuple[str, str]]] = field(default_factory=dict)


@dataclass
class AssistantSession:
    """Main session state for the assistant."""
    files: Dict[str, pd.DataFrame]           # keyed by name-no-ext
    row_hashes: Dict[str, Dict[str, int]]    # name-no-ext -> hash -> row_index
    lemmatizer: Lemmatizer
    mode: str
    validate_all: bool
    output_target: str                        # "stdout" or file path
    output_mode: str                           # "append" or "replace"
    active_target: Optional[str]              # name-no-ext
    index: CorpusIndex
    lemma_cache: Dict[str, str] = field(default_factory=dict)  # surface -> lemma_key
    _file_paths: Dict[str, str] = field(default_factory=dict)  # name-no-ext -> path


# =============================================================================
# Helper Functions
# =============================================================================

def compute_row_hash(label_sent: str, replacement_sent: str) -> str:
    """Compute SHA-1 hash for row identity."""
    combined = f"{label_sent}\x00{replacement_sent}"
    return hashlib.sha1(combined.encode('utf-8')).hexdigest()


def get_tag_spans(text: str) -> List[Tuple[str, str, int, int]]:
    """Extract all tag spans: (prefix, content, start_pos, end_pos)."""
    spans = []
    for tag_match in re.finditer(r"<([A-Z]+)(\d+)>", text):
        prefix = tag_match.group(1)
        num = int(tag_match.group(2))
        start = tag_match.start()
        close_match = re.search(rf"</{prefix}{num}>", text)
        if close_match:
            end = close_match.end()
            content = text[start + len(f"<{prefix}{num}>"):end - len(f"</{prefix}{num}>")]
            spans.append((prefix, content, start, end))
    return spans


def strip_tags(text: str) -> str:
    """Remove all tags from text, returning plain text."""
    return re.sub(r'<[^>]+>', '', text)


def _resolve_lemma_key(
    surface: str,
    lemmatizer: Lemmatizer,
    cache: Dict[str, str],
) -> str:
    """Return cached lemma key for surface, computing and storing if absent."""
    if surface not in cache:
        lemmas = lemmatizer.lemmatize(surface)
        cache[surface] = " ".join(lemmas).lower()
    return cache[surface]


def build_corpus_index(
    files: Dict[str, pd.DataFrame],
    row_hashes: Dict[str, Dict[str, int]],
    lemmatizer: Lemmatizer,
    mode: str,
    validate_all: bool,
    lemma_cache: Optional[Dict[str, str]] = None,
) -> CorpusIndex:
    """Build the complete corpus index from all loaded files."""
    if lemma_cache is None:
        lemma_cache = {}

    index = CorpusIndex()

    for file_name, df in tqdm(files.items(), desc="files", unit="file"):
        valid_rows = [
            (idx, row) for idx, row in df.iterrows()
            if validate_all
            or str(row.get("Valid", "")).strip().lower() == "+"
        ]

        # Bulk-process label sentences (plain text) for token-level NLP.
        label_texts = [
            strip_tags(str(row.get("Label sentence", "")))
            for _, row in valid_rows
        ]
        label_docs = lemmatizer._nlp.bulk_process(label_texts) if label_texts else []

        # Bulk-process replacement sentences for span-level lemmatisation.
        all_span_surfaces: List[str] = []
        for _, row in valid_rows:
            for sent_col in ("Label sentence", "Replacement sentence"):
                sent = str(row.get(sent_col, ""))
                for _, content, _, _ in get_tag_spans(sent):
                    if content not in lemma_cache:
                        all_span_surfaces.append(content)

        unique_new = list(dict.fromkeys(all_span_surfaces))
        if unique_new:
            span_docs = lemmatizer._nlp.bulk_process(unique_new)
            for surface, doc in zip(unique_new, span_docs):
                if surface not in lemma_cache:
                    lemmas = [
                        w.lemma.lower()
                        for sent in doc.sentences
                        for w in sent.words
                    ]
                    lemma_cache[surface] = " ".join(lemmas)

        for (idx, row), doc in tqdm(
            zip(valid_rows, label_docs),
            desc=file_name,
            total=len(valid_rows),
            unit="row",
            leave=False,
        ):
            row_index = idx + 2
            label_sent = str(row.get("Label sentence", ""))
            replacement_sent = str(row.get("Replacement sentence", ""))

            _process_sentence(
                index, file_name, row_index, label_sent,
                replacement_sent, lemmatizer, is_label=True, doc=doc
            )
            _process_paired_sentence(
                index, file_name, row_index, replacement_sent,
                label_sent, lemmatizer, lemma_cache
            )

    return index


def _build_char_label_map(
    sentence: str,
) -> Dict[int, Tuple[str, str]]:
    """Map each plain-text character index to (prefix, tag_id).

    Iterates the tagged sentence once, tracking the current open tag so
    that every character that belongs to a span — including characters
    inside multi-word spans — is labelled correctly.
    """
    result: Dict[int, Tuple[str, str]] = {}
    tag_re = re.compile(r"<(/?)([A-Z]+)(\d+)>")
    plain_pos = 0
    i = 0
    current_label: Optional[Tuple[str, str]] = None  # (prefix, tag_id)

    while i < len(sentence):
        m = tag_re.match(sentence, i)
        if m:
            is_close = m.group(1) == "/"
            prefix = m.group(2)
            tag_id = f"{prefix}{m.group(3)}"
            current_label = None if is_close else (prefix, tag_id)
            i = m.end()
        else:
            if current_label is not None:
                result[plain_pos] = current_label
            plain_pos += 1
            i += 1

    return result


def _process_sentence(
    index: CorpusIndex,
    file_name: str,
    row_index: int,
    sentence: str,
    other_sentence: str,
    lemmatizer: Lemmatizer,
    is_label: bool,
    doc=None,
) -> None:
    """Process a sentence to extract tokens and spans."""
    # Build a plain-text-position → (prefix, tag_id) map so that every
    # character inside a span — including multi-word spans — is labelled.
    char_label_map = _build_char_label_map(sentence)

    plain_text = strip_tags(sentence)

    # Tokenize using Stanza
    if doc is None:
        doc = lemmatizer._nlp(plain_text)
    for sent in doc.sentences:
        for word in sent.words:
            surface = word.text
            lemma = word.lemma.lower()

            # A token is labelled if *any* of its characters map to a span.
            # Using the first character is sufficient because tags never
            # split a token mid-word.
            label = char_label_map.get(word.start_char)

            if label is not None:
                prefix, tag_id = label
                token_entry = TokenEntry(
                    file=file_name,
                    row_index=row_index,
                    prefix=prefix,
                    tag_id=tag_id,
                    surface=surface,
                    lemma_key=lemma,
                )
                index.tokens.append(token_entry)
            else:
                token_entry = TokenEntry(
                    file=file_name,
                    row_index=row_index,
                    prefix="o",
                    tag_id=None,
                    surface=surface,
                    lemma_key=lemma,
                )
                index.tokens.append(token_entry)

                key = (file_name, row_index)
                if key not in index.unlabeled_tokens:
                    index.unlabeled_tokens[key] = []
                index.unlabeled_tokens[key].append((surface, lemma))


def _process_paired_sentence(
    index: CorpusIndex,
    file_name: str,
    row_index: int,
    paired_sentence: str,
    original_sentence: str,
    lemmatizer: Lemmatizer,
    lemma_cache: Optional[Dict[str, str]] = None,
) -> None:
    """Process paired sentence to build corpus entries."""
    if lemma_cache is None:
        lemma_cache = {}

    # Get tag spans from original sentence
    original_spans = {}
    for prefix, content, start, end in get_tag_spans(original_sentence):
        num_match = re.search(r'(\d+)', original_sentence[start:])
        if num_match:
            tag_id = f"{prefix}{num_match.group(1)}"
            original_spans[tag_id] = (prefix, content, start, end)

    # Get tag spans from paired sentence
    paired_spans = {}
    for prefix, content, start, end in get_tag_spans(paired_sentence):
        num_match = re.search(r'(\d+)', paired_sentence[start:])
        if num_match:
            tag_id = f"{prefix}{num_match.group(1)}"
            paired_spans[tag_id] = (prefix, content, start, end)

    # Build corpus entries for labeled spans in original
    for tag_id, (prefix, content, start, end) in original_spans.items():
        lemma_key = _resolve_lemma_key(content, lemmatizer, lemma_cache)

        # Find paired tag
        paired_tag_id = None
        paired_content = ""
        paired_lemma_key = ""

        if prefix == "L" and f"N{tag_id[1:]}" in paired_spans:
            paired_tag_id = f"N{tag_id[1:]}"
            paired_content = paired_spans[paired_tag_id][1]
            paired_lemma_key = _resolve_lemma_key(paired_content, lemmatizer, lemma_cache)
        elif prefix == "N" and f"L{tag_id[1:]}" in paired_spans:
            paired_tag_id = f"L{tag_id[1:]}"
            paired_content = paired_spans[paired_tag_id][1]
            paired_lemma_key = _resolve_lemma_key(paired_content, lemmatizer, lemma_cache)
        elif prefix == "CS" and f"CN{tag_id[2:]}" in paired_spans:
            paired_tag_id = f"CN{tag_id[2:]}"
            paired_content = paired_spans[paired_tag_id][1]
            paired_lemma_key = _resolve_lemma_key(paired_content, lemmatizer, lemma_cache)
        elif prefix == "CN" and f"CS{tag_id[2:]}" in paired_spans:
            paired_tag_id = f"CS{tag_id[2:]}"
            paired_content = paired_spans[paired_tag_id][1]
            paired_lemma_key = _resolve_lemma_key(paired_content, lemmatizer, lemma_cache)
        elif prefix == "NE":
            # NE is identity pair
            if tag_id in paired_spans:
                paired_tag_id = tag_id
                paired_content = paired_spans[tag_id][1]
                paired_lemma_key = _resolve_lemma_key(paired_content, lemmatizer, lemma_cache)
        
        is_replaced = paired_content != content if paired_content else False
        # If paired_content is empty but tag existed, there was no actual content to compare
        # so don't mark as replaced - the span existed but had no text
        if paired_content == "" and paired_tag_id:
            is_replaced = False
        
        entry = CorpusEntry(
            file=file_name,
            row_index=row_index,
            tag_id=tag_id,
            prefix=prefix,
            surface=content,
            lemma_key=lemma_key,
            paired_tag_id=paired_tag_id or "",
            paired_surface=paired_content,
            paired_lemma_key=paired_lemma_key,
            is_replaced=is_replaced
        )
        index.entries.append(entry)


# =============================================================================
# File Reload
# =============================================================================

def _reload_file(session: AssistantSession, file_name: str) -> None:
    """Reload a single file and update the corpus index."""
    file_path = session._file_paths[file_name]
    
    try:
        # Read the file
        new_df = pd.read_excel(file_path)
        
        # Compute new hashes
        new_hashes = {}
        for idx, row in new_df.iterrows():
            label_sent = str(row.get("Label sentence", ""))
            replacement_sent = str(row.get("Replacement sentence", ""))
            row_hash = compute_row_hash(label_sent, replacement_sent)
            new_hashes[row_hash] = idx + 2  # 1-based row index
        
        # Get old hashes
        old_hashes = session.row_hashes.get(file_name, {})
        
        old_hash_set = set(old_hashes.keys())
        new_hash_set = set(new_hashes.keys())
        added_count = len(new_hash_set - old_hash_set)
        removed_count = len(old_hash_set - new_hash_set)

        # Update DataFrame and hashes
        session.files[file_name] = new_df
        session.row_hashes[file_name] = new_hashes

        # Full re-index for this file: drop all existing entries/tokens then
        # rebuild from scratch, identical to what build_corpus_index does.
        session.index.entries = [
            e for e in session.index.entries if e.file != file_name
        ]
        session.index.tokens = [
            t for t in session.index.tokens if t.file != file_name
        ]
        session.index.unlabeled_tokens = {
            k: v
            for k, v in session.index.unlabeled_tokens.items()
            if k[0] != file_name
        }

        valid_rows = [
            (idx, row) for idx, row in new_df.iterrows()
            if session.validate_all
            or str(row.get("Valid", "")).strip().lower() == "+"
        ]

        texts = [
            strip_tags(str(row.get("Label sentence", "")))
            for _, row in valid_rows
        ]
        docs = session.lemmatizer._nlp.bulk_process(texts) if texts else []

        for (idx, row), doc in tqdm(
            zip(valid_rows, docs),
            desc=file_name,
            total=len(valid_rows),
            unit="row",
            leave=False,
        ):
            row_index = idx + 2
            label_sent = str(row.get("Label sentence", ""))
            replacement_sent = str(row.get("Replacement sentence", ""))

            _process_sentence(
                session.index, file_name, row_index, label_sent,
                replacement_sent, session.lemmatizer, is_label=True, doc=doc
            )
            _process_paired_sentence(
                session.index, file_name, row_index, replacement_sent,
                label_sent, session.lemmatizer, session.lemma_cache
            )

        return (
            f"[reload] {file_name}.xlsx: "
            f"+{added_count} added, -{removed_count} removed rows.",
            False,
        )

    except Exception as e:
        return (f"Error reloading {file_name}: {str(e)}", True)


# =============================================================================
# REPL Commands
# =============================================================================

def cmd_target(session: AssistantSession, args: List[str]) -> Tuple[str, bool]:
    """Set or show active target file."""
    if not args:
        if session.active_target:
            return (f"Current target: '{session.active_target}'", False)
        return ("No target set.", False)

    target_name = args[0].lower()
    matched = None
    for name in session.files.keys():
        if name.lower() == target_name:
            matched = name
            break

    if not matched:
        available = ", ".join(sorted(session.files.keys()))
        return (f"Unknown target '{args[0]}'. Available: {available}", True)

    session.active_target = matched
    return (f"Target set to '{matched}'.", False)


def cmd_validate(session: AssistantSession, args: List[str]) -> Tuple[str, bool]:
    """Validate rows in target file."""
    if not session.active_target:
        return ("No target selected. Use 'target <name>' first.", True)

    file_name = session.active_target
    df = session.files[file_name]

    results = []
    for idx, row in df.iterrows():
        row_index = idx + 2
        if not session.validate_all:
            row_valid = str(row.get("Valid", "")).strip().lower()
            if row_valid != "+":
                continue

        errors = validate_row(row, session.mode)
        if errors:
            results.append(RowResult(row_index=row_index, errors=errors))

    if not results:
        return ("No issues found.", False)

    error_rows = sum(1 for r in results if any(e.severity == "error" for e in r.errors))
    warn_rows = sum(1 for r in results if all(e.severity == "warning" for e in r.errors))
    summary = f"{len(results)} rows with issues ({error_rows} errors, {warn_rows} warnings)"

    output_lines = [summary]
    for result in results:
        output_lines.append(f"{result.row_index}:")
        for err in result.errors:
            prefix = "[warn] " if err.severity == "warning" else ""
            output_lines.append(f"    - {prefix}[{err.rule_id}] {err.field}: {err.message}")

    return ("\n".join(output_lines), False)


def cmd_contradict(session: AssistantSession, args: List[str]) -> Tuple[str, bool]:
    """Report labeling contradictions across all loaded files."""
    # Collect tokens with their lemma_key and label status
    # lemma_key -> list of (file, row_index, prefix, surface)
    token_occurrences: Dict[str, List[Tuple[str, int, str, str]]] = defaultdict(list)

    for token in session.index.tokens:
        token_occurrences[token.lemma_key].append(
            (token.file, token.row_index, token.prefix, token.surface)
        )

    contradictions = []

    for lemma_key, occurrences in token_occurrences.items():
        prefixes_in_use = {prefix for _, _, prefix, _ in occurrences}

        has_labeled = any(p != 'o' for p in prefixes_in_use)
        has_unlabeled = 'o' in prefixes_in_use

        if (has_labeled and has_unlabeled) or len(prefixes_in_use) > 1:
            contradictions.append({
                'lemma_key': lemma_key,
                'labeling': occurrences,
            })

    if not contradictions:
        return ("No contradictions found.", False)

    output_lines = []
    for c in sorted(contradictions, key=lambda x: x['lemma_key']):
        output_lines.append(f"{c['lemma_key']}:")
        output_lines.append("    [labeling]")
        for file_name, row_index, prefix, surface in sorted(
            c['labeling'], key=lambda x: (x[0], x[1])
        ):
            output_lines.append(
                f"    - {file_name}:{row_index}: {prefix}, {surface}"
            )

    return ("\n".join(output_lines), False)


def cmd_replace(session: AssistantSession, args: List[str]) -> Tuple[str, bool]:
    """List available replacement lemma keys."""
    if not session.active_target:
        return ("No target selected. Use 'target <name>' first.", True)
    
    file_name = session.active_target
    
    # Collect lemma keys and their replacements
    lemma_replacements = defaultdict(lambda: defaultdict(set))
    
    for entry in session.index.entries:
        if entry.file != file_name:
            continue
        if entry.prefix != "L":
            continue

        # Sheet source: span-level paired lemma key
        lemma_replacements[entry.lemma_key]['sheet'].add(entry.paired_lemma_key)

    if not lemma_replacements:
        return ("No labeled spans found in specified rows.", False)

    output_lines = []
    for lemma_key in sorted(lemma_replacements.keys()):
        sources = lemma_replacements[lemma_key]
        output_lines.append(f"{lemma_key}:")

        has_replacements = False
        for src_name, repl_keys in sources.items():
            if not repl_keys:
                continue
            has_replacements = True
            for repl_key in sorted(repl_keys):
                output_lines.append(f"    - {repl_key} ({file_name})")

        if not has_replacements:
            output_lines.append("    (no replacements found)")

    return ("\n".join(output_lines), False)


def cmd_stats(session: AssistantSession, args: List[str]) -> Tuple[str, bool]:
    """Print aggregate and per-file statistics."""
    # Compute global stats
    total_words = len(session.index.tokens)
    total_lemma_variants = len(set(t.lemma_key for t in session.index.tokens))
    o_label_words = sum(1 for t in session.index.tokens if t.prefix == "o")
    o_lemma_variants = len(set(t.lemma_key for t in session.index.tokens if t.prefix == "o"))
    
    L_spans = sum(1 for e in session.index.entries if e.prefix == "L")
    L_words = sum(1 for t in session.index.tokens if t.prefix == "L")
    L_lemma_variants = len(set(t.lemma_key for t in session.index.tokens if t.prefix == "L"))
    L_replaced = sum(1 for e in session.index.entries if e.prefix == "L" and e.is_replaced)
    
    CS_spans = sum(1 for e in session.index.entries if e.prefix == "CS")
    CS_words = sum(1 for t in session.index.tokens if t.prefix == "CS")
    CS_lemma_variants = len(set(t.lemma_key for t in session.index.tokens if t.prefix == "CS"))
    CS_replaced = sum(1 for e in session.index.entries if e.prefix == "CS" and e.is_replaced)
    
    NE_spans = sum(1 for e in session.index.entries if e.prefix == "NE")
    NE_words = sum(1 for t in session.index.tokens if t.prefix == "NE")
    NE_lemma_variants = len(set(t.lemma_key for t in session.index.tokens if t.prefix == "NE"))
    
    # Precompute validated (file, row_index) pairs in one pass per file.
    validated_pairs: Set[Tuple[str, int]] = set()
    for fn, df in session.files.items():
        for idx, row in df.iterrows():
            if str(row.get("Valid", "")).strip().lower() == "+":
                validated_pairs.add((fn, idx + 2))

    # Sentence counts: total = any row with "+" or "-" in Valid column.
    total_sentence_pairs: Set[Tuple[str, int]] = set()
    for fn, df in session.files.items():
        for idx, row in df.iterrows():
            valid_val = str(row.get("Valid", "")).strip().lower()
            if valid_val in ("+", "-"):
                total_sentence_pairs.add((fn, idx + 2))
    total_sentences = len(total_sentence_pairs)
    validated_sentences = sum(1 for p in total_sentence_pairs if p in validated_pairs)
    
    # Duplicate lemma variants: lemma keys appearing in more than one span entry.
    global_lemma_counts: Dict[str, int] = defaultdict(int)
    for e in session.index.entries:
        global_lemma_counts[e.lemma_key] += 1
    duplicate_lemma_variants = sum(1 for c in global_lemma_counts.values() if c > 1)
    
    # Total span count
    total_spans = len(session.index.entries)
    
    def pct_str(n, total):
        if total == 0:
            return "0.0"
        return f"{round(100 * n / total, 1)}"
    
    lines = ["total:"]
    lines.append(f"    - total sentences:       {total_sentences}")
    lines.append(f"    - validated sentences:   {validated_sentences}")
    lines.append(f"    - total words:           {total_words}")
    lines.append(f"    - total lemma variants: {total_lemma_variants}")
    lines.append(f"    - total spans:           {total_spans}")
    lines.append(f"    - duplicate lemma vars:  {duplicate_lemma_variants}")
    lines.append(f"    - o label words:         {o_label_words}")
    lines.append(f"    - o lemma variants:      {o_lemma_variants}")
    lines.append(f"    - L spans:               {L_spans}")
    lines.append(f"    - L words:               {L_words} ({pct_str(L_replaced, L_words)}% replaced)")
    lines.append(f"    - L lemma variants:      {L_lemma_variants}")
    
    if session.mode == "extended":
        lines.append(f"    - CS spans:              {CS_spans}")
        lines.append(f"    - CS words:              {CS_words} ({pct_str(CS_replaced, CS_words)}% replaced)")
        lines.append(f"    - CS lemma variants:     {CS_lemma_variants}")
        lines.append(f"    - NE spans:              {NE_spans}")
        lines.append(f"    - NE words:              {NE_words}")
        lines.append(f"    - NE lemma variants:     {NE_lemma_variants}")
    
    # Per-file stats
    for file_name in sorted(session.files.keys()):
        f_tokens = [t for t in session.index.tokens if t.file == file_name]
        f_entries = [e for e in session.index.entries if e.file == file_name]
        
        f_total = len(f_tokens)
        f_total_lemma = len(set(t.lemma_key for t in f_tokens))
        f_o = sum(1 for t in f_tokens if t.prefix == "o")
        f_o_lemma = len(set(t.lemma_key for t in f_tokens if t.prefix == "o"))
        f_L_spans = sum(1 for e in f_entries if e.prefix == "L")
        f_L_words = sum(1 for t in f_tokens if t.prefix == "L")
        f_L_lemma = len(set(t.lemma_key for t in f_tokens if t.prefix == "L"))
        f_L_replaced = sum(1 for e in f_entries if e.prefix == "L" and e.is_replaced)
        f_CS_spans = sum(1 for e in f_entries if e.prefix == "CS")
        f_CS_words = sum(1 for t in f_tokens if t.prefix == "CS")
        f_CS_lemma = len(set(t.lemma_key for t in f_tokens if t.prefix == "CS"))
        f_CS_replaced = sum(1 for e in f_entries if e.prefix == "CS" and e.is_replaced)
        f_NE_spans = sum(1 for e in f_entries if e.prefix == "NE")
        f_NE_words = sum(1 for t in f_tokens if t.prefix == "NE")
        f_NE_lemma = len(set(t.lemma_key for t in f_tokens if t.prefix == "NE"))
        
        f_df = session.files[file_name]
        f_sentence_pairs: Set[int] = set()
        for idx, row in f_df.iterrows():
            valid_val = str(row.get("Valid", "")).strip().lower()
            if valid_val in ("+", "-"):
                f_sentence_pairs.add(idx + 2)
        f_sentences = len(f_sentence_pairs)
        f_validated = sum(
            1 for row_idx in f_sentence_pairs
            if (file_name, row_idx) in validated_pairs
        )
        
        f_lemma_counts: Dict[str, int] = defaultdict(int)
        for e in f_entries:
            f_lemma_counts[e.lemma_key] += 1
        f_duplicate_lemma_variants = sum(1 for c in f_lemma_counts.values() if c > 1)
        
        f_total_spans = len(f_entries)
        
        lines.append(f"\n{file_name}:")
        lines.append(f"    - total sentences:       {f_sentences}")
        lines.append(f"    - validated sentences:   {f_validated}")
        lines.append(f"    - total words:           {f_total}")
        lines.append(f"    - total lemma variants: {f_total_lemma}")
        lines.append(f"    - total spans:           {f_total_spans}")
        lines.append(f"    - duplicate lemma vars:  {f_duplicate_lemma_variants}")
        lines.append(f"    - o label words:         {f_o}")
        lines.append(f"    - o lemma variants:      {f_o_lemma}")
        lines.append(f"    - L spans:               {f_L_spans}")
        lines.append(f"    - L words:               {f_L_words} ({pct_str(f_L_replaced, f_L_words)}% replaced)")
        lines.append(f"    - L lemma variants:      {f_L_lemma}")
        
        if session.mode == "extended":
            lines.append(f"    - CS spans:              {f_CS_spans}")
            lines.append(f"    - CS words:              {f_CS_words} ({pct_str(f_CS_replaced, f_CS_words)}% replaced)")
            lines.append(f"    - CS lemma variants:     {f_CS_lemma}")
            lines.append(f"    - NE spans:              {f_NE_spans}")
            lines.append(f"    - NE words:              {f_NE_words}")
            lines.append(f"    - NE lemma variants:     {f_NE_lemma}")
    
    return ("\n".join(lines), False)


def cmd_mode(session: AssistantSession, args: List[str]) -> Tuple[str, bool]:
    """Change or show the active mode."""
    if not args:
        return (f"Current mode: {session.mode}", False)

    new_mode = args[0].lower()
    if new_mode not in ("baseline", "extended"):
        return (f"Invalid mode '{new_mode}'. Use 'baseline' or 'extended'.", True)

    if new_mode == session.mode:
        return (f"Mode already set to '{new_mode}'.", False)

    session.mode = new_mode

    total_rows = 0
    for df in session.files.values():
        for idx, row in df.iterrows():
            row_valid = str(row.get("Valid", "")).strip().lower()
            if not session.validate_all and row_valid != "+":
                continue
            total_rows += 1

    click.echo("Rebuilding index...")
    session.index = build_corpus_index(
        session.files,
        session.row_hashes,
        session.lemmatizer,
        session.mode,
        session.validate_all,
        session.lemma_cache,
    )

    return (f"Mode set to '{new_mode}'. Index rebuilt: {total_rows} rows processed across {len(session.files)} files.", False)


def cmd_output(session: AssistantSession, args: List[str]) -> Tuple[str, bool]:
    """Change output target."""
    if not args:
        return (f"Current output: {session.output_target}", False)

    new_target = args[0]
    if new_target != "stdout":
        p = Path(new_target)
        if not p.parent.exists():
            return (f"Directory '{p.parent}' does not exist.", True)
        try:
            with open(p, "a"):
                pass
        except OSError as e:
            return (f"Cannot write to '{new_target}': {e}", True)

    session.output_target = new_target
    return (f"Output set to '{new_target}'.", False)


def cmd_output_mode(session: AssistantSession, args: List[str]) -> Tuple[str, bool]:
    """Change or show output mode (append or replace)."""
    if not args:
        return (f"Current output-mode: {session.output_mode}", False)

    new_mode = args[0].lower()
    if new_mode not in ("append", "replace"):
        return (f"Invalid output-mode '{new_mode}'. Use 'append' or 'replace'.", True)

    if new_mode == session.output_mode:
        return (f"Output-mode already set to '{new_mode}'.", False)

    session.output_mode = new_mode
    return (f"Output-mode set to '{new_mode}'.", False)


def cmd_reload(session: AssistantSession, args: List[str]) -> Tuple[str, bool]:
    """Reload the target file from disk."""
    if not session.active_target:
        return ("No target selected. Use 'target <name>' first.", True)

    return _reload_file(session, session.active_target)


def cmd_validate_mode(session: AssistantSession, args: List[str]) -> Tuple[str, bool]:
    """Change or show the validate mode (which rows are included in search)."""
    if not args:
        mode = "all" if session.validate_all else "valid"
        return (f"Current validate-mode: {mode}", False)

    new_mode = args[0].lower()
    if new_mode not in ("valid", "all"):
        return (f"Invalid validate-mode '{new_mode}'. Use 'valid' or 'all'.", True)

    if new_mode == "valid":
        if not session.validate_all:
            return ("Validate-mode already set to 'valid'.", False)
        session.validate_all = False
    else:
        if session.validate_all:
            return ("Validate-mode already set to 'all'.", False)
        session.validate_all = True

    total_rows = 0
    for df in session.files.values():
        for idx, row in df.iterrows():
            row_valid = str(row.get("Valid", "")).strip().lower()
            if not session.validate_all and row_valid != "+":
                continue
            total_rows += 1

    click.echo("Rebuilding index...")
    session.index = build_corpus_index(
        session.files,
        session.row_hashes,
        session.lemmatizer,
        session.mode,
        session.validate_all,
        session.lemma_cache,
    )

    return (f"Validate-mode set to '{new_mode}'. Index rebuilt: {total_rows} rows.", False)


def cmd_contradict_repl(session: AssistantSession, args: List[str]) -> Tuple[str, bool]:
    """Report replacement contradictions across all loaded files."""
    # lemma_key -> {paired_lemma_key -> [(file, row_index), ...]}
    source_to_repls: Dict[str, Dict[str, List[Tuple[str, int]]]] = defaultdict(lambda: defaultdict(list))
    # paired_lemma_key -> {lemma_key -> [(file, row_index), ...]}
    repl_to_sources: Dict[str, Dict[str, List[Tuple[str, int]]]] = defaultdict(lambda: defaultdict(list))

    for entry in session.index.entries:
        if entry.prefix != "L":
            continue
        if not entry.paired_lemma_key:
            continue

        source_to_repls[entry.lemma_key][entry.paired_lemma_key].append((entry.file, entry.row_index))
        repl_to_sources[entry.paired_lemma_key][entry.lemma_key].append((entry.file, entry.row_index))

    output_lines = []

    # 1. Same loanword, different replacements
    same_source_conflicts = {
        lk: repls
        for lk, repls in source_to_repls.items()
        if len(repls) > 1
    }
    if same_source_conflicts:
        output_lines.append("same loanword, different replacements:")
        for lemma_key in sorted(same_source_conflicts):
            output_lines.append(f"  {lemma_key}:")
            for paired_key in sorted(same_source_conflicts[lemma_key]):
                locs = sorted(set(same_source_conflicts[lemma_key][paired_key]))
                locs_str = ", ".join(f"{f}:{r}" for f, r in locs)
                output_lines.append(f"    - {paired_key} ({locs_str})")

    # 2. Same replacement, different sources
    same_repl_conflicts = {
        rk: sources
        for rk, sources in repl_to_sources.items()
        if len(sources) > 1
    }
    if same_repl_conflicts:
        if output_lines:
            output_lines.append("")
        output_lines.append("same replacement, different sources:")
        for repl_key in sorted(same_repl_conflicts):
            output_lines.append(f"  {repl_key}:")
            for source_key in sorted(same_repl_conflicts[repl_key]):
                locs = sorted(set(same_repl_conflicts[repl_key][source_key]))
                locs_str = ", ".join(f"{f}:{r}" for f, r in locs)
                output_lines.append(f"    - {source_key} ({locs_str})")

    if not output_lines:
        return ("No replacement contradictions found.", False)

    return ("\n".join(output_lines), False)


def cmd_multiword(session: AssistantSession, args: List[str]) -> Tuple[str, bool]:
    """Find all spans with multiple tokens for a given span type."""
    if not args:
        return ("Usage: multiword <span_type> (e.g. multiword L)", True)

    span_type = args[0].upper()

    # Group tokens by (file, row_index, tag_id)
    span_tokens: Dict[Tuple[str, int, str], List[str]] = defaultdict(list)
    for token in session.index.tokens:
        if token.prefix == span_type:
            span_tokens[(token.file, token.row_index, token.tag_id)].append(token.surface)

    multiword_spans = {
        key: tokens for key, tokens in span_tokens.items() if len(tokens) > 1
    }

    if not multiword_spans:
        return (f"No multiword spans found for type '{span_type}'.", False)

    output_lines = [f"multiword spans ({span_type}):"]
    for (file_name, row_index, tag_id), tokens in sorted(multiword_spans.items()):
        output_lines.append(
            f"    - {file_name}:{row_index} [{tag_id}]: {' '.join(tokens)}"
        )

    return ("\n".join(output_lines), False)


def cmd_dupes(session: AssistantSession, args: List[str]) -> Tuple[str, bool]:
    """Report clusters of near-duplicate sentences by token-level Jaccard similarity."""
    if not args:
        return ("Usage: dupes <threshold> (e.g. dupes 0.8)", True)

    try:
        threshold = float(args[0])
    except ValueError:
        return (f"Invalid threshold '{args[0]}': must be a float.", True)

    if not (0.0 <= threshold <= 1.0):
        return ("Threshold must be between 0.0 and 1.0.", True)

    # Build row_id -> set of lemma keys
    row_lemmas: Dict[Tuple[str, int], Set[str]] = defaultdict(set)
    for token in session.index.tokens:
        row_lemmas[(token.file, token.row_index)].add(token.lemma_key)

    row_ids = list(row_lemmas.keys())
    n = len(row_ids)

    if n == 0:
        return ("No rows in index.", False)

    # Inverted index: lemma_key -> list of row indices (into row_ids)
    inv: Dict[str, List[int]] = defaultdict(list)
    for i, rid in enumerate(row_ids):
        for lk in row_lemmas[rid]:
            inv[lk].append(i)

    # Candidate pairs sharing >= 1 lemma key
    candidate_pairs: Set[Tuple[int, int]] = set()
    for lk, idxs in inv.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if i > j:
                    i, j = j, i
                candidate_pairs.add((i, j))

    # Union-Find
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    # Similarity cache: (i, j) -> jaccard
    pair_sim: Dict[Tuple[int, int], float] = {}

    for i, j in candidate_pairs:
        a_set = row_lemmas[row_ids[i]]
        b_set = row_lemmas[row_ids[j]]
        sa, sb = len(a_set), len(b_set)
        # Upper-bound prune
        if min(sa, sb) / max(sa, sb) < threshold:
            continue
        intersection = len(a_set & b_set)
        jaccard = intersection / (sa + sb - intersection)
        if jaccard >= threshold:
            pair_sim[(i, j)] = jaccard
            union(i, j)

    # Group by cluster root; only keep clusters with > 1 member
    clusters: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    multi = {root: members for root, members in clusters.items() if len(members) > 1}

    if not multi:
        return (f"No duplicate clusters found at threshold {threshold}.", False)

    # Build output
    output_lines = [f"dupes (threshold: {threshold}):"]
    for cluster_num, (root, members) in enumerate(
        sorted(multi.items(), key=lambda x: row_ids[x[0]]), start=1
    ):
        output_lines.append(f"\ncluster {cluster_num}:")
        anchor = members[0]
        anchor_id = row_ids[anchor]
        anchor_size = len(row_lemmas[anchor_id])
        output_lines.append(
            f"    - {anchor_id[0]}:{anchor_id[1]}  ({anchor_size} tokens)"
        )
        for member in members[1:]:
            mid = row_ids[member]
            msize = len(row_lemmas[mid])
            i, j = (anchor, member) if anchor < member else (member, anchor)
            sim = pair_sim.get((i, j), 0.0)
            output_lines.append(
                f"    - {mid[0]}:{mid[1]}  ({msize} tokens)  similarity: {sim:.2f}"
            )

    return ("\n".join(output_lines), False)


def _parse_freq_args(args: List[str]):
    """Parse arguments for the freq command."""
    file_filter = None
    span_type = None
    by = "both"
    scope = "both"

    i = 0
    while i < len(args):
        if args[i] == "--file" and i + 1 < len(args):
            file_filter = args[i + 1]
            i += 2
        elif args[i] == "--by" and i + 1 < len(args):
            by = args[i + 1].lower()
            if by not in ("surface", "lemma", "both"):
                return None, f"Invalid --by value '{args[i + 1]}'. Use surface, lemma, or both."
            i += 2
        elif args[i] == "--scope" and i + 1 < len(args):
            scope = args[i + 1].lower()
            if scope not in ("word", "span", "both"):
                return None, f"Invalid --scope value '{args[i + 1]}'. Use word, span, or both."
            i += 2
        elif args[i] == "--span" and i + 1 < len(args):
            span_type = args[i + 1].upper()
            i += 2
        else:
            return None, f"Unknown argument '{args[i]}'."

    return {"span_type": span_type, "file": file_filter, "by": by, "scope": scope}, None


def cmd_freq(session: AssistantSession, args: List[str]) -> Tuple[str, bool]:
    """Print frequency lists per span type (word-level and span-level)."""
    params, err = _parse_freq_args(args)
    if err:
        return (err, True)

    span_type = params["span_type"]  # None means all span types
    file_filter = params["file"]
    by = params["by"]
    scope = params["scope"]

    # --file all overrides to no filter; no --file uses active target
    if file_filter is not None:
        if file_filter.lower() == "all":
            file_filter = None
        else:
            matched = next(
                (n for n in session.files if n.lower() == file_filter.lower()), None
            )
            if matched is None:
                available = ", ".join(sorted(session.files.keys()))
                return (f"Unknown file '{file_filter}'. Available: {available}", True)
            file_filter = matched
    else:
        if session.active_target:
            file_filter = session.active_target

    # Determine which span types to report.
    # "o" tokens only exist in index.tokens (no CorpusEntry for unlabeled).
    all_token_prefixes = sorted(set(t.prefix for t in session.index.tokens))
    all_entry_prefixes = sorted(set(e.prefix for e in session.index.entries))
    if span_type is None:
        token_types = all_token_prefixes
        entry_types = all_entry_prefixes
    else:
        token_types = [span_type] if span_type in all_token_prefixes else []
        entry_types = [span_type] if span_type in all_entry_prefixes else []

    def _freq_block(
        tokens: List[TokenEntry],
        entries: List[CorpusEntry],
        label: str,
    ) -> List[str]:
        lines = [f"{label}:"]

        types_for_word = token_types
        types_for_span = entry_types

        if scope in ("word", "both"):
            for st in types_for_word:
                relevant_tokens = [t for t in tokens if t.prefix == st]
                lines.append(f"  [{st}] word:")
                if by in ("surface", "both"):
                    counts: Dict[str, int] = defaultdict(int)
                    for t in relevant_tokens:
                        counts[t.surface] += 1
                    lines.append("    surface:")
                    if counts:
                        for form, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                            lines.append(f"      {cnt:>5}  {form}")
                    else:
                        lines.append("      (none)")
                if by in ("lemma", "both"):
                    counts = defaultdict(int)
                    for t in relevant_tokens:
                        counts[t.lemma_key] += 1
                    lines.append("    lemma:")
                    if counts:
                        for lk, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                            lines.append(f"      {cnt:>5}  {lk}")
                    else:
                        lines.append("      (none)")

        if scope in ("span", "both"):
            for st in types_for_span:
                if st == "o":
                    continue  # no span-level entries for unlabeled tokens
                relevant_entries = [e for e in entries if e.prefix == st]
                lines.append(f"  [{st}] span:")
                if by in ("surface", "both"):
                    counts = defaultdict(int)
                    for e in relevant_entries:
                        counts[e.surface] += 1
                    lines.append("    surface:")
                    if counts:
                        for form, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                            lines.append(f"      {cnt:>5}  {form}")
                    else:
                        lines.append("      (none)")
                if by in ("lemma", "both"):
                    counts = defaultdict(int)
                    for e in relevant_entries:
                        counts[e.lemma_key] += 1
                    lines.append("    lemma:")
                    if counts:
                        for lk, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                            lines.append(f"      {cnt:>5}  {lk}")
                    else:
                        lines.append("      (none)")

        return lines


def cmd_collisions(session: AssistantSession, args: List[str]) -> Tuple[str, bool]:
    """Report hash collisions across all loaded files."""
    output_lines = []
    total_collisions = 0

    for file_name, df in sorted(session.files.items()):
        hashes: Dict[str, List[int]] = defaultdict(list)
        for idx, row in df.iterrows():
            label_sent = str(row.get("Label sentence", ""))
            replacement_sent = str(row.get("Replacement sentence", ""))
            row_hash = compute_row_hash(label_sent, replacement_sent)
            hashes[row_hash].append(idx + 2)

        colliding = {h: rows for h, rows in hashes.items() if len(rows) > 1}
        if colliding:
            count = sum(len(rows) - 1 for rows in colliding.values())
            total_collisions += count
            output_lines.append(f"{file_name}: {count} collision(s)")
            for h, rows in sorted(colliding.items()):
                output_lines.append(f"    hash {h[:12]}...: rows {', '.join(str(r) for r in rows)}")

    if not output_lines:
        return ("No hash collisions found.", False)

    output_lines.append(f"\ntotal: {total_collisions} collision(s) across {len(session.files)} file(s)")
    return ("\n".join(output_lines), False)


def cmd_help(session: AssistantSession, args: List[str]) -> Tuple[str, bool]:
    """List all available commands."""
    commands = [
        ("collisions", "Report hash collisions across all loaded files"),
        ("target <name>", "Set active target file"),
        ("reload", "Reload target file from disk"),
        ("validate-mode <valid|all>", "Set search scope (valid=+ only, or all rows)"),
        ("validate", "Validate rows in target file"),
        ("contradict", "Report labeling contradictions"),
        ("contradict-repl", "Report replacement contradictions"),
        ("multiword <span_type>", "Find spans with multiple tokens"),
        ("dupes <threshold>", "Report near-duplicate sentence clusters"),
        ("freq [--span <type|o>] [--file <name|all>] [--by surface|lemma|both] [--scope word|span|both]",
         "Print frequency lists (all spans by default, defaults to active target)"),
        ("replace", "List available replacement lemma keys"),
        ("stats", "Print aggregate and per-file statistics"),
        ("mode <baseline|extended>", "Change or show active mode"),
        ("output <stdout|path>", "Change output target"),
        ("output-mode <append|replace>", "Set file output mode"),
        ("help", "Show this help message"),
        ("exit / quit", "Exit the assistant"),
    ]

    lines = ["Available commands:"]
    for cmd, desc in commands:
        lines.append(f"  {cmd:<25} {desc}")

    return ("\n".join(lines), False)


# =============================================================================
# Main Entry Point
# =============================================================================

def run_assistant(
    files: List[str],
    language: str,
    mode: str,
    validate_all: bool,
    output: str,
) -> None:
    """Run the interactive assistant session."""

    # 3.1: Validate file paths
    for f in files:
        if not Path(f).exists():
            click.secho(f"Error: File not found: {f}", fg="red")
            return

    # 3.2: Construct Lemmatizer
    try:
        lemmatizer = Lemmatizer(language)
    except ValueError as e:
        click.secho(f"Error: {e}", fg="red")
        return

    # 3.4: Load and validate each Excel file
    session_files = {}
    session_hashes = {}
    file_paths = {}

    for f in files:
        file_name = Path(f).stem
        file_paths[file_name] = f

        results = validate_file(f, mode, validate_all)
        click.echo(f"{file_name}.xlsx: validation done")
        for result in results:
            for err in result.errors:
                prefix = "[warn] " if err.severity == "warning" else "[error] "
                click.echo(f"  row {result.row_index}: {prefix}[{err.rule_id}] {err.field}: {err.message}")

        df = pd.read_excel(f)
        session_files[file_name] = df

        hashes = {}
        for idx, row in df.iterrows():
            label_sent = str(row.get("Label sentence", ""))
            replacement_sent = str(row.get("Replacement sentence", ""))
            row_hash = compute_row_hash(label_sent, replacement_sent)
            hashes[row_hash] = idx + 2

        session_hashes[file_name] = hashes

        collision_count = len(df) - len(hashes)
        if collision_count > 0:
            click.echo(f"  Warning: {collision_count} hash collision(s) detected in {file_name}.xlsx")

    # 3.5: Build corpus index
    lemma_cache: Dict[str, str] = {}
    click.echo("\nBuilding index...")
    index = build_corpus_index(session_files, session_hashes, lemmatizer, mode, validate_all, lemma_cache)

    total_rows = len([e for e in index.entries if e.prefix == "L"])
    total_collisions = sum(
        max(0, len(df) - len(session_hashes[fn]))
        for fn, df in session_files.items()
    )
    collision_msg = f"  ({total_collisions} total hash collision(s))" if total_collisions else ""
    click.echo(f"Index built: {total_rows} labeled spans, {len(index.tokens)} tokens across {len(session_files)} files.{collision_msg}")

    # Create session
    session = AssistantSession(
        files=session_files,
        row_hashes=session_hashes,
        lemmatizer=lemmatizer,
        mode=mode,
        validate_all=validate_all,
        output_target=output,
        output_mode="append",
        active_target=None,
        index=index,
        lemma_cache=lemma_cache,
        _file_paths=file_paths,
    )

    # 3.6: Enter REPL
    click.echo("\nConLoan Assistant (type 'help' for commands)")

    prompt_session = PromptSession(history=InMemoryHistory())

    COMMANDS = {
        "target": cmd_target,
        "reload": cmd_reload,
        "validate-mode": cmd_validate_mode,
        "validate": cmd_validate,
        "contradict": cmd_contradict,
        "collisions": cmd_collisions,
        "freq": cmd_freq,
        "contradict-repl": cmd_contradict_repl,
        "multiword": cmd_multiword,
        "dupes": cmd_dupes,
        "replace": cmd_replace,
        "stats": cmd_stats,
        "mode": cmd_mode,
        "output": cmd_output,
        "output-mode": cmd_output_mode,
        "help": cmd_help,
    }

    while True:
        try:
            target_label = session.active_target or "none"
            prompt_text = f"[{session.mode}|{target_label}]> "
            command = prompt_session.prompt(prompt_text).strip()

            if not command:
                continue

            parts = command.split()
            cmd = parts[0].lower()
            args = parts[1:]

            if cmd in ("exit", "quit"):
                click.echo("Exiting assistant...")
                break

            if cmd not in COMMANDS:
                click.secho(f"Unknown command '{cmd}'. Type 'help' for list.", fg="yellow")
                continue

            result, is_error = COMMANDS[cmd](session, args)

            if is_error:
                click.secho(result, fg="yellow")
                continue

            if session.output_target == "stdout":
                click.echo(result)
            else:
                file_mode = "a" if session.output_mode == "append" else "w"
                with open(session.output_target, file_mode) as fh:
                    fh.write("=" * 60 + "\n")
                    fh.write(result + "\n")
                    fh.write("=" * 60 + "\n")
                click.echo(result)
                click.echo(f"Output written to '{session.output_target}'.")

        except KeyboardInterrupt:
            click.echo("\nUse 'exit' to quit.")
        except EOFError:
            break


@click.command("assistant")
@click.argument("excel_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--language", required=True, help="Canonical language name")
@click.option(
    "--mode",
    type=click.Choice(["baseline", "extended"]),
    default="baseline",
    show_default=True,
)
@click.option("--validate-all", is_flag=True, help="Include all rows in index")
@click.option(
    "--output",
    default="stdout",
    help="Output target (stdout or file path)",
)
def assistant(
    excel_files,
    language,
    mode,
    validate_all,
    output,
):
    """Interactive CLI for loanword annotation management."""
    run_assistant(
        list(excel_files),
        language,
        mode,
        validate_all,
        output,
    )


if __name__ == "__main__":
    assistant()
