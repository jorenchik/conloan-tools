"""assistant - Interactive CLI for loanword annotation management."""

import re
import hashlib
import click
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Set, Optional, Dict, Tuple
from collections import defaultdict
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from conloan_tools.stz.lemmatize import Lemmatizer
from conloan_tools.wordnet.query import WordNet

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
    wordnet: Optional[WordNet]
    mode: str
    validate_all: bool
    warn_undecided: bool
    output_target: str                        # "stdout" or file path
    output_mode: str                           # "append" or "replace"
    active_target: Optional[str]              # name-no-ext
    index: CorpusIndex
    watch_queue: List[str]
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


def build_corpus_index(
    files: Dict[str, pd.DataFrame],
    row_hashes: Dict[str, Dict[str, int]],
    lemmatizer: Lemmatizer,
    mode: str,
    validate_all: bool
) -> CorpusIndex:
    """Build the complete corpus index from all loaded files."""
    index = CorpusIndex()
    
    for file_name, df in files.items():
        for idx, row in df.iterrows():
            row_valid = str(row.get("Valid", "")).strip().lower()
            if not validate_all and row_valid != "+":
                continue
            
            row_index = idx + 2  # 1-based Excel row number
            label_sent = str(row.get("Label sentence", ""))
            replacement_sent = str(row.get("Replacement sentence", ""))
            
            # Process Label sentence tokens
            _process_sentence(
                index, file_name, row_index, label_sent,
                replacement_sent, lemmatizer, is_label=True
            )
            
            # Process Replacement sentence for paired info
            _process_paired_sentence(
                index, file_name, row_index, replacement_sent,
                label_sent, lemmatizer
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
) -> None:
    """Process a sentence to extract tokens and spans."""
    # Build a plain-text-position → (prefix, tag_id) map so that every
    # character inside a span — including multi-word spans — is labelled.
    char_label_map = _build_char_label_map(sentence)

    plain_text = strip_tags(sentence)

    # Tokenize using Stanza
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
    lemmatizer: Lemmatizer
) -> None:
    """Process paired sentence to build corpus entries."""
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
        # Lemmatize the span content
        span_lemmas = lemmatizer.lemmatize(content)
        lemma_key = " ".join(span_lemmas).lower()
        
        # Find paired tag
        paired_tag_id = None
        paired_content = ""
        paired_lemma_key = ""
        
        if prefix == "L" and f"N{tag_id[1:]}" in paired_spans:
            paired_tag_id = f"N{tag_id[1:]}"
            paired_content = paired_spans[paired_tag_id][1]
            paired_lemmas = lemmatizer.lemmatize(paired_content)
            paired_lemma_key = " ".join(paired_lemmas).lower()
        elif prefix == "N" and f"L{tag_id[1:]}" in paired_spans:
            paired_tag_id = f"L{tag_id[1:]}"
            paired_content = paired_spans[paired_tag_id][1]
            paired_lemmas = lemmatizer.lemmatize(paired_content)
            paired_lemma_key = " ".join(paired_lemmas).lower()
        elif prefix == "CS" and f"CN{tag_id[2:]}" in paired_spans:
            paired_tag_id = f"CN{tag_id[2:]}"
            paired_content = paired_spans[paired_tag_id][1]
            paired_lemmas = lemmatizer.lemmatize(paired_content)
            paired_lemma_key = " ".join(paired_lemmas).lower()
        elif prefix == "CN" and f"CS{tag_id[2:]}" in paired_spans:
            paired_tag_id = f"CS{tag_id[2:]}"
            paired_content = paired_spans[paired_tag_id][1]
            paired_lemmas = lemmatizer.lemmatize(paired_content)
            paired_lemma_key = " ".join(paired_lemmas).lower()
        elif prefix == "NE":
            # NE is identity pair
            if tag_id in paired_spans:
                paired_tag_id = tag_id
                paired_content = paired_spans[tag_id][1]
                paired_lemmas = lemmatizer.lemmatize(paired_content)
                paired_lemma_key = " ".join(paired_lemmas).lower()
        
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
# File Watching
# =============================================================================

def process_watch_queue(session: AssistantSession) -> None:
    """Process any pending file reload events."""
    while session.watch_queue:
        event = session.watch_queue.pop(0)
        if event.startswith("RELOAD:"):
            file_name = event.split(":", 1)[1]
            _reload_file(session, file_name)


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

        for idx, row in new_df.iterrows():
            row_index = idx + 2
            row_valid = str(row.get("Valid", "")).strip().lower()
            if not session.validate_all and row_valid != "+":
                continue

            label_sent = str(row.get("Label sentence", ""))
            replacement_sent = str(row.get("Replacement sentence", ""))

            _process_sentence(
                session.index, file_name, row_index, label_sent,
                replacement_sent, session.lemmatizer, is_label=True
            )
            _process_paired_sentence(
                session.index, file_name, row_index, replacement_sent,
                label_sent, session.lemmatizer
            )

        summary = (
            f"[reload] {file_name}.xlsx: "
            f"+{added_count} added, -{removed_count} removed rows."
        )
        session.watch_queue.append(summary)
        
    except Exception as e:
        session.watch_queue.append(f"[watch] Error reloading {file_name}: {str(e)}")


# =============================================================================
# REPL Commands
# =============================================================================

def cmd_target(session: AssistantSession, args: List[str]) -> str:
    """Set or show active target file."""
    if not args:
        if session.active_target:
            return f"Current target: '{session.active_target}'"
        return "No target set."
    
    target_name = args[0].lower()
    matched = None
    for name in session.files.keys():
        if name.lower() == target_name:
            matched = name
            break
    
    if not matched:
        available = ", ".join(sorted(session.files.keys()))
        return f"Unknown target '{args[0]}'. Available: {available}"
    
    session.active_target = matched
    return f"Target set to '{matched}'."


def cmd_validate(session: AssistantSession, args: List[str]) -> str:
    """Validate rows in target file."""
    if not session.active_target:
        return "No target selected. Use 'target <name>' first."
    
    file_name = session.active_target
    df = session.files[file_name]
    
    # Parse row argument
    row_range = None
    if args:
        row_arg = args[0]
        if '-' in row_arg:
            parts = row_arg.split('-')
            try:
                start = int(parts[0])
                end = int(parts[1])
                row_range = (start, end)
            except ValueError:
                return f"Invalid row range: {row_arg}"
        else:
            try:
                row_num = int(row_arg)
                row_range = (row_num, row_num)
            except ValueError:
                return f"Invalid row number: {row_arg}"
    
    results = []
    for idx, row in df.iterrows():
        row_index = idx + 2
        if row_range:
            if row_index < row_range[0] or row_index > row_range[1]:
                continue
        
        if not session.validate_all:
            row_valid = str(row.get("Valid", "")).strip().lower()
            if row_valid != "+":
                continue
        
        errors = validate_row(row, session.mode, session.warn_undecided)
        if errors:
            results.append(RowResult(row_index=row_index, errors=errors))
    
    if not results:
        return "No issues found."
    
    output_lines = []
    for result in results:
        output_lines.append(f"{result.row_index}:")
        for err in result.errors:
            prefix = "[warn]" if err.severity == "warning" else ""
            output_lines.append(f"    - {prefix}[{err.rule_id}] {err.field}: {err.message}")
    
    return "\n".join(output_lines)


def cmd_contradict(session: AssistantSession, args: List[str]) -> str:
    """Report labeling contradictions."""
    if not session.active_target:
        return "No target selected. Use 'target <name>' first."
    
    file_name = session.active_target
    
    # Parse row argument to determine starting point rows
    target_rows = None
    if args:
        row_arg = args[0]
        if '-' in row_arg:
            parts = row_arg.split('-')
            try:
                start = int(parts[0])
                end = int(parts[1])
                target_rows = set(range(start, end + 1))
            except ValueError:
                return f"Invalid row range: {row_arg}"
        else:
            try:
                row_num = int(row_arg)
                target_rows = {row_num}
            except ValueError:
                return f"Invalid row number: {row_arg}"
    
    # Collect tokens with their lemma_key and label status
    # lemma_key -> list of (prefix, surface, row_index)
    token_occurrences = defaultdict(list)
    starting_point_lemmas = set()
    
    for token in session.index.tokens:
        if token.file != file_name:
            continue
        if target_rows and token.row_index not in target_rows:
            continue
        
        token_occurrences[token.lemma_key].append(
            (token.prefix, token.surface, token.row_index)
        )
        if target_rows is None or token.row_index in target_rows:
            starting_point_lemmas.add(token.lemma_key)
    
    # Find contradictions: same lemma appears with different labels (L/CR/o)
    contradictions = []
    
    for lemma_key in starting_point_lemmas:
        occurrences = token_occurrences.get(lemma_key, [])
        if not occurrences:
            continue
        
        prefixes_in_use = set()
        for prefix, surface, row_index in occurrences:
            prefixes_in_use.add(prefix)
        
        # Contradiction: same lemma has both labeled (L/CR) and unlabeled (o) tokens
        # OR same lemma has multiple different label prefixes
        has_labeled = any(p not in ('o',) for p in prefixes_in_use)
        has_unlabeled = 'o' in prefixes_in_use
        
        if (has_labeled and has_unlabeled) or len(prefixes_in_use) > 1:
            contradiction = {
                'lemma_key': lemma_key,
                'labeling': []
            }
            for prefix, surface, row_index in occurrences:
                tag_id = prefix if prefix == 'o' else f"{prefix}"
                contradiction['labeling'].append((row_index, tag_id, surface))
            contradictions.append(contradiction)
    
    if not contradictions:
        return "No contradictions found."
    
    output_lines = []
    for c in sorted(contradictions, key=lambda x: x['lemma_key']):
        output_lines.append(f"{c['lemma_key']}:")
        output_lines.append("    [labeling]")
        for row_index, tag_id, surface in sorted(c['labeling'], key=lambda x: x[0]):
            output_lines.append(f"    - Row {row_index}: {tag_id}, {surface}")
    
    return "\n".join(output_lines)


def cmd_replace(session: AssistantSession, args: List[str]) -> str:
    """List available replacement lemma keys."""
    if not session.active_target:
        return "No target selected. Use 'target <name>' first."
    
    file_name = session.active_target
    
    # Determine which rows to process
    target_rows = None
    if args:
        row_arg = args[0]
        if '-' in row_arg:
            parts = row_arg.split('-')
            try:
                start = int(parts[0])
                end = int(parts[1])
                target_rows = set(range(start, end + 1))
            except ValueError:
                return f"Invalid row range: {row_arg}"
        else:
            try:
                row_num = int(row_arg)
                target_rows = {row_num}
            except ValueError:
                return f"Invalid row number: {row_arg}"
    
    # Collect lemma keys and their replacements
    lemma_replacements = defaultdict(lambda: defaultdict(set))
    
    for entry in session.index.entries:
        if entry.file != file_name:
            continue
        if target_rows and entry.row_index not in target_rows:
            continue
        if entry.prefix != "L":
            continue

        # Sheet source: span-level paired lemma key
        lemma_replacements[entry.lemma_key]['sheet'].add(entry.paired_lemma_key)

    # WordNet source: per constituent word of the span lemma key
    if session.wordnet:
        for lemma_key in lemma_replacements:
            words = lemma_key.split()
            for word in words:
                wn_result = session.wordnet.get_synonym_groups(word)
                if wn_result.found:
                    for entry_match in wn_result.entries:
                        for sense in entry_match.senses:
                            for syn in sense.synonyms:
                                syn_lemma = session.lemmatizer.lemmatize_word(syn)
                                lemma_replacements[lemma_key].setdefault(
                                    f'wordnet:{word}', set()
                                ).add(syn_lemma.lower())

    if not lemma_replacements:
        return "No labeled spans found in specified rows."

    output_lines = []
    for lemma_key in sorted(lemma_replacements.keys()):
        sources = lemma_replacements[lemma_key]
        output_lines.append(f"{lemma_key}:")

        has_replacements = False
        for src_name, repl_keys in sources.items():
            if not repl_keys:
                continue
            has_replacements = True
            if src_name == 'sheet':
                for repl_key in sorted(repl_keys):
                    output_lines.append(f"    - {repl_key} ({file_name})")
            else:
                # src_name is 'wordnet:<word>'
                constituent = src_name.split(':', 1)[1]
                for repl_key in sorted(repl_keys):
                    output_lines.append(
                        f"    - [{constituent}] {repl_key} (wordnet)"
                    )

        if not has_replacements:
            output_lines.append("    (no replacements found)")

    return "\n".join(output_lines)


def cmd_stats(session: AssistantSession, args: List[str]) -> str:
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
    
    # Sentence counts (unique file+row combinations)
    total_sentences = len(set((e.file, e.row_index) for e in session.index.entries))
    validated_sentences = 0
    for (file_name, row_index) in set((e.file, e.row_index) for e in session.index.entries):
        df = session.files.get(file_name)
        if df is not None:
            idx = row_index - 2
            if idx >= 0 and idx < len(df):
                row_valid = str(df.iloc[idx].get("Valid", "")).strip().lower()
                if row_valid == "+":
                    validated_sentences += 1
    
    # Duplicate lemma variants for labeled words (lemmas appearing more than once in label_occurrences)
    lemma_counts = defaultdict(int)
    for file_name, df in session.files.items():
        for idx, row in df.iterrows():
            label_sent = str(row.get("Label sentence", ""))
            spans = get_tag_spans(label_sent)
            for prefix, content, start, end in spans:
                lemmas = session.lemmatizer.lemmatize(content)
                lemma_key = " ".join(lemmas).lower()
                lemma_counts[lemma_key] += 1
    duplicate_lemma_variants = sum(1 for c in lemma_counts.values() if c > 1)
    
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
        
        f_sentences = len(set(e.row_index for e in f_entries))
        f_df = session.files[file_name]
        f_validated = 0
        for row_idx in set(e.row_index for e in f_entries):
            idx = row_idx - 2
            if 0 <= idx < len(f_df):
                row_valid = str(f_df.iloc[idx].get("Valid", "")).strip().lower()
                if row_valid == "+":
                    f_validated += 1
        
        f_labeled_lemmas = [e.lemma_key for e in f_entries]
        f_lemma_counts = {}
        for lemma in f_labeled_lemmas:
            f_lemma_counts[lemma] = f_lemma_counts.get(lemma, 0) + 1
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
    
    return "\n".join(lines)


def cmd_mode(session: AssistantSession, args: List[str]) -> str:
    """Change or show the active mode."""
    if not args:
        return f"Current mode: {session.mode}"
    
    new_mode = args[0].lower()
    if new_mode not in ("baseline", "extended"):
        return f"Invalid mode '{new_mode}'. Use 'baseline' or 'extended'."
    
    if new_mode == session.mode:
        return f"Mode already set to '{new_mode}'."
    
    session.mode = new_mode
    
    # Rebuild corpus index
    total_rows = 0
    for df in session.files.values():
        for idx, row in df.iterrows():
            row_valid = str(row.get("Valid", "")).strip().lower()
            if not session.validate_all and row_valid != "+":
                continue
            total_rows += 1
    
    session.index = build_corpus_index(
        session.files,
        session.row_hashes,
        session.lemmatizer,
        session.mode,
        session.validate_all
    )
    
    return f"Mode set to '{new_mode}'. Rebuilding index...\nIndex rebuilt: {total_rows} rows processed across {len(session.files)} files."


def cmd_output(session: AssistantSession, args: List[str]) -> str:
    """Change output target."""
    if not args:
        return f"Current output: {session.output_target}"
    
    new_target = args[0]
    if new_target != "stdout" and not new_target.endswith('.txt'):
        return "Output target must be 'stdout' or a .txt file path."
    
    session.output_target = new_target
    
    return f"Output set to '{new_target}'."


def cmd_output_mode(session: AssistantSession, args: List[str]) -> str:
    """Change or show output mode (append or replace)."""
    if not args:
        return f"Current output-mode: {session.output_mode}"
    
    new_mode = args[0].lower()
    if new_mode not in ("append", "replace"):
        return f"Invalid output-mode '{new_mode}'. Use 'append' or 'replace'."
    
    if new_mode == session.output_mode:
        return f"Output-mode already set to '{new_mode}'."
    
    session.output_mode = new_mode
    return f"Output-mode set to '{new_mode}'."


def cmd_reload(session: AssistantSession, args: List[str]) -> str:
    """Reload the target file from disk."""
    if not session.active_target:
        return "No target selected. Use 'target <name>' first."
    
    file_name = session.active_target
    _reload_file(session, file_name)
    
    # Build change summary from queue
    summary_lines = []
    while session.watch_queue:
        summary_lines.append(session.watch_queue.pop(0))
    
    if summary_lines:
        return "\n".join(summary_lines)
    return f"{file_name} reloaded."


def cmd_validate_mode(session: AssistantSession, args: List[str]) -> str:
    """Change or show the validate mode (which rows are included in search)."""
    if not args:
        mode = "all" if session.validate_all else "valid"
        return f"Current validate-mode: {mode}"
    
    new_mode = args[0].lower()
    if new_mode not in ("valid", "all"):
        return f"Invalid validate-mode '{new_mode}'. Use 'valid' or 'all'."
    
    if new_mode == "valid":
        if not session.validate_all:
            return "Validate-mode already set to 'valid'."
        session.validate_all = False
    else:
        if session.validate_all:
            return "Validate-mode already set to 'all'."
        session.validate_all = True
    
    # Rebuild index with new filter
    total_rows = 0
    for df in session.files.values():
        for idx, row in df.iterrows():
            row_valid = str(row.get("Valid", "")).strip().lower()
            if not session.validate_all and row_valid != "+":
                continue
            total_rows += 1
    
    session.index = build_corpus_index(
        session.files,
        session.row_hashes,
        session.lemmatizer,
        session.mode,
        session.validate_all
    )
    
    return f"Validate-mode set to '{new_mode}'. Index rebuilt: {total_rows} rows."


def cmd_contradict_repl(session: AssistantSession, args: List[str]) -> str:
    """Report replacement contradictions."""
    if not session.active_target:
        return "No target selected. Use 'target <name>' first."

    file_name = session.active_target

    target_rows = None
    if args:
        row_arg = args[0]
        if '-' in row_arg:
            parts = row_arg.split('-')
            try:
                start = int(parts[0])
                end = int(parts[1])
                target_rows = set(range(start, end + 1))
            except ValueError:
                return f"Invalid row range: {row_arg}"
        else:
            try:
                row_num = int(row_arg)
                target_rows = {row_num}
            except ValueError:
                return f"Invalid row number: {row_arg}"

    # Collect L-prefix entries for the target file/rows
    # lemma_key -> {paired_lemma_key -> [row_index, ...]}
    source_to_repls: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    # paired_lemma_key -> {lemma_key -> [row_index, ...]}
    repl_to_sources: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

    for entry in session.index.entries:
        if entry.file != file_name:
            continue
        if entry.prefix != "L":
            continue
        if target_rows and entry.row_index not in target_rows:
            continue
        if not entry.paired_lemma_key:
            continue

        source_to_repls[entry.lemma_key][entry.paired_lemma_key].append(entry.row_index)
        repl_to_sources[entry.paired_lemma_key][entry.lemma_key].append(entry.row_index)

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
                rows = sorted(set(same_source_conflicts[lemma_key][paired_key]))
                rows_str = ", ".join(str(r) for r in rows)
                output_lines.append(f"    - {paired_key} (rows: {rows_str})")

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
                rows = sorted(set(same_repl_conflicts[repl_key][source_key]))
                rows_str = ", ".join(str(r) for r in rows)
                output_lines.append(f"    - {source_key} (rows: {rows_str})")

    if not output_lines:
        return "No replacement contradictions found."

    return "\n".join(output_lines)


def cmd_help(session: AssistantSession, args: List[str]) -> str:
    """List all available commands."""
    commands = [
        ("target <name>", "Set active target file"),
        ("reload", "Reload target file from disk"),
        ("validate-mode <valid|all>", "Set search scope (valid=+ only, or all rows)"),
        ("validate <row?>", "Validate rows in target file"),
        ("contradict <row?>", "Report labeling contradictions"),
        ("contradict-repl <row?>", "Report replacement contradictions"),
        ("replace <row?>", "List available replacement lemma keys"),
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
    
    return "\n".join(lines)


# =============================================================================
# Main Entry Point
# =============================================================================

def run_assistant(
    files: List[str],
    language: str,
    wordnet_path: Optional[str],
    mode: str,
    validate_all: bool,
    warn_undecided: bool,
    output: str
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
    
    # 3.3: Load WordNet if provided
    wordnet = None
    if wordnet_path:
        try:
            wordnet = WordNet(wordnet_path)
        except Exception as e:
            click.secho(f"Error loading WordNet: {e}", fg="red")
            return
    
    # 3.4: Load and validate each Excel file
    session_files = {}
    session_hashes = {}
    file_paths = {}
    
    for f in files:
        file_name = Path(f).stem  # name without extension
        file_paths[file_name] = f
        
        # Validate file
        results = validate_file(f, mode, validate_all, warn_undecided)
        
        error_count = sum(1 for r in results if any(e.severity == "error" for e in r.errors))
        warn_count = sum(1 for r in results if all(e.severity == "warning" for e in r.errors))
        
        click.echo(f"{file_name}.xlsx: {len(results)} issues ({error_count} errors, {warn_count} warnings)")
        
        # Load DataFrame
        df = pd.read_excel(f)
        session_files[file_name] = df
        
        # Compute hashes
        hashes = {}
        for idx, row in df.iterrows():
            label_sent = str(row.get("Label sentence", ""))
            replacement_sent = str(row.get("Replacement sentence", ""))
            row_hash = compute_row_hash(label_sent, replacement_sent)
            # Collision policy: last-occurring wins
            hashes[row_hash] = idx + 2
        
        session_hashes[file_name] = hashes
        
        # Check for hash collisions
        if len(hashes) != len(df):
            click.echo(f"  Warning: Hash collision detected in {file_name}.xlsx")
    
    # 3.5: Build corpus index
    click.echo(f"\nBuilding index...")
    index = build_corpus_index(session_files, session_hashes, lemmatizer, mode, validate_all)
    
    total_rows = len([e for e in index.entries if e.prefix == "L"])
    click.echo(f"Index built: {total_rows} labeled spans, {len(index.tokens)} tokens across {len(session_files)} files.")
    
    # Create session
    session = AssistantSession(
        files=session_files,
        row_hashes=session_hashes,
        lemmatizer=lemmatizer,
        wordnet=wordnet,
        mode=mode,
        validate_all=validate_all,
        warn_undecided=warn_undecided,
        output_target=output,
        output_mode="append",
        active_target=None,
        index=index,
        watch_queue=[],
        _file_paths=file_paths
    )
    
    # 3.7: Enter REPL
    click.echo("\nConLoan Assistant (type 'help' for commands)")
    
    output_lines = []
    
    while True:
        try:
            # Flush watch queue before prompt
            if session.watch_queue:
                for line in session.watch_queue:
                    output_lines.append(line)
                session.watch_queue.clear()
            
            # Show prompt
            command = input("> ").strip()
            
            if not command:
                continue
            
            # Parse command
            parts = command.split()
            cmd = parts[0].lower()
            args = parts[1:]
            
            # Process watch queue first
            if session.watch_queue:
                for line in session.watch_queue:
                    output_lines.append(line)
                session.watch_queue.clear()
            
            # Execute command
            if cmd in ("exit", "quit"):
                output_lines.append("Exiting assistant...")
                break
            elif cmd == "target":
                result = cmd_target(session, args)
                output_lines.append(result)
            elif cmd == "reload":
                result = cmd_reload(session, args)
                output_lines.append(result)
            elif cmd == "validate-mode":
                result = cmd_validate_mode(session, args)
                output_lines.append(result)
            elif cmd == "validate":
                result = cmd_validate(session, args)
                output_lines.append(result)
            elif cmd == "contradict":
                result = cmd_contradict(session, args)
                output_lines.append(result)
            elif cmd == "contradict-repl":
                result = cmd_contradict_repl(session, args)
                output_lines.append(result)
            elif cmd == "replace":
                result = cmd_replace(session, args)
                output_lines.append(result)
            elif cmd == "stats":
                result = cmd_stats(session, args)
                output_lines.append(result)
            elif cmd == "mode":
                result = cmd_mode(session, args)
                output_lines.append(result)
            elif cmd == "output":
                result = cmd_output(session, args)
                output_lines.append(result)
            elif cmd == "output-mode":
                result = cmd_output_mode(session, args)
                output_lines.append(result)
            elif cmd == "help":
                result = cmd_help(session, args)
                output_lines.append(result)
            else:
                output_lines.append(f"Unknown command '{cmd}'. Type 'help' for list.")
            
            # Flush output
            if session.output_target == "stdout":
                for line in output_lines:
                    click.echo(line)
            else:
                mode = "a" if session.output_mode == "append" else "w"
                with open(session.output_target, mode) as f:
                    if output_lines:
                        f.write("=" * 60 + "\n")
                        f.write("\n".join(output_lines) + "\n")
                        f.write("=" * 60 + "\n")
                if output_lines:
                    click.echo(f"Output written to '{session.output_target}'.")
            output_lines.clear()
            
        except KeyboardInterrupt:
            click.echo("\nUse 'exit' to quit.")
        except EOFError:
            break


@click.command("assistant")
@click.argument("excel_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--language", required=True, help="Canonical language name")
@click.option("--wordnet", default=None, help="WordNet LMF XML file path")
@click.option(
    "--mode",
    type=click.Choice(["baseline", "extended"]),
    default="baseline",
    show_default=True,
)
@click.option("--validate-all", is_flag=True, help="Include all rows in index")
@click.option(
    "--warn-undecided/--no-warn-undecided",
    default=True,
)
@click.option(
    "--output",
    default="stdout",
    help="Output target (stdout or file path)",
)
def assistant(
    excel_files,
    language,
    wordnet,
    mode,
    validate_all,
    warn_undecided,
    output
):
    """Interactive CLI for loanword annotation management."""
    run_assistant(
        list(excel_files),
        language,
        wordnet,
        mode,
        validate_all,
        warn_undecided,
        output
    )


if __name__ == "__main__":
    assistant()
