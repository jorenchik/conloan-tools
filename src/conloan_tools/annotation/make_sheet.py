import csv
import re

import click
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from conloan_tools.corpus.query import CandidateRecord
from conloan_tools.annotation import annotation
from .excel import write_sheet 
import json


def _load_candidates(path: str) -> list[CandidateRecord]:
    import dataclasses
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records.append(CandidateRecord(**d))
    return records


def select_top(
    pool: list[CandidateRecord],
    results: int,
) -> list[CandidateRecord]:
    sorted_pool = sorted(pool, key=lambda r: r.score_total, reverse=True)
    return sorted_pool[:results] if results > 0 else sorted_pool


def get_tagged_lemmas(parsed_result, lemma_set_lower, primary_lemma):
    """Return dict mapping tag_num -> lemma for all loanwords."""
    if parsed_result is None:
        return {1: primary_lemma}

    loanword_positions = []
    for i, token in enumerate(parsed_result.tokens):
        token_lemma = token.lemma.lower()
        if token_lemma in lemma_set_lower:
            orig_lemma = lemma_set_lower[token_lemma]
            is_primary = token_lemma == primary_lemma.lower()
            loanword_positions.append((i, orig_lemma, is_primary))

    if not loanword_positions:
        return {1: primary_lemma}

    loanword_positions.sort(key=lambda x: (not x[2], x[0]))
    return {
        tag_num: lemma
        for tag_num, (_, lemma, _) in enumerate(
            loanword_positions, start=1
        )
    }


def strip_tags(sentence):
    """Remove <L1></L1> etc. tags."""
    return re.sub(r"</?[LN]\d+>", "", sentence)


def normalize_slashes(sentence: str) -> str:
    """Replace '//zx' with '/' in sentence strings."""
    return sentence.replace("//zx", "/") if sentence else sentence


def create_native_template(sentence_with_loan_tags):
    """Replace <L𝑛>word</L𝑛> with <N𝑛>word</N𝑛>."""

    def replace_tag(match):
        tag_num = match.group(1)
        word = match.group(2)
        return f"<N{tag_num}>{word}</N{tag_num}>"

    return re.sub(
        r"<L(\d+)>([^<]+)</L\d+>",
        replace_tag,
        sentence_with_loan_tags,
    )


def build_row(
    rec_word,
    sentence_loan,
    parsed_result,
    lemma_set_lower,
    lemma_to_lang_info,
):
    """Build a single output row dict."""
    if sentence_loan is None:
        sentence_loan = f"<L1>{rec_word}</L1>"
        found = False
    else:
        found = True

    sentence_loan = normalize_slashes(sentence_loan)
    sentence_native = normalize_slashes(create_native_template(sentence_loan))
    tagged_lemmas = get_tagged_lemmas(
        parsed_result, lemma_set_lower, rec_word
    )

    return {
        "Loanword sentence": sentence_loan,
        "Native sentence": sentence_native,
        "Target": "",
        "Valid": "",
        "Suggestions": "",
    }, found


def select_stratified(
    pool: list[CandidateRecord],
    results: int = 0,
    verbose: bool = False,
) -> list[CandidateRecord]:
    """Round-robin selection across NE type strata with equal per-stratum counts."""
    from collections import defaultdict
    
    # Group sentences by their NE type (ORG, LOC, MISC, etc.)
    type_to_sents: dict[str, list[CandidateRecord]] = defaultdict(list)
    for rec in pool:
        tags = getattr(rec, 'tag_map', None)
        if not tags:
            continue
        types_in_record = set(tags.values())
        for t in types_in_record:
            type_to_sents[t].append(rec)
    
    # Sort each stratum by score descending
    for t in type_to_sents:
        type_to_sents[t].sort(key=lambda r: r.score_total, reverse=True)
    
    num_strata = len(type_to_sents)
    max_results = results if results > 0 else float('inf')
    
    # Equal count per stratum
    per_stratum = max_results // num_strata if num_strata else 0
    remainder = max_results % num_strata if num_strata else 0
    
    final: list[CandidateRecord] = []
    sent_used: set[int] = set()
    
    with tqdm(desc="Stratified round-robin", unit="sent") as pbar:
        for t in sorted(type_to_sents.keys()):
            stratum = type_to_sents[t]
            picked = 0
            extra = 1 if remainder > 0 else 0
            remainder -= 1 if extra else 0
            limit = per_stratum + extra
            
            for sent in stratum:
                if picked >= limit:
                    break
                if id(sent) not in sent_used:
                    final.append(sent)
                    sent_used.add(id(sent))
                    picked += 1
                    pbar.update(1)
    
    covered_types = {t for r in final for t in getattr(r, 'tag_map', {}).values()}
    click.echo(
        f"\nStratified summary"
        f"\n  Pool size      : {len(pool)}"
        f"\n  Selected       : {len(final)}"
        f"\n  Strata types   : {num_strata} ({sorted(type_to_sents.keys())})"
        f"\n  Per stratum    : {per_stratum} (+ remainder {max_results % num_strata})"
        f"\n  Types covered  : {len(covered_types)}"
    )
    if verbose:
        click.echo("  Per-type counts (records may appear in multiple types):")
        for t in sorted(type_to_sents.keys()):
            count = sum(1 for r in final if t in getattr(r, 'tag_map', {}).values())
            click.echo(f"    - {t}: {count}")
    
    return final


def select_greedy(
    pool: list[CandidateRecord],
    max_sentences_per_lemma: int = 1,
    verbose: bool = False,
) -> list[CandidateRecord]:
    """Greedy coverage selection with configurable lemma ordering."""
    from heapq import heappush, heappop
    
    # Build lemma → sentences index
    lemma_to_sents: dict[str, list[CandidateRecord]] = defaultdict(list)
    for rec in pool:
        for lemma in rec.matched_lemmas:
            lemma_to_sents[lemma].append(rec)
    
    lemma_usage_count: dict[str, int] = defaultdict(int)
    sent_used: set[int] = set()
    final: list[CandidateRecord] = []
    
    def rebuild_queue():
        queue = []
        for lemma, sents in lemma_to_sents.items():
            if lemma_usage_count[lemma] >= max_sentences_per_lemma:
                continue
            available = sum(1 for s in sents if id(s) not in sent_used)
            if available > 0:
                heappush(queue, (available, lemma))
        return queue
    
    pq = rebuild_queue()
    
    def lemmas_to_try():
        while pq:
            available_count, lemma = heappop(pq)
            if lemma_usage_count[lemma] < max_sentences_per_lemma:
                yield lemma
    
    
    iterations = 0
    
    with tqdm(desc=f"Greedy selection", unit="lem") as pbar:
        for lemma in lemmas_to_try():
            iterations += 1
            
            # Find best available sentence for this lemma
            candidates = lemma_to_sents[lemma]
            best_sent = None
            best_score = -1
            
            for sent in candidates:
                if id(sent) in sent_used:
                    continue
                can_use = all(
                    lemma_usage_count[l] < max_sentences_per_lemma 
                    for l in sent.matched_lemmas
                )
                if can_use and sent.score_total > best_score:
                    best_score = sent.score_total
                    best_sent = sent
            
            if best_sent is None:
                pbar.update(1)
                continue
            
            final.append(best_sent)
            sent_used.add(id(best_sent))
            for l in best_sent.matched_lemmas:
                lemma_usage_count[l] += 1
            
            pbar.update(1)
            pbar.set_postfix(selected=len(final), lemmas_covered=sum(1 for v in lemma_usage_count.values() if v > 0))

    all_lemmas = set(lemma_to_sents.keys())
    covered = sum(1 for v in lemma_usage_count.values() if v > 0)
    saturated = sum(1 for v in lemma_usage_count.values() if v >= max_sentences_per_lemma)
    density_counts = [len(r.matched_lemmas) for r in final]
    avg_density = sum(density_counts) / len(density_counts) if density_counts else 0
    multi_lemma = sum(1 for d in density_counts if d > 1)
    uncovered_lemmas = sorted(all_lemmas - {l for r in final for l in r.matched_lemmas})

    click.echo(
        f"\nGreedy selection summary"
        f"\n  Pool size      : {len(pool)}"
        f"\n  Selected       : {len(final)}"
        f"\n  Iterations     : {iterations}"
        f"\n  Unique lemmas  : {len(all_lemmas)}"
        f"\n  Covered        : {covered}  ({100 * covered / len(all_lemmas):.1f}%)"
        f"\n  Saturated      : {saturated}  (hit {max_sentences_per_lemma}-sentence cap)"
        f"\n  Uncovered      : {len(uncovered_lemmas)}"
        f"\n  Avg density    : {avg_density:.2f} lemmas/sentence"
        f"\n  Multi-lemma    : {multi_lemma} sentences ({100 * multi_lemma / len(final):.1f}%)"
    )
    if verbose and uncovered_lemmas:
        click.echo("  Uncovered lemmas:")
        for lemma in uncovered_lemmas:
            count = len(lemma_to_sents.get(lemma, []))
            click.echo(f"    - {lemma} ({count} candidates in pool)")

    return final


def _record_to_row(rec: CandidateRecord) -> dict:
    row, _ = build_row(
        rec_word=rec.matched_lemmas[0] if rec.matched_lemmas else "",
        sentence_loan=rec.sentence,
        parsed_result=None,
        lemma_set_lower={},
        lemma_to_lang_info={},
    )
    row["Cluster_ID"] = str(rec.cqp_id)
    row["Matched_Lemmas"] = "|".join(rec.matched_lemmas)
    row["Density"] = len(rec.matched_lemmas)
    row["Mode"] = rec.mode
    return row


@click.command("make-sheet")
@click.argument(
    "candidates",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--output", default="conloan_annotation.xlsx", show_default=True)
@click.option(
    "--strategy",
    type=click.Choice(["rarest", "sequential", "stratified"], case_sensitive=False),
    default="rarest",
    show_default=True,
    help="rarest=lemma with fewest candidates first, sequential=in pool order, stratified=round-robin by NE type (ne only)",
)
@click.option(
    "--stream-type",
    type=click.Choice(["lw", "cs", "ne"], case_sensitive=False),
    default="lw",
    show_default=True,
    help="lw=lemma-counting mode, cs/ne=top-by-score mode",
)
@click.option("--max-per-lemma", type=int, default=1, show_default=True)
@click.option("--verbose-stats", is_flag=True, default=False, help="List uncovered lemmas.")
@click.option("--results", type=int, default=0, show_default=True, help="0 = all")
@click.option("--missing-placeholder", is_flag=True, default=False)
@click.option("--ignore-zero-score", is_flag=True, default=False, help="Skip candidates with score_total == 0.0")
@click.option("--sort-input", is_flag=True, default=False, help="Sort pool by score before selection.")
@click.option("--sort-output", is_flag=True, default=False, help="Sort output by score descending.")
def make_sheet(
    candidates,
    output,
    strategy,
    stream_type,
    max_per_lemma,
    results,
    missing_placeholder,
    verbose_stats,
    ignore_zero_score,
    sort_input,
    sort_output,
):
    """Generate annotation sheet from a JSONL candidates file."""
    pool = _load_candidates(candidates)
    if ignore_zero_score:
        before = len(pool)
        pool = [r for r in pool if r.score_total != 0.0]
        click.echo(f"Filtered {before - len(pool)} zero-score candidates. Remaining: {len(pool)}.")
    click.echo(f"Loaded {len(pool)} candidates from {candidates}.")
    
    if sort_input:
        pool = sorted(pool, key=lambda r: r.score_total, reverse=True)
        click.echo("Pool sorted by score descending.")
    
    # Diagnostic: per-lemma candidate counts
    lemma_candidate_counts: dict[str, int] = defaultdict(int)
    for rec in pool:
        for lemma in rec.matched_lemmas:
            lemma_candidate_counts[lemma] += 1
    
    if strategy == "greedy":
        if stream_type in ("cs", "ne"):
            raise click.BadParameter(
                "--strategy=greedy is only valid with --stream-type=lw"
            )
        selected = select_greedy(
            pool,
            max_sentences_per_lemma=max_per_lemma,
            verbose=verbose_stats,
        )
    elif strategy == "stratified":
        if stream_type != "ne":
            raise click.BadParameter(
                "--strategy=stratified is only valid with --stream-type=ne"
            )
        selected = select_stratified(pool, results=results if results > 0 else 0, verbose=verbose_stats)
    else:
        selected = select_top(pool, results=results)

    if sort_output:
        selected = sorted(selected, key=lambda r: r.score_total, reverse=True)

    rows = [_record_to_row(r) for r in selected]

    if missing_placeholder:
        visited = {l for r in selected for l in r.matched_lemmas}
        all_lemmas = {l for r in pool for l in r.matched_lemmas}
        for lemma in sorted(all_lemmas - visited):
            row, _ = build_row(
                rec_word=lemma,
                sentence_loan=None,
                parsed_result=None,
                lemma_set_lower={},
                lemma_to_lang_info={},
            )
            rows.append(row)
        click.echo(f"{len(all_lemmas - visited)} placeholder rows added.")

    write_sheet(rows, output)
    placeholder_count = sum(1 for r in rows if r.get("Cluster_ID") is None)
    real_count = len(rows) - placeholder_count
    click.echo(
        f"\nOutput summary"
        f"\n  File           : {output}"
        f"\n  Total rows     : {len(rows)}"
        f"\n  Real sentences : {real_count}"
        f"\n  Placeholders   : {placeholder_count}"
    )


if __name__ == "__main__":
    annotation()
