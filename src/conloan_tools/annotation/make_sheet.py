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

    sentence_native = create_native_template(sentence_loan)
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


def select_greedy(
    pool: list[CandidateRecord],
    max_sentences_per_lemma: int = 1,
) -> list[CandidateRecord]:
    """Greedy coverage selection over a pre-fetched pool."""
    lemma_usage_count: dict[str, int] = defaultdict(int)
    final: list[CandidateRecord] = []
    skipped = 0

    for rec in tqdm(pool, desc="Greedy selection", unit="sent"):
        matched = rec.matched_lemmas
        if not matched:
            skipped += 1
            continue

        can_use = all(
            lemma_usage_count[l] < max_sentences_per_lemma for l in matched
        )
        if can_use:
            final.append(rec)
            for l in matched:
                lemma_usage_count[l] += 1
        else:
            skipped += 1

    covered = sum(1 for v in lemma_usage_count.values() if v > 0)
    click.echo(
        f"Greedy: {len(final)} kept, {skipped} skipped. "
        f"Covered {covered} unique lemmas."
    )
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
    type=click.Choice(["greedy", "top"]),
    default="greedy",
    show_default=True,
)
@click.option("--max-per-lemma", type=int, default=1, show_default=True)
@click.option("--results", type=int, default=0, show_default=True, help="0 = all")
@click.option("--missing-placeholder", is_flag=True, default=False)
def make_sheet(candidates, output, strategy, max_per_lemma, results, missing_placeholder):
    """Generate annotation sheet from a JSONL candidates file."""
    pool = _load_candidates(candidates)
    click.echo(f"Loaded {len(pool)} candidates from {candidates}.")

    if strategy == "greedy":
        selected = select_greedy(pool, max_sentences_per_lemma=max_per_lemma)
    else:
        selected = select_top(pool, results=results)

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
    click.echo(f"Written {len(rows)} rows to {output}.")


if __name__ == "__main__":
    annotation()
