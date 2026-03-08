import csv
import re

import click
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from conloan_tools.corpus.query import (
    query_cqp,
    parse_cqp_line,
    score_sentence,
    count_hits_in_sentence,
    load_scoring_config,
    DEFAULT_CQP_BIN,
)
from conloan_tools.annotation import annotation
from .excel import write_sheet 


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


def tag_all_loanwords(parsed_result, lemma_set_lower, primary_lemma):
    """Tag all loanwords in sentence with L1, L2, L3…"""
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

    tokens = []
    for i, t in enumerate(parsed_result.tokens):
        if i in tag_map:
            n = tag_map[i]
            tokens.append(f"<L{n}>{t.word}</L{n}>")
        else:
            tokens.append(t.word)

    return " ".join(tokens)


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


def build_or_query(lemmas):
    """Creates the CQP pipe-separated regex for bulk retrieval."""
    escaped = [re.escape(l) for l in lemmas]
    return f'[lemma="{"|".join(escaped)}"]'


def mine_and_select(
    corpus,
    query_limit,
    cqp_bin,
    registry_dir,
    scoring_config_path,
    lemma_set_lower,
    lemma_to_lang_info,
    max_sentences_per_lemma=1,
    use_scoring=True,
):
    """Bulk-mine a large candidate pool, then greedily select the best
    sentences by density score, respecting per-lemma usage limits."""
    all_target_lemmas = list(lemma_set_lower.values())

    click.echo(
        f"Querying from {corpus} "
        f"({len(all_target_lemmas)} target lemmas, "
        f"limit={query_limit})…"
    )
    raw_output = query_cqp(
        corpus=corpus,
        query=build_or_query(all_target_lemmas),
        limit=query_limit,
        cqp_bin=cqp_bin,
        registry_dir=registry_dir,
    )

    cfg = load_scoring_config(scoring_config_path)
    unique_candidates = {}
    lines = raw_output.split("\n")
    raw_hit_count = 0
    for line in tqdm(lines, desc="Parsing results, computing hit count", unit="line"):
        parsed = parse_cqp_line(line)
        if not parsed:
            continue
        raw_hit_count += 1
        hits = count_hits_in_sentence(parsed, set(lemma_set_lower.keys()))
        score_val = score_sentence(parsed, hits, cfg).score_total if use_scoring else 0
        sent_text = parsed.text
        if sent_text not in unique_candidates:
            unique_candidates[sent_text] = (score_val, parsed)

    click.echo(
        f"Results: {raw_hit_count} → "
        f"{len(unique_candidates)} results after dedup."
    )
    sorted_pool = sorted(
        unique_candidates.values(), key=lambda x: x[0], reverse=True
    )
    click.echo(
        "Results sorted by scoring"
    )

    final_rows = []
    lemma_usage_count = defaultdict(int)
    cluster_id_counter = 0
    skipped = 0

    for score, parsed in tqdm(
        sorted_pool, desc="Greedy selection (mining for samples)", unit="sent"
    ):
        matched_in_sent = {
            t.lemma.lower() for t in parsed.tokens if t.lemma.lower() in lemma_set_lower
        }

        can_use = all(
            lemma_usage_count[l] < max_sentences_per_lemma
            for l in matched_in_sent
        )

        if can_use and matched_in_sent:
            cluster_id = f"cluster_{cluster_id_counter}"
            cluster_id_counter += 1

            primary = list(matched_in_sent)[0]
            output_row, found = build_row(
                rec_word=primary,
                sentence_loan=tag_all_loanwords(
                    parsed, lemma_set_lower, primary
                ),
                parsed_result=parsed,
                lemma_set_lower=lemma_set_lower,
                lemma_to_lang_info=lemma_to_lang_info,
            )

            output_row["Cluster_ID"] = cluster_id
            output_row["Matched_Lemmas"] = "|".join(
                sorted(matched_in_sent)
            )
            output_row["Density"] = len(matched_in_sent)

            final_rows.append(output_row)

            for l in matched_in_sent:
                lemma_usage_count[l] += 1
        else:
            skipped += 1

    covered = sum(
        1 for v in lemma_usage_count.values() if v > 0
    )
    click.echo(
        f"Greedy selection: {len(final_rows)} sentences kept, "
        f"{skipped} skipped. "
        f"Covered {covered}/{len(lemma_set_lower)} target lemmas."
    )

    return final_rows


@click.command("make-sheet")
@click.argument("corpus")
@click.argument(
    "inputs", nargs=-1, required=True, type=click.Path(exists=True)
)
@click.option(
    "--output",
    default="conloan_annotation.xlsx",
    show_default=True,
    help="Output file (.xlsx or .csv)",
)
@click.option(
    "--query-limit",
    type=int,
    default=500000,
    show_default=True,
    help="Corpus pool size",
)
@click.option(
    "--max-per-lemma",
    type=int,
    default=1,
    show_default=True,
    help="Max sentences per loanword",
)
@click.option(
    "--registry-dir",
    default=None,
    help="Path to cwb directory.",
)
@click.option(
    "--scoring-config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="TOML file overriding default scoring parameters.",
)
@click.option("--cqp-bin", default=DEFAULT_CQP_BIN, show_default=True)
@click.option(
    "--scoring/--no-scoring",
    default=True,
    show_default=True,
    help="Enable or disable sentence density scoring.",
)
def make_sheet(
    corpus, inputs, output, query_limit, max_per_lemma, registry_dir, scoring_config, cqp_bin, scoring
):
    """Generate a ConLoan annotation sheet via global corpus mining.

    Run ``inject-suggestions`` afterwards to populate the Suggestions
    column from WordNet.

    Run ``translate`` afterwards to populate the Target using NT.
    """

    input_rows = []
    for file_path in inputs:
        with open(file_path, mode="r", encoding="utf-8") as f:
            input_rows.extend(list(csv.DictReader(f)))

    all_lemmas = [row.get("word", "").strip() for row in input_rows]
    lemma_set_lower = {l.lower(): l for l in all_lemmas if l}
    lemma_to_lang_info = {
        row.get("word", "").strip().lower(): row.get("info", "").strip()
        for row in input_rows
        if row.get("word")
    }

    click.echo(
        f"Loaded {len(lemma_set_lower)} unique lemmas "
        f"from {len(inputs)} input file(s)."
    )
    click.echo(f"Mining {query_limit} sentences using OR strategy…")
    found_rows = mine_and_select(
        corpus=corpus,
        query_limit=query_limit,
        cqp_bin=cqp_bin,
        registry_dir=registry_dir,
        scoring_config_path=scoring_config,
        lemma_set_lower=lemma_set_lower,
        lemma_to_lang_info=lemma_to_lang_info,
        max_sentences_per_lemma=max_per_lemma,
        use_scoring=scoring,
    )

    visited_lemmas = set()
    for row in found_rows:
        for l in row.get("Matched_Lemmas", "").split("|"):
            visited_lemmas.add(l.lower())

    missing_lemmas = [
        l for l in lemma_set_lower if l not in visited_lemmas
    ]
    not_found_rows = []
    for lemma_low in missing_lemmas:
        row, _ = build_row(
            rec_word=lemma_set_lower[lemma_low],
            sentence_loan=None,
            parsed_result=None,
            lemma_set_lower=lemma_set_lower,
            lemma_to_lang_info=lemma_to_lang_info,
        )
        not_found_rows.append(row)

    rows_to_write = found_rows + not_found_rows

    click.echo(
        f"Selected {len(found_rows)} sentences "
        f"covering {len(visited_lemmas)}/{len(lemma_set_lower)} lemmas. "
        f"{len(missing_lemmas)} lemmas not found (placeholder rows added)."
    )
    click.echo(f"Generating {output}…")

    write_sheet(rows_to_write, output)
    click.echo("Success.")


if __name__ == "__main__":
    annotation()
