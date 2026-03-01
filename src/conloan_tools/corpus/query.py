import subprocess
import math
import os
import tempfile
import sys
import click
from dataclasses import dataclass
from typing import List, Optional

from conloan_tools.corpus import corpus

DEFAULT_CQP_BIN = "cqp"
DEFAULT_RESULTS = 200


@dataclass
class Token:
    word: str
    pos: str
    lemma: str


@dataclass
class CQPResult:
    cqp_id: int
    tokens: List[Token]
    match_index: int

    @property
    def text(self):
        return " ".join([t.word for t in self.tokens])

    @property
    def target_word(self):
        if 0 <= self.match_index < len(self.tokens):
            return self.tokens[self.match_index]
        return None


def parse_cqp_line(line: str) -> Optional[CQPResult]:
    try:
        id_part, content_part = line.split(":", 1)
        cqp_id = int(id_part.strip())
    except ValueError:
        return None

    raw_tokens = content_part.strip().split()
    parsed_tokens = []
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


def count_loanwords_in_sentence(parsed_result, lemma_set_lower):
    if parsed_result is None:
        return set()

    matched_lemmas = set()
    for token in parsed_result.tokens:
        token_lemma = token.lemma.lower()
        if token_lemma in lemma_set_lower:
            matched_lemmas.add(token_lemma)

    return matched_lemmas


def _gaussian(x, mu, sigma):
    return math.exp(-pow(x - mu, 2) / (2 * pow(sigma, 2)))


def _resolve_registry(registry_dir: Optional[str]) -> tuple[str, str]:
    """Return (registry_dir, registry) from explicit dir or CORPUS_REGISTRY."""
    if registry_dir:
        return registry_dir, f"{registry_dir}/registry"

    env = os.environ.get("CORPUS_REGISTRY")
    if env:
        # CORPUS_REGISTRY typically points directly at the registry dir
        registry = env
        registry_dir = os.path.dirname(registry)
        return registry_dir, registry

    click.echo(
        "Error: --cqp-dir not provided and CORPUS_REGISTRY not set.",
        err=True,
    )
    sys.exit(1)


def query_cqp(corpus, query, limit, cqp_bin = None, registry_dir = None):

    if cqp_bin is None:
        cqp_bin = DEFAULT_CQP_BIN
    registry_dir, registry = _resolve_registry(registry_dir)

    commands = [
        f"{corpus};",
        "set Context 1 s;",
        "set PrintMode ascii;",
        "show -pos -lemma;",
        "show +pos +lemma;",
        "set PrintOptions noheader;",
        f"Results = {query};",
    ]

    if limit is not None and limit > 0:
        commands.append(f"reduce Results to {limit};")

    commands.append("cat Results;")

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as tf:
        tf.write("\n".join(commands))
        temp_file_path = tf.name

    try:
        process = subprocess.Popen(
            [cqp_bin, "-r", registry, "-S", "-f", temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=registry_dir,
        )
        stdout, _ = process.communicate()
        return stdout
    except FileNotFoundError:
        print(
            f"Error: Could not find CQP binary at '{cqp_bin}'",
            file=sys.stderr,
        )
        return ""
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def score_sentence(
    res: CQPResult, lemma_set_lower: Optional[dict] = None
) -> float:
    if lemma_set_lower is None:
        lemma_set_lower = {}

    tokens = res.tokens
    s_len = len(tokens)
    if s_len < 5:
        return 0.0

    score_len = _gaussian(s_len, 18, 7)

    matched = count_loanwords_in_sentence(res, lemma_set_lower)
    n_loan = len(matched)

    if n_loan == 0:
        loan_score = 0.0
    elif 1 <= n_loan <= 3:
        loan_score = 1.0
    elif n_loan == 4:
        loan_score = 0.7
    else:
        loan_score = 0.2

    alpha_ratio = sum(1 for t in tokens if t.word.isalnum()) / s_len

    ne_count = sum(
        1 for t in tokens if t.word[0].isupper() and t.pos.startswith("n")
    )
    ne_penalty = 1.0 if ne_count < 3 else 0.4

    return score_len * loan_score * alpha_ratio * ne_penalty


@click.command("query")
@click.argument("corpus")
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
    help="Path to cwb directory. Falls back to CORPUS_REGISTRY env var.",
)
def query(corpus, lemmas, limit, cqp_bin, registry_dir):
    """Query CORPUS for LEMMAS (comma-separated) and rank results."""
    lemma_list = [l.strip() for l in lemmas.split(",") if l.strip()]

    if not lemma_list:
        click.echo("Error: no valid lemmas provided.", err=True)
        sys.exit(1)

    lemma_alt = "|".join(lemma_list)
    search_query = f'[lemma="{lemma_alt}"]'
    click.echo(f"--- Querying {corpus} for: {search_query} ---")

    raw_output = query_cqp(corpus, search_query, limit, cqp_bin, registry_dir)

    results = []
    if raw_output:
        for line in raw_output.split("\n"):
            if not line.strip():
                continue
            parsed = parse_cqp_line(line)
            if parsed:
                results.append(parsed)

    click.echo(f"Parsed {len(results)} results.\n")

    scored_results = []
    for s in results:
        score = score_sentence(s, set(lemma_list))
        scored_results.append((score, s))

    scored_results.sort(key=lambda x: x[0], reverse=True)

    click.echo(f"Top 5 Results for '{lemmas}':")
    click.echo("-" * 60)
    for score, res in scored_results[:5]:
        sent_str = ""
        for i, t in enumerate(res.tokens):
            if i == res.match_index:
                sent_str += f"\033[92m>>{t.word}<<\033[0m "
            else:
                sent_str += f"{t.word} "

        click.echo(f"Score: {score:.4f} | ID: {res.cqp_id}")
        click.echo(sent_str.strip())
        click.echo("-" * 60)


if __name__ == "__main__":
    corpus()
