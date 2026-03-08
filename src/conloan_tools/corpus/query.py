import subprocess
import os
import tempfile
import click
from typing import List, Optional

from conloan_tools.corpus import corpus
from .scoring import (
    Token,
    CQPResult,
    ScoringConfig,
    ScoredResult,
    load_scoring_config,
    score_sentence,
    count_hits_in_sentence,
)

DEFAULT_CQP_BIN = "cqp"
DEFAULT_RESULTS = 200

def scoring_config_option(f):
    return click.option(
        "--scoring-config",
        type=click.Path(exists=True, dir_okay=False),
        default=None,
        help="TOML file overriding default scoring parameters.",
    )(f)

def lemmas_option(f):
    return click.option(
        "--lemmas",
        required=True,
        help="Comma-separated lemma list for loanword density scoring.",
    )(f)

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

def query_cqp(
    corpus: str,
    query: str,
    limit: int,
    cqp_bin: Optional[str] = None,
    registry_dir: Optional[str] = None,
) -> str:
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

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".cqp"
    ) as tf:
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
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise click.ClickException(
                f"CQP exited with code {process.returncode}:\n{stderr}"
            )
        if stderr.strip():
            click.echo(f"CQP stderr: {stderr.strip()}", err=True)
        return stdout
    except FileNotFoundError:
        raise click.ClickException(
            f"Could not find CQP binary at '{cqp_bin}'"
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def _parse_lemma_arg(lemmas: str) -> List[str]:
    parsed = [l.strip() for l in lemmas.split(",") if l.strip()]
    if not parsed:
        raise click.UsageError("No valid lemmas provided.")
    return parsed

def _run_query_and_score(
    corpus_name: str,
    lemmas: str,
    limit: int,
    cqp_bin: str,
    registry_dir: Optional[str],
    cfg: ScoringConfig,
) -> List[ScoredResult]:
    lemma_list = _parse_lemma_arg(lemmas)
    lemma_alt = "|".join(lemma_list)
    search_query = f'[lemma="{lemma_alt}"]'
    click.echo(f"--- Querying {corpus_name} for: {search_query} ---")

    raw_output = query_cqp(
        corpus_name, search_query, limit, cqp_bin, registry_dir
    )
    lemma_set = {l.lower() for l in lemma_list}
    scored: List[ScoredResult] = []

    if raw_output:
        for line in raw_output.split("\n"):
            if not line.strip():
                continue
            parsed = parse_cqp_line(line)
            if parsed:
                hits = count_hits_in_sentence(parsed, lemma_set)
                scored.append(score_sentence(parsed, hits, cfg))

    scored.sort(key=lambda r: r.score_total, reverse=True)
    click.echo(f"Parsed and scored {len(scored)} results.\n")
    return scored

@click.command("query")
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
    help="Path to cwb directory. Falls back to CORPUS_REGISTRY env var.",
)
@scoring_config_option
def query(
    corpus_name, lemmas, limit, cqp_bin, registry_dir, scoring_config
):
    """Query CORPUS_NAME for LEMMAS (comma-separated)."""
    cfg = load_scoring_config(scoring_config)
    scored = _run_query_and_score(
        corpus_name, lemmas, limit, cqp_bin, registry_dir, cfg
    )

    click.echo(f"Top 5 Results for '{lemmas}':")
    click.echo("-" * 60)
    for r in scored[:5]:
        click.echo(
            f"Score: {r.score_total:.4f}  "
            f"(len={r.score_length:.2f}  hits={r.score_hit_density:.2f}"
            f"  clean={r.score_cleanliness:.2f}  ne={r.score_ne_penalty:.2f})"
            f"  | ID: {r.cqp_id}"
            f"  {'[FILTERED: ' + r.filter_reason + ']' if r.filtered else ''}"
        )
        click.echo(r.text)
        click.echo("-" * 60)

if __name__ == "__main__":
    corpus()
