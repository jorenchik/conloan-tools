import subprocess
import math
import os
import tempfile
import sys
import click
from dataclasses import dataclass, asdict
from typing import List, Optional, Set

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from conloan_tools.corpus import corpus

DEFAULT_CQP_BIN = "cqp"
DEFAULT_RESULTS = 200


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


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
    def text(self) -> str:
        return " ".join(t.word for t in self.tokens)

    @property
    def target_word(self) -> Optional[Token]:
        if 0 <= self.match_index < len(self.tokens):
            return self.tokens[self.match_index]
        return None


@dataclass
class ScoringConfig:

    # Hard filters
    min_tokens:       int   = 5
    min_alpha_ratio:  float = 0.5
    filter_zero_hits: bool  = True

    # Length gaussian
    length_mu:    float = 18.0
    length_sigma: float = 7.0

    # Hit density - Gaussian peak around ideal
    # Favors sentences where the loanword isn't alone, but isn't crowded
    hit_mu:    float = 2.0
    hit_sigma: float = 1.5

    # NE penalty - continuous decay beyond ne_free free NEs
    ne_free:  int   = 2
    ne_decay: float = 0.3

    # Component weights (normalised at scoring time)
    weight_length: float = 0.20
    weight_hit:    float = 0.20
    weight_clean:  float = 0.40
    weight_ne:     float = 0.20


@dataclass
class ScoredResult:
    cqp_id: int
    text: str
    target_word: str
    filtered: bool
    filter_reason: str
    score_length: float
    score_hit_density: float
    score_cleanliness: float
    score_ne_penalty: float
    score_total: float


def load_scoring_config(path: Optional[str] = None) -> ScoringConfig:
    """Load a ScoringConfig from a TOML file, falling back to defaults.

    Keys are read from a ``[scoring]`` table if present, otherwise from
    the top level.  Only recognised fields are used; unknown keys are
    silently ignored.
    """
    if path is None:
        return ScoringConfig()

    with open(path, "rb") as f:
        data = tomllib.load(f)

    section = data.get("scoring", data)
    defaults = asdict(ScoringConfig())
    filtered = {
        k: type(defaults[k])(v) for k, v in section.items() if k in defaults
    }
    return ScoringConfig(**filtered)


def scoring_config_option(f):
    """Reusable Click decorator that adds ``--scoring-config``."""
    return click.option(
        "--scoring-config",
        type=click.Path(exists=True, dir_okay=False),
        default=None,
        help="TOML file overriding default scoring parameters.",
    )(f)


def lemmas_option(f):
    """Reusable Click decorator that adds ``--lemmas``."""
    return click.option(
        "--lemmas",
        required=True,
        help="Comma-separated lemma list for loanword density scoring.",
    )(f)


# ---------------------------------------------------------------------------
# CQP parsing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _gaussian(x: float, mu: float, sigma: float) -> float:
    return math.exp(-pow(x - mu, 2) / (2 * pow(sigma, 2)))


def count_hits_in_sentence(
    result: CQPResult, lemma_set_lower: Set[str]
) -> int:
    return len({
         t.lemma.lower()
         for t in result.tokens
         if t.lemma.lower() in lemma_set_lower
    })


def _normalised_weights(cfg: ScoringConfig) -> dict[str, float]:
    raw = {
        "length": cfg.weight_length,
        "hit": cfg.weight_hit,
        "clean": cfg.weight_clean,
        "ne": cfg.weight_ne,
    }
    total = sum(raw.values())
    if total == 0:
        return {k: 0.25 for k in raw}
    return {k: v / total for k, v in raw.items()}


def _make_filtered(
    res: CQPResult, reason: str
) -> ScoredResult:
    target = res.target_word
    return ScoredResult(
        cqp_id=res.cqp_id,
        text=res.text,
        target_word=target.word if target else "",
        filtered=True,
        filter_reason=reason,
        score_length=0.0,
        score_hit_density=0.0,
        score_cleanliness=0.0,
        score_ne_penalty=0.0,
        score_total=0.0,
    )


def score_sentence(
    res: CQPResult,
    hit_count: int = 0,
    cfg: Optional[ScoringConfig] = None,
) -> ScoredResult:

    if cfg is None:
        cfg = ScoringConfig()

    tokens = res.tokens
    s_len = len(tokens)

    # ------------------------------------------------------------------
    # Step 1: Hard filters
    # ------------------------------------------------------------------
    if s_len < cfg.min_tokens:
        return _make_filtered(res, "too_short")

    alpha_ratio = (
        sum(1 for t in tokens if t.word.isalnum()) / s_len
    )
    if alpha_ratio < cfg.min_alpha_ratio:
        return _make_filtered(res, "low_alpha")

    if cfg.filter_zero_hits and hit_count == 0:
        return _make_filtered(res, "zero_loanwords")

    # ------------------------------------------------------------------
    # Step 2: Continuous component scores, each in [0, 1]
    # ------------------------------------------------------------------
    score_length = _gaussian(s_len, cfg.length_mu, cfg.length_sigma)

    # Hit density - Gaussian
    score_hit = _gaussian(float(hit_count), cfg.hit_mu, cfg.hit_sigma)

    score_clean = alpha_ratio

    # NE penalty - ne_free NEs at no cost, exp-decay beyond
    ne_count = sum(
        1
        for i, t in enumerate(tokens)
        if i > 0 and t.word and t.word[0].isupper() and t.pos.startswith("n")
    )
    ne_excess = max(0, ne_count - cfg.ne_free)
    score_ne = math.exp(-cfg.ne_decay * ne_excess)

    # ------------------------------------------------------------------
    # Step 3: Weighted sum
    # ------------------------------------------------------------------
    w = _normalised_weights(cfg)
    score_total = (
        w["length"] * score_length
        + w["hit"] * score_hit
        + w["clean"] * score_clean
        + w["ne"] * score_ne
    )

    target = res.target_word
    return ScoredResult(
        cqp_id=res.cqp_id,
        text=res.text,
        target_word=target.word if target else "",
        filtered=False,
        filter_reason="",
        score_length=score_length,
        score_hit_density=score_hit,
        score_cleanliness=score_clean,
        score_ne_penalty=score_ne,
        score_total=score_total,
    )


# ---------------------------------------------------------------------------
# CQP interaction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CLI: query
# ---------------------------------------------------------------------------


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

# ---------------------------------------------------------------------------
# Register commands
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    corpus()
