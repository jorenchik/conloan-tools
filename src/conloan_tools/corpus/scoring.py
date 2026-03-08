import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Set

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

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

    if s_len < cfg.min_tokens:
        return _make_filtered(res, "too_short")
    alpha_ratio = (
        sum(1 for t in tokens if t.word.isalnum()) / s_len
    )
    if alpha_ratio < cfg.min_alpha_ratio:
        return _make_filtered(res, "low_alpha")
    if cfg.filter_zero_hits and hit_count == 0:
        return _make_filtered(res, "zero_loanwords")

    score_length = _gaussian(s_len, cfg.length_mu, cfg.length_sigma)
    score_hit = _gaussian(float(hit_count), cfg.hit_mu, cfg.hit_sigma)
    score_clean = alpha_ratio
    ne_count = sum(
        1
        for i, t in enumerate(tokens)
        if i > 0 and t.word and t.word[0].isupper() and t.pos.startswith("n")
    )
    ne_excess = max(0, ne_count - cfg.ne_free)
    score_ne = math.exp(-cfg.ne_decay * ne_excess)

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
