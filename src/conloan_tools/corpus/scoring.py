import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Set, Tuple

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
    sent_num: int = -1

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

    # Per-mask hard filters (opt-in)
    filter_require_loanword:     bool = False
    filter_require_code_switch:  bool = False
    filter_require_named_entity: bool = False

    # Length Gaussian
    length_mu:    float = 18.0
    length_sigma: float = 7.0

    # Loanword density Gaussian (density = count / sentence_len)
    loanword_mu:    float = 0.10
    loanword_sigma: float = 0.08

    # Code-switch density Gaussian
    code_switch_mu:    float = 0.15
    code_switch_sigma: float = 0.10

    # Named-entity density Gaussian
    # Low mu: a few NEs are fine, many are penalised
    named_entity_mu:    float = 0.05
    named_entity_sigma: float = 0.08

    # Component weights (normalised at scoring time)
    weight_length:      float = 0.20
    weight_clean:       float = 0.40
    weight_loanword:    float = 0.15
    weight_code_switch: float = 0.15
    weight_named_entity: float = 0.10

@dataclass
class ScoredResult:
    cqp_id: int
    tokens: List[Token]
    target_word: str
    filtered: bool
    filter_reason: str
    score_length: float
    score_cleanliness: float
    score_loanword: float
    score_code_switch: float
    score_named_entity: float
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

def _has_single_char_sequence(tokens: List[Token], limit: int = 3) -> bool:
    """Detects sequences of consecutive single-character tokens."""
    count = 0
    for t in tokens:
        # We only count it if it's alphanumeric (to avoid being tripped up by " , ")
        # or you can count everything if you want to be very aggressive.
        if len(t.word) == 1:
            count += 1
            if count >= limit:
                return True
        else:
            count = 0
    return False

def _is_contentful_run(run_tokens: List[Token]) -> bool:
    # Count tokens that are purely alphabetic and longer than 1 char
    content_tokens = [
        t for t in run_tokens 
        if t.word.isalpha() and len(t.word) > 1
    ]
    # If less than 50% of the run is actual words, it's noise/data
    return (len(content_tokens) / len(run_tokens)) > 0.5


def build_loanword_mask(
    result: CQPResult, lemma_set_lower: Set[str]
) -> List[bool]:
    """True for each token whose lemma is in the loanword set."""
    return [t.lemma.lower() in lemma_set_lower for t in result.tokens]


def build_named_entity_mask(result: CQPResult) -> List[bool]:
    """True for tokens identified as named entities (capitalised nouns)."""
    return [
        i > 0 and bool(t.word) and t.word[0].isupper() and t.pos.startswith("n")
        for i, t in enumerate(result.tokens)
    ]

def _normalised_weights(cfg: ScoringConfig) -> dict[str, float]:
    raw = {
        "length": cfg.weight_length,
        "clean":  cfg.weight_clean,
        "loanword":     cfg.weight_loanword,
        "code_switch":  cfg.weight_code_switch,
        "named_entity": cfg.weight_named_entity,
    }
    total = sum(raw.values())
    if total == 0:
        n = len(raw)
        return {k: 1.0 / n for k in raw}
    return {k: v / total for k, v in raw.items()}

def _make_filtered(
    res: CQPResult, reason: str
) -> ScoredResult:
   target = res.target_word
   return ScoredResult(
        cqp_id=res.cqp_id,
        tokens=res.tokens,
        target_word=target.word if target else "",
        filtered=True,
        filter_reason=reason,
        score_length=0.0,
        score_cleanliness=0.0,
        score_loanword=0.0,
        score_code_switch=0.0,
        score_named_entity=0.0,
        score_total=0.0,
    )

def score_sentence(
    res: CQPResult,
    loanword_mask: Optional[List[int]] = None,
    code_switch_mask: Optional[List[int]] = None,
    named_entity_mask: Optional[List[int]] = None,
    cfg: Optional[ScoringConfig] = None,
) -> ScoredResult:

    if cfg is None:
        cfg = ScoringConfig()

    tokens = res.tokens
    s_len = len(tokens)
    if s_len == 0:
        return _make_filtered(res, "empty")

    # Normalise masks — default to all-False if not provided
    loanword_mask     = loanword_mask     or [0] * s_len
    code_switch_mask  = code_switch_mask  or [0] * s_len
    named_entity_mask = named_entity_mask or [0] * s_len

    if s_len < cfg.min_tokens:
        return _make_filtered(res, "too_short")

    alpha_ratio = sum(1 for t in tokens if t.word.isalnum()) / s_len
    if alpha_ratio < cfg.min_alpha_ratio:
        return _make_filtered(res, "low_alpha")

    loanword_count    = sum(loanword_mask)
    code_switch_count = sum(code_switch_mask)
    ne_count          = sum(named_entity_mask)

    if cfg.filter_require_loanword and loanword_count == 0:
        return _make_filtered(res, "zero_loanwords")
    if cfg.filter_require_code_switch and code_switch_count == 0:
        return _make_filtered(res, "zero_code_switch")
    if cfg.filter_require_named_entity and ne_count == 0:
        return _make_filtered(res, "zero_named_entities")

    score_length = _gaussian(s_len, cfg.length_mu, cfg.length_sigma)
    score_clean  = alpha_ratio

    score_loanword    = _gaussian(loanword_count / s_len,    cfg.loanword_mu,    cfg.loanword_sigma)
    score_code_switch = _gaussian(code_switch_count / s_len, cfg.code_switch_mu, cfg.code_switch_sigma)
    score_named_entity = _gaussian(ne_count / s_len,         cfg.named_entity_mu, cfg.named_entity_sigma)

    w = _normalised_weights(cfg)
    score_total = (
        w["length"] * score_length
        + w["clean"] * score_clean
        + w["loanword"]     * score_loanword
        + w["code_switch"]  * score_code_switch
        + w["named_entity"] * score_named_entity
    )

    target = res.target_word
    return ScoredResult(
        cqp_id=res.cqp_id,
        tokens=res.tokens,
        target_word=target.word if target else "",
        filtered=False,
        filter_reason="",
        score_length=score_length,
        score_cleanliness=score_clean,
        score_loanword=score_loanword,
        score_code_switch=score_code_switch,
        score_named_entity=score_named_entity,
        score_total=score_total,
    )
