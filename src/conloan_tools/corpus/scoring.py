import math
from dataclasses import dataclass, asdict
from typing import List, Literal, Optional, Set
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import click


QueryProfile = Literal["lemma", "code_switch", "ner", "generic"]


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
    # Hard gates
    min_tokens:      int   = 5
    max_tokens:      int   = 60
    min_alpha_ratio: float = 0.5

    # Primary-signal gates (profiles set these)
    filter_require_loanword:     bool = False
    filter_require_code_switch:  bool = False
    filter_require_named_entity: bool = False

    # Mirror: forbid any presence of the signal
    filter_forbid_loanword:      bool = False
    filter_forbid_code_switch:   bool = False
    filter_forbid_named_entity:  bool = False

    # Minimum density gates
    filter_min_loanword_density:     float = 0.0
    filter_min_code_switch_density:  float = 0.0
    filter_min_named_entity_density: float = 0.0

    # Length Gaussian
    length_mu:    float = 18.0
    length_sigma: float = 7.0

    # Loanword density Gaussian
    loanword_mu:    float = 0.10
    loanword_sigma: float = 0.08

    # Code-switch density Gaussian
    code_switch_mu:    float = 0.15
    code_switch_sigma: float = 0.10

    # Named-entity density Gaussian
    named_entity_mu:    float = 0.05
    named_entity_sigma: float = 0.08

    # Component weights (normalised at scoring time)
    weight_length:       float = 0.20
    weight_clean:        float = 0.40
    weight_loanword:     float = 0.15
    weight_code_switch:  float = 0.15
    weight_named_entity: float = 0.10

    # Cross-signal soft penalties
    # Applied as: component_score *= (1 - penalty * offending_density)
    penalty_excess_named_entity: float = 0.0
    penalty_excess_code_switch:  float = 0.0


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


_PROFILE_DEFAULTS: dict[QueryProfile, dict] = {
    "lemma": {
        "filter_require_loanword":         True,
        # "filter_require_code_switch":      False,
        # "filter_require_named_entity":     False,
        # "filter_forbid_loanword":          False,
        # "filter_forbid_code_switch":       False,
        "filter_forbid_named_entity":      True,
        "filter_min_loanword_density":     0.00,
        "weight_loanword":                 0.40,
        "weight_code_switch":              0.05,
        "weight_named_entity":             0.05,
        "weight_length":                   0.20,
        "weight_clean":                    0.30,
        "penalty_excess_named_entity":     0.30,
        "penalty_excess_code_switch":      0.20,
    },
    "code_switch": {
        # "filter_require_loanword":         False,
        "filter_require_code_switch":      True,
        # "filter_require_named_entity":     False,
        # "filter_forbid_loanword":          False,
        # "filter_forbid_code_switch":       False,
        # "filter_forbid_named_entity":      False,
        "filter_min_code_switch_density":  0.00,
        "weight_code_switch":              0.40,
        "weight_loanword":                 0.10,
        "weight_named_entity":             0.05,
        "weight_length":                   0.20,
        "weight_clean":                    0.25,
        "code_switch_mu":                  0.30,
        "code_switch_sigma":               0.15,
        "min_tokens":                      8,
    },
    "ner": {
        # "filter_require_loanword":         False,
        # "filter_require_code_switch":      False,
        "filter_require_named_entity":     True,
        # "filter_forbid_loanword":          False,
        # "filter_forbid_code_switch":       False,
        # "filter_forbid_named_entity":      False,
        "filter_min_named_entity_density":     0.00,
        "weight_named_entity":                 0.40,
        "weight_loanword":                     0.10,
        "weight_code_switch":                  0.05,
        "weight_length":                       0.20,
        "weight_clean":                        0.25,
        "named_entity_mu":                     0.15,
        "named_entity_sigma":                  0.10,
    },
    "generic": {},
}


def load_scoring_config(
    path: Optional[str] = None,
    profile: QueryProfile = "generic",
) -> ScoringConfig:
    """
    Build a ScoringConfig by merging four layers in order:
        1. ScoringConfig dataclass defaults
        2. Profile defaults (_PROFILE_DEFAULTS[profile])
        3. [scoring] section in TOML file          (global overrides)
        4. [scoring.profiles.<profile>] in TOML    (per-profile overrides)
    Each layer only overrides what it explicitly specifies.
    """
    defaults = asdict(ScoringConfig())
    merged: dict = {**defaults, **_PROFILE_DEFAULTS.get(profile, {})}

    if path is not None:
        with open(path, "rb") as f:
            data = tomllib.load(f)

        base: dict = data.get("scoring", {})

        # Validate global keys
        unknown_global = {
            k for k in base if k not in defaults and k != "profiles"
        }
        if unknown_global:
            raise click.UsageError(
                f"Unknown keys in [scoring]: {sorted(unknown_global)}"
            )

        # Layer 3: global file overrides
        for k, v in base.items():
            if k in defaults:
                merged[k] = type(defaults[k])(v)

        # Layer 4: per-profile file overrides
        profile_section: dict = base.get("profiles", {}).get(profile, {})
        unknown_profile = {k for k in profile_section if k not in defaults}
        if unknown_profile:
            raise click.UsageError(
                f"Unknown keys in [scoring.profiles.{profile}]: "
                f"{sorted(unknown_profile)}"
            )
        for k, v in profile_section.items():
            merged[k] = type(defaults[k])(v)

    cfg = ScoringConfig(**{k: merged[k] for k in defaults})

    # Validate require/forbid contradictions
    pairs = [
        ("filter_require_loanword",     "filter_forbid_loanword",     "loanword"),
        ("filter_require_code_switch",  "filter_forbid_code_switch",  "code_switch"),
        ("filter_require_named_entity", "filter_forbid_named_entity", "named_entity"),
    ]
    for require_field, forbid_field, label in pairs:
        if getattr(cfg, require_field) and getattr(cfg, forbid_field):
            raise click.UsageError(
                f"Contradictory config: filter_require_{label} and "
                f"filter_forbid_{label} are both True."
            )
    
    return cfg


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


def _gaussian(x: float, mu: float, sigma: float) -> float:
    return math.exp(-pow(x - mu, 2) / (2 * pow(sigma, 2)))


def _normalised_weights(cfg: ScoringConfig) -> dict[str, float]:
    raw = {
        "length":       cfg.weight_length,
        "clean":        cfg.weight_clean,
        "loanword":     cfg.weight_loanword,
        "code_switch":  cfg.weight_code_switch,
        "named_entity": cfg.weight_named_entity,
    }
    total = sum(raw.values())
    if total == 0:
        n = len(raw)
        return {k: 1.0 / n for k in raw}
    return {k: v / total for k, v in raw.items()}


def _make_filtered(res: CQPResult, reason: str) -> ScoredResult:
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


def _apply_hard_gates(
    tokens: List[Token],
    loanword_mask: List[int],
    code_switch_mask: List[int],
    named_entity_mask: List[int],
    cfg: ScoringConfig,
) -> Optional[str]:
    """
    Evaluate all hard-gate conditions in order.
    Returns a filter reason string on the first failure, None if all pass.
    """
    s_len = len(tokens)

    if s_len == 0:
        return "empty"
    if s_len < cfg.min_tokens:
        return "too_short"
    if s_len > cfg.max_tokens:
        return "too_long"

    alpha_ratio = sum(1 for t in tokens if t.word.isalnum()) / s_len
    if alpha_ratio < cfg.min_alpha_ratio:
        return "low_alpha"

    lw_density = sum(loanword_mask) / s_len
    cs_density = sum(code_switch_mask) / s_len
    ne_density = sum(named_entity_mask) / s_len

    if cfg.filter_require_loanword and lw_density == 0.0:
        return "zero_loanwords"
    if cfg.filter_require_code_switch and cs_density == 0.0:
        return "zero_code_switch"
    if cfg.filter_require_named_entity and ne_density == 0.0:
        return "zero_named_entities"

    if cfg.filter_forbid_loanword and lw_density > 0.0:
        return "forbidden_loanwords"
    if cfg.filter_forbid_code_switch and cs_density > 0.0:
        return "forbidden_code_switch"
    if cfg.filter_forbid_named_entity and ne_density > 0.0:
        return "forbidden_named_entities"

    if lw_density < cfg.filter_min_loanword_density:
        return "loanword_density_below_min"
    if cs_density < cfg.filter_min_code_switch_density:
        return "code_switch_density_below_min"
    if ne_density < cfg.filter_min_named_entity_density:
        return "named_entity_density_below_min"

    return None


def _compute_scores(
    tokens: List[Token],
    loanword_mask: List[int],
    code_switch_mask: List[int],
    named_entity_mask: List[int],
    cfg: ScoringConfig,
) -> tuple[float, float, float, float, float]:
    """
    Compute the five component scores.
    Cross-signal penalties are applied here as multiplicative reductions
    on the loanword component to avoid inflating loanword scores when
    NE/CS density is high.
    Returns: (length, clean, loanword, code_switch, named_entity)
    """
    s_len = len(tokens)
    alpha_ratio = sum(1 for t in tokens if t.word.isalnum()) / s_len

    lw_density = sum(loanword_mask) / s_len
    cs_density = sum(code_switch_mask) / s_len
    ne_density = sum(named_entity_mask) / s_len

    score_length       = _gaussian(s_len,      cfg.length_mu,       cfg.length_sigma)
    score_clean        = alpha_ratio
    score_loanword     = _gaussian(lw_density, cfg.loanword_mu,     cfg.loanword_sigma)
    score_code_switch  = _gaussian(cs_density, cfg.code_switch_mu,  cfg.code_switch_sigma)
    score_named_entity = _gaussian(ne_density, cfg.named_entity_mu, cfg.named_entity_sigma)

    # Cross-signal soft penalties on loanword score
    score_loanword *= max(0.0, 1.0 - cfg.penalty_excess_named_entity * ne_density)
    score_loanword *= max(0.0, 1.0 - cfg.penalty_excess_code_switch  * cs_density)

    return score_length, score_clean, score_loanword, score_code_switch, score_named_entity


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

    loanword_mask     = loanword_mask     or [0] * s_len
    code_switch_mask  = code_switch_mask  or [0] * s_len
    named_entity_mask = named_entity_mask or [0] * s_len

    reason = _apply_hard_gates(
        tokens, loanword_mask, code_switch_mask, named_entity_mask, cfg
    )
    if reason:
        return _make_filtered(res, reason)

    score_length, score_clean, score_loanword, score_code_switch, score_named_entity = (
        _compute_scores(
            tokens, loanword_mask, code_switch_mask, named_entity_mask, cfg
        )
    )

    w = _normalised_weights(cfg)
    score_total = (
        w["length"]       * score_length
        + w["clean"]      * score_clean
        + w["loanword"]   * score_loanword
        + w["code_switch"] * score_code_switch
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
