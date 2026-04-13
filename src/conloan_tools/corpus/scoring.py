import math
from enum import Enum
from dataclasses import dataclass, asdict, field
from typing import List, Literal, Optional, Set
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import click


class QueryProfile(str, Enum):
    LEMMAS = "lemmas"
    CODE_SWITCH = "code-switch"
    NER = "ner"

    def __str__(self) -> str:
        return self.value


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
    min_tokens:      int   = 0
    max_tokens:      int   = 9999
    min_alpha_ratio: float = 0.0

    # Primary-signal gates
    filter_require_loanword:     bool = False
    filter_require_code_switch:  bool = False
    filter_require_named_entity: bool = False

    filter_forbid_loanword:      bool = False
    filter_forbid_code_switch:   bool = False
    filter_forbid_named_entity:  bool = False
    filter_require_context_lang: str = ""

    # NER label ignore list (replaces ner_ignore_misc)
    ner_ignore_labels: set[str] = field(default_factory=set)

    filter_min_loanword_density:     float = 0.0
    filter_min_code_switch_density:  float = 0.0
    filter_min_named_entity_density: float = 0.0

    # Length Gaussian
    length_mu:    float = 20.0
    length_sigma: float = 10.0

    # Loanword density Gaussian
    loanword_mu:    float = 0.15
    loanword_sigma: float = 0.10

    # Code-switch density Gaussian
    code_switch_mu:    float = 0.15
    code_switch_sigma: float = 0.10

    # Named-entity density Gaussian
    named_entity_mu:    float = 0.15
    named_entity_sigma: float = 0.10

    # Component weights (normalised at scoring time)
    weight_length:       float = 0.0
    weight_alpha:        float = 0.0
    weight_loanword:     float = 0.0
    weight_code_switch:  float = 0.0
    weight_named_entity: float = 0.0


@dataclass
class ScoredResult:
    cqp_id: int
    tokens: List[Token]
    target_word: str
    filtered: bool
    filter_reason: str
    score_length: float
    score_alpha: float
    score_loanword: float
    score_code_switch: float
    score_named_entity: float
    score_total: float


_PROFILE_DEFAULTS: dict[QueryProfile, dict] = {
    QueryProfile.LEMMAS: {
        "min_tokens":                      8,
        "max_tokens":                      128,
        "min_alpha_ratio":                 0.5,
        "filter_require_loanword":         True,
        "filter_forbid_named_entity":      True,

        "weight_length":                   0.4,
        "weight_alpha":                    0.4,
        "weight_loanword":                 0.2,

        "length_mu":                       30, # real median
        "length_sigma":                    10, # 15.6 real sigma -> 10
    },
    QueryProfile.CODE_SWITCH: {
        "min_tokens":                      5,
        "max_tokens":                      128,
        "min_alpha_ratio":                 0.5,
        "filter_require_code_switch":      True,
        "filter_forbid_named_entity":      True,
        "filter_require_context_lang":     "lv",

        "weight_code_switch":              0.33,
        "weight_length":                   0.33,
        "weight_alpha":                    0.33,

        "length_mu":                       30,
        "length_sigma":                    10,

        "code_switch_mu":                  0.30,
        "code_switch_sigma":               0.15,
    },
    QueryProfile.NER: {
        "min_tokens":                      5,
        "max_tokens":                      128,
        "min_alpha_ratio":                 0.5,
        "filter_require_named_entity":     True,

        "weight_named_entity":             0.33,
        "weight_length":                   0.33,
        "weight_alpha":                    0.33,

        "length_mu":                       30,
        "length_sigma":                    10,

        "named_entity_mu":                 0.30,
        "named_entity_sigma":              0.15,
    },
}


def _precompute_densities(
    tokens: List[Token],
    loanword_mask: List[int],
    code_switch_mask: List[int],
    named_entity_mask: List[int],
) -> tuple[int, float, float, float, float]:
    s_len = len(tokens)
    alpha_ratio = sum(1 for t in tokens if t.word.isalnum()) / s_len
    lw_density  = sum(loanword_mask)     / s_len
    cs_density  = sum(code_switch_mask)  / s_len
    ne_density  = sum(named_entity_mask) / s_len
    return s_len, alpha_ratio, lw_density, cs_density, ne_density


def _validate_profile_defaults(
    profile: QueryProfile,
    known_fields: set[str],
) -> None:
    """Raise if a profile references a field not in ScoringConfig."""
    overrides = _PROFILE_DEFAULTS.get(profile, {})
    unknown = {k for k in overrides if k not in known_fields}
    if unknown:
        raise RuntimeError(
            f"Profile {profile!r} references unknown ScoringConfig "
            f"field(s): {sorted(unknown)}"
        )


_KNOWN_FIELDS = set(asdict(ScoringConfig()).keys())
for _p in QueryProfile:
    _validate_profile_defaults(_p, _KNOWN_FIELDS)


def load_scoring_config(
    path: Optional[str] = None,
    profile: QueryProfile | str = QueryProfile.LEMMAS,
) -> ScoringConfig:
    """
    Build a ScoringConfig by merging layers in order:
        1. ScoringConfig dataclass defaults  (all-zero weights — intentional)
        2. Profile defaults
        3. [scoring] section in TOML file
        4. [scoring.profiles.<profile>] in TOML
    A profile is always required; there is no generic/passthrough profile.
    """
    if not isinstance(profile, QueryProfile):
        try:
            profile = QueryProfile(profile)
        except ValueError:
            valid = [p.value for p in QueryProfile]
            raise click.UsageError(
                f"Unknown profile {profile!r}. Valid profiles: {valid}"
            )

    defaults = asdict(ScoringConfig())
    merged: dict = {**defaults, **_PROFILE_DEFAULTS[profile]}

    if path is not None:
        with open(path, "rb") as f:
            data = tomllib.load(f)

        base: dict = data.get("scoring", {})

        unknown_global = {
            k for k in base if k not in defaults and k != "profiles"
        }
        if unknown_global:
            raise click.UsageError(
                f"Unknown keys in [scoring]: {sorted(unknown_global)}"
            )

        for k, v in base.items():
            if k in defaults:
                merged[k] = type(defaults[k])(v)

        profile_key = profile.value
        profile_section: dict = base.get("profiles", {}).get(profile_key, {})
        unknown_profile = {k for k in profile_section if k not in defaults}
        if unknown_profile:
            raise click.UsageError(
                f"Unknown keys in [scoring.profiles.{profile_key}]: "
                f"{sorted(unknown_profile)}"
            )
        for k, v in profile_section.items():
            merged[k] = type(defaults[k])(v)

    cfg = ScoringConfig(**{k: merged[k] for k in defaults})

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
        "alpha":        cfg.weight_alpha,
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
        score_alpha=0.0,
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
    densities: tuple[int, float, float, float, float] | None = None,
    detected_lang: str | None = None,
) -> Optional[str]:
    if densities is None:
        densities = _precompute_densities(
            tokens, loanword_mask, code_switch_mask, named_entity_mask
        )
    s_len, alpha_ratio, lw_density, cs_density, ne_density = densities

    if s_len == 0:
        return "empty"
    if s_len < cfg.min_tokens:
        return "too_short"
    if s_len > cfg.max_tokens:
        return "too_long"
    if alpha_ratio < cfg.min_alpha_ratio:
        return "low_alpha"
    if cfg.filter_require_context_lang and detected_lang != cfg.filter_require_context_lang:
        return "wrong_context_language"

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
    densities: tuple[int, float, float, float, float] | None = None,
) -> tuple[float, float, float, float, float]:
    if densities is None:
        densities = _precompute_densities(
            tokens, loanword_mask, code_switch_mask, named_entity_mask
        )
    s_len, alpha_ratio, lw_density, cs_density, ne_density = densities

    score_length       = _gaussian(s_len,      cfg.length_mu,       cfg.length_sigma)
    score_alpha        = alpha_ratio
    score_loanword     = _gaussian(lw_density, cfg.loanword_mu,     cfg.loanword_sigma)
    score_code_switch  = _gaussian(cs_density, cfg.code_switch_mu,  cfg.code_switch_sigma)
    score_named_entity = _gaussian(ne_density, cfg.named_entity_mu, cfg.named_entity_sigma)

    return score_length, score_alpha, score_loanword, score_code_switch, score_named_entity


_weights_cache: dict[int, dict[str, float]] = {}

def _get_weights(cfg: ScoringConfig) -> dict[str, float]:
    key = id(cfg)
    if key not in _weights_cache:
        _weights_cache[key] = _normalised_weights(cfg)
    return _weights_cache[key]


def score_sentence(
    res: CQPResult,
    loanword_mask: Optional[List[int]] = None,
    code_switch_mask: Optional[List[int]] = None,
    named_entity_mask: Optional[List[int]] = None,
    detected_lang: str | None = None,
    cfg: Optional[ScoringConfig] = None,
) -> ScoredResult:
    if cfg is None:
        cfg = ScoringConfig()

    tokens = res.tokens
    s_len = len(tokens)

    loanword_mask     = loanword_mask     or [0] * s_len
    code_switch_mask  = code_switch_mask  or [0] * s_len
    named_entity_mask = named_entity_mask or [0] * s_len

    densities = _precompute_densities(
        tokens, loanword_mask, code_switch_mask, named_entity_mask
    )
    reason = _apply_hard_gates(
        tokens, loanword_mask, code_switch_mask, named_entity_mask,
        cfg, densities, detected_lang=detected_lang,
    )
    if reason:
        return _make_filtered(res, reason)

    score_length, score_alpha, score_loanword, score_code_switch, score_named_entity = (
        _compute_scores(
            tokens, loanword_mask, code_switch_mask, named_entity_mask, cfg, densities
        )
    )

    w = _get_weights(cfg)
    score_total = (
          w["length"]       * score_length
        + w["alpha"]        * score_alpha
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
        score_alpha=score_alpha,
        score_loanword=score_loanword,
        score_code_switch=score_code_switch,
        score_named_entity=score_named_entity,
        score_total=score_total,
    )
