"""Centralised model configuration for TemplateForecaster.

Keeping these objects in a single module avoids merge-conflicts and makes it
possible to tweak/benchmark models without touching application code.
"""

from typing import Any

from forecasting_tools import GeneralLlm

from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback

__all__ = [
    "DISAGREEMENT_ANALYZER_LLM",
    "FORECASTER_LLMS",
    "FORECASTER_MODEL_NAMES",
    "MARKET_QUERY_AUTHOR_LLM_CONFIG",
    "MARKET_RANKER_LLM_CONFIG",
    "PARSER_LLM",
    "RESEARCHER_LLM",
    "STACKER_FALLBACK_LLM",
    "STACKER_LLM",
    "SUMMARIZER_LLM",
]
# Reasoning models ignore (or degrade under) explicit sampling params, so we
# defer to provider defaults. temperature=None is explicit but redundant on
# ft 0.2.92, whose GeneralLlm ctor already defaults temperature to None (0.2.54
# injected 0 when the arg was omitted); top_p flows via **kwargs and is never set.
REASONING_MODEL_CONFIG: dict[str, Any] = {
    "temperature": None,
    "max_tokens": 64_000,  # Gemma's free forecaster slot overrides this to its 32k catalog limit.
    "stream": False,
    "timeout": 480,
    "allowed_tries": 3,
}
# Low-effort utility slots (parser, summarizer, analyzer). Same sampling-param
# rationale as REASONING_MODEL_CONFIG: temperature=None defers to provider
# defaults (redundant on ft 0.2.92, whose ctor default is already None); top_p
# left unset.
UTILITY_MODEL_CONFIG: dict[str, Any] = {
    "temperature": None,
    "max_tokens": 32_000,
    "stream": False,
    "timeout": 300,
    "allowed_tries": 3,
}
ACCEPTABLE_QUANTS = [
    "fp8",
    "fp16",
    "bf16",
    "fp32",
    "unknown",
]

# Per-instance allowed_tries=1 override (Round-2): forecaster .invoke is wrapped
# in the broad retry gated on TRANSIENT_RETRY_MAX_ELAPSED_S (forecaster_runners.py)
# so we can impose the universal "never retry a slow failure" deadline-safety rule
# that forecasting-tools' un-gated tenacity cannot. Spread per-instance (NOT by mutating
# REASONING_MODEL_CONFIG) so PARSER_LLM / STACKER configs are untouched.
_FORECASTER_CONFIG = {**REASONING_MODEL_CONFIG, "allowed_tries": 1}


def forecaster_role(model: str) -> str:
    """``forecaster:<vendor>`` for an ``openrouter/<vendor>/<model>`` roster slug.

    The CREDIT_ROLE_SPEND spend line every roster slot books under. The roster is
    latest-per-vendor, one slot each, so the VENDOR is the stable identity of a slot
    across model rotations — a per-model role would start a new time series at every swap
    and defeat the era-over-era cost comparison this exists for.
    """
    parts = model.split("/")
    if len(parts) < 3 or parts[0] != "openrouter":
        raise ValueError(f"forecaster_role expects an openrouter/<vendor>/<model> slug, got {model!r}")
    return f"forecaster:{parts[1]}"


def _forecaster_slot(model: str, **kwargs: Any) -> GeneralLlm:
    """One roster member, booked in the CREDIT_ROLE_SPEND ledger under ``forecaster:<vendor>``.

    The role is derived from the slug rather than written beside it so a roster swap cannot
    leave a slot mislabeled.
    """
    config = {**_FORECASTER_CONFIG, **kwargs}
    return build_llm_with_openrouter_fallback(model=model, role=forecaster_role(model), **config)


# 2026-10-02: operator-approved free-model config-era boundary. The OpenRouter catalog had
# no free Inception route; the third slot uses the catalog-listed free Gemma model instead.
# All three ids and zero-price catalog entries were verified live before this change.
FORECASTER_LLMS: list[GeneralLlm] = [
    _forecaster_slot("openrouter/nvidia/nemotron-3-super-120b-a12b:free"),
    _forecaster_slot("openrouter/qwen/qwen3.8-27b:free"),
    _forecaster_slot("openrouter/google/gemma-4-31b-it:free", max_tokens=32_000),
]


def _forecaster_display_name(llm: GeneralLlm) -> str:
    """Short label for a forecaster (e.g. 'claude-opus-5.5') — strips the 'openrouter/<provider>/' prefix.

    Used by performance_analysis.parsing to map 'Forecaster N' labels in bot comments
    back to a model name without having to hand-maintain a parallel list.
    """
    return llm.model.rsplit("/", 1)[-1]


FORECASTER_MODEL_NAMES: list[str] = [_forecaster_display_name(llm) for llm in FORECASTER_LLMS]

# AskNews-only summarization. Keep one try because orchestrator owns its elapsed-gated retry.
SUMMARIZER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    role="summarizer",
    **{**UTILITY_MODEL_CONFIG, "allowed_tries": 1},
)
# Structured extraction uses the free Gemma route, which passed the existing parser bake-off.
PARSER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/google/gemma-4-31b-it:free",
    role="parser",
    **UTILITY_MODEL_CONFIG,
)
# Populated to avoid framework defaults; the pipeline uses its research providers instead.
RESEARCHER_LLM = SUMMARIZER_LLM

# Stacking is production-disabled; both configured attempts use free, cross-provider models.
STACKER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    role="stacker",
    **{**REASONING_MODEL_CONFIG, "allowed_tries": 1},
)

# Fallback used when the primary stacker times out or errors; one try and a tighter timeout.
STACKER_FALLBACK_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/qwen/qwen3.8-27b:free",
    role="stacker_fallback",
    **{**REASONING_MODEL_CONFIG, "allowed_tries": 1, "timeout": 300},
)

# --- The prediction-market provider's two LLM stages ---
# Both use free Qwen routes, stay lazy until invoked, and leave retry ownership to llm_retry.

# Ranker emits up to eight rows. No output cap: a truncated ranking loses the whole result.
MARKET_RANKER_LLM_CONFIG: dict = {
    "model": "openrouter/qwen/qwen3.8-27b:free",
    "role": "market_ranker",
    "temperature": None,
    "timeout": 90,
    "allowed_tries": 1,
}

# Query author adds domain vocabulary to deterministic query variants; failure costs recall nothing.
MARKET_QUERY_AUTHOR_LLM_CONFIG: dict = {
    "model": "openrouter/qwen/qwen3.8-27b:free",
    "role": "market_query_author",
    "temperature": None,
    "timeout": 45,
    "allowed_tries": 1,
}


# Low-effort disagreement analysis; the caller owns the elapsed-gated retry.
DISAGREEMENT_ANALYZER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/qwen/qwen3.8-27b:free",
    role="crux_analyzer",
    **{**UTILITY_MODEL_CONFIG, "allowed_tries": 1},
)
