"""Thinking-mode primitives for Tau coding sessions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

ThinkingLevel = Literal["default", "off", "minimal", "low", "medium", "high", "xhigh", "max"]
ThinkingParameter = Literal["reasoning_effort", "reasoning.effort", "anthropic.thinking"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]

THINKING_LEVELS: tuple[ThinkingLevel, ...] = (
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
DEFAULT_THINKING_LEVEL: ThinkingLevel = "medium"
#: Defers the level downstream: to the provider catalog's `thinking_default`,
#: and when the catalog names none, to the API server (no thinking field sent).
#: Not one of THINKING_LEVELS, the concrete levels a request can carry.
UPSTREAM_THINKING_LEVEL: Literal["default"] = "default"

THINKING_LEVEL_DESCRIPTIONS: dict[ThinkingLevel, str] = {
    "default": "Catalog default, else the server's",
    "off": "No reasoning",
    "minimal": "Very brief reasoning",
    "low": "Light reasoning",
    "medium": "Moderate reasoning",
    "high": "Deep reasoning",
    "xhigh": "Very deep reasoning",
    "max": "Maximum reasoning",
}


def normalize_thinking_level(value: str | None) -> ThinkingLevel:
    """Return a valid Tau thinking level or raise a user-facing error."""
    if value is None:
        return DEFAULT_THINKING_LEVEL
    normalized = value.strip().lower()
    if normalized == UPSTREAM_THINKING_LEVEL or normalized in THINKING_LEVELS:
        return normalized
    allowed = ", ".join((UPSTREAM_THINKING_LEVEL, *THINKING_LEVELS))
    raise ValueError(f"Unknown thinking mode: {value}. Available modes: {allowed}")


def normalize_thinking_levels(values: Sequence[str]) -> tuple[ThinkingLevel, ...]:
    """Return a validated, duplicate-free thinking level tuple."""
    if isinstance(values, str) or not values:
        allowed = ", ".join(THINKING_LEVELS)
        raise ValueError(f"Thinking modes must be a non-empty list. Available modes: {allowed}")

    normalized = tuple(normalize_thinking_level(value) for value in values)
    if UPSTREAM_THINKING_LEVEL in normalized:
        # Offered for every model with levels; a level list names concrete modes.
        allowed = ", ".join(THINKING_LEVELS)
        raise ValueError(
            f"Thinking mode default is implicit and cannot be listed. Available modes: {allowed}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError("Thinking modes must be unique")
    return normalized


def reasoning_effort_for_level(level: str | None) -> ReasoningEffort:
    """Map Tau's UI thinking level to an OpenAI-compatible reasoning effort."""
    normalized = normalize_thinking_level(level)
    if normalized == UPSTREAM_THINKING_LEVEL:
        raise ValueError("Thinking mode default has no effort; resolve it first")
    if normalized == "off":
        return "none"
    return normalized


def anthropic_thinking_budget_for_level(level: str | None) -> int | None:
    """Map Tau's UI thinking level to Anthropic extended-thinking tokens."""
    normalized = normalize_thinking_level(level)
    if normalized == UPSTREAM_THINKING_LEVEL:
        raise ValueError("Thinking mode default has no budget; resolve it first")
    if normalized == "off":
        return None
    return {
        "minimal": 1024,
        "low": 2048,
        "medium": 4096,
        "high": 8192,
        "xhigh": 16384,
        "max": 32768,
    }[normalized]


def next_thinking_level(
    current: str | None,
    *,
    available: tuple[ThinkingLevel, ...] = THINKING_LEVELS,
) -> ThinkingLevel:
    """Return the next thinking level in a stable cycle."""
    if not available:
        return DEFAULT_THINKING_LEVEL
    try:
        normalized_current = normalize_thinking_level(current)
        index = available.index(normalized_current)
    except ValueError:
        return available[0]
    return available[(index + 1) % len(available)]
