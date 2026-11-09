"""Per-model serving profiles: what one response says about the model that served it.

A profile records how a served model behaves on one endpoint: how fast it
decodes (by context size), how fast it reads an uncached prompt, how many
characters it writes per token (thinking and visible output apart), and the
limits and request parameters the server was seen to honour. Each value is a
token-weighted moving average stored with its sample count and last update.

This module is pure: it turns an :class:`AssistantMessage` into samples and
folds samples into a profile. Where profiles are stored, and when they are
updated, is left to the application (``tau_coding.model_profiles``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from tau_agent.messages import AssistantMessage, ToolCall, Usage

#: Base smoothing factor per 1000 tokens of evidence.
PROFILE_ALPHA = 0.1
#: A sample more than this factor off the current value is clipped to it.
PROFILE_CLIP_FACTOR = 3.0
#: Clipping applies only once a value rests on at least this many tokens, so
#: a short bad first sample cannot clip every good one after it. A long one
#: can, for the few replies the average takes to recover.
CLIP_MIN_WEIGHT = 500
#: Decode-speed and chars-per-token samples need at least this many tokens;
#: a shorter reply says more about overhead than about the model.
MIN_MEASURED_TOKENS = 32
#: Context-size buckets for decode speed: (name, lower bound in prompt tokens).
DECODE_BUCKETS: tuple[tuple[str, int], ...] = (
    ("lt_8k", 0),
    ("8k_32k", 8_000),
    ("ge_32k", 32_000),
)
#: A prompt with fewer uncached tokens than this gives no prefill sample: the
#: time to first output is then mostly request overhead, not reading.
MIN_PREFILL_TOKENS = 128
#: Stop reasons whose usage and timing describe a whole response.
_MEASURED_STOP_REASONS = frozenset({"stop", "length", "toolUse"})

type DecodeBucket = Literal["lt_8k", "8k_32k", "ge_32k"]


def prompt_tokens(usage: Usage) -> int:
    """Every prompt token of one request: fresh, cache read and cache written."""
    return usage.input + usage.cache_read + usage.cache_write


def uncached_prompt_tokens(usage: Usage) -> int:
    """Prompt tokens the server had to read: ``Usage.input`` plus cache writes.

    Tau's adapters report ``input`` with cache reads and writes already taken
    out (the Pi convention), so the cached part is never in it.
    """
    return usage.input + usage.cache_write


def decode_bucket(context_tokens: int) -> DecodeBucket:
    """The decode-speed bucket for a request of ``context_tokens`` prompt tokens."""
    name = DECODE_BUCKETS[0][0]
    for bucket, lower in DECODE_BUCKETS:
        if context_tokens >= lower:
            name = bucket
    return name  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ProfileKey:
    """Which served model a profile describes.

    Two servers serving the same model are different profiles: ``endpoint`` is
    the base URL (or another stable provider-side identity when there is none).
    """

    provider: str
    endpoint: str
    model: str


@dataclass(frozen=True, slots=True)
class AveragedValue:
    """A token-weighted moving average with its sample count, weight and last update.

    ``weight`` is the number of tokens of evidence folded in so far.
    """

    value: float
    samples: int = 1
    updated_at_ms: int = 0
    weight: float = 0.0

    def updated(self, sample: float, tokens: int, now_ms: int) -> AveragedValue:
        """Fold one sample backed by ``tokens`` tokens into this average.

        The step is ``max(1 - (1 - PROFILE_ALPHA) ** (tokens / 1000),
        tokens / (weight + tokens))``. While little evidence has been seen the
        second term wins and the value is the token-weighted mean of every
        sample so far, so a tiny first sample cannot fix the value; once the
        weight passes about ``1000 / PROFILE_ALPHA`` tokens the moving average
        takes over. When the weight is at least ``CLIP_MIN_WEIGHT``, a sample
        more than ``PROFILE_CLIP_FACTOR`` off the current value is clipped to
        that bound rather than dropped, so one stall or cache miss cannot swing
        it.
        """
        tokens = max(tokens, 0)
        if self.weight >= CLIP_MIN_WEIGHT:
            sample = min(
                max(sample, self.value / PROFILE_CLIP_FACTOR), self.value * PROFILE_CLIP_FACTOR
            )
        moving = 1 - (1 - PROFILE_ALPHA) ** (tokens / 1000)
        total = self.weight + tokens
        cumulative = tokens / total if total > 0 else 0.0
        alpha = max(moving, cumulative)
        return AveragedValue(
            value=self.value + alpha * (sample - self.value),
            samples=self.samples + 1,
            updated_at_ms=now_ms,
            weight=total,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "samples": self.samples,
            "weight": self.weight,
            "updated_at_ms": self.updated_at_ms,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> AveragedValue:
        return cls(
            value=float(data["value"]),
            samples=int(data.get("samples", 1)),
            updated_at_ms=int(data.get("updated_at_ms", 0)),
            weight=float(data.get("weight") or 0.0),
        )


def averaged(
    current: AveragedValue | None, sample: float, tokens: int, now_ms: int
) -> AveragedValue:
    """``current`` updated by ``sample``; the first sample sets the value."""
    if current is None:
        return AveragedValue(
            value=sample, samples=1, updated_at_ms=now_ms, weight=float(max(tokens, 0))
        )
    return current.updated(sample, tokens, now_ms)


@dataclass(frozen=True, slots=True)
class RateSample:
    """One measured ratio and the number of tokens behind it."""

    value: float
    tokens: int


@dataclass(frozen=True, slots=True)
class ResponseSamples:
    """What one assistant response measured.

    ``thinking_chars_per_token`` and ``output_chars_per_token`` are set only
    when the response's output tokens can be split between them: the provider
    reported ``Usage.reasoning``, or the response held only one of the two.
    ``combined_chars_per_token`` is set whenever any output was measured.
    Decode-speed and chars-per-token samples each need at least
    ``MIN_MEASURED_TOKENS`` tokens behind them.
    """

    context_tokens: int
    output_tokens: int
    decode_tokens_per_second: RateSample | None = None
    prefill_tokens_per_second: RateSample | None = None
    thinking_chars_per_token: RateSample | None = None
    output_chars_per_token: RateSample | None = None
    combined_chars_per_token: RateSample | None = None


def _usage_estimated(message: AssistantMessage) -> bool:
    return any(
        (diagnostic.details or {}).get("usage_estimated") is True
        for diagnostic in message.diagnostics or ()
    )


def _visible_chars(message: AssistantMessage) -> int:
    chars = len(message.text)
    for block in message.content:
        if isinstance(block, ToolCall):
            chars += len(block.name) + len(
                json.dumps(block.arguments, ensure_ascii=False, separators=(",", ":"))
            )
    return chars


def _ratio(chars: int, tokens: int) -> RateSample | None:
    if chars <= 0 or tokens < MIN_MEASURED_TOKENS:
        return None
    return RateSample(value=chars / tokens, tokens=tokens)


def samples_from_response(message: AssistantMessage) -> ResponseSamples | None:
    """The samples one finished response gives, or None when it gives none.

    A response is skipped when the provider reported no output usage, when
    its usage was estimated by Tau rather than reported (a cut stream), and
    when it ended in an error or abort.
    """
    usage = message.usage
    if usage.output <= 0 or message.stop_reason not in _MEASURED_STOP_REASONS:
        return None
    if _usage_estimated(message):
        return None

    context = prompt_tokens(usage)
    decode: RateSample | None = None
    prefill: RateSample | None = None
    timing = message.timing
    if timing is not None and timing.time_to_first_output_ms is not None:
        first_ms = timing.time_to_first_output_ms
        decode_ms = timing.total_duration_ms - first_ms
        if decode_ms > 0 and usage.output >= MIN_MEASURED_TOKENS:
            decode = RateSample(value=usage.output * 1000 / decode_ms, tokens=usage.output)
        uncached = uncached_prompt_tokens(usage)
        mostly_cached = usage.cache_read > uncached
        if first_ms > 0 and uncached >= MIN_PREFILL_TOKENS and not mostly_cached:
            prefill = RateSample(value=uncached * 1000 / first_ms, tokens=uncached)

    thinking_chars = len(message.thinking_text)
    visible_chars = _visible_chars(message)
    thinking_tokens: int | None = None
    if usage.reasoning is not None and (usage.reasoning > 0 or thinking_chars == 0):
        thinking_tokens = min(usage.reasoning, usage.output)
    elif thinking_chars == 0:
        thinking_tokens = 0
    elif visible_chars == 0:
        thinking_tokens = usage.output
    thinking_cpt = output_cpt = None
    if thinking_tokens is not None:
        thinking_cpt = _ratio(thinking_chars, thinking_tokens)
        output_cpt = _ratio(visible_chars, usage.output - thinking_tokens)

    return ResponseSamples(
        context_tokens=context,
        output_tokens=usage.output,
        decode_tokens_per_second=decode,
        prefill_tokens_per_second=prefill,
        thinking_chars_per_token=thinking_cpt,
        output_chars_per_token=output_cpt,
        combined_chars_per_token=_ratio(thinking_chars + visible_chars, usage.output),
    )


def _optional(data: Mapping[str, Any], name: str) -> AveragedValue | None:
    value = data.get(name)
    return AveragedValue.from_json(value) if isinstance(value, Mapping) else None


def _optional_json(value: AveragedValue | None) -> dict[str, Any] | None:
    return value.to_json() if value is not None else None


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Everything recorded about one served model; see :class:`ProfileKey`."""

    key: ProfileKey
    decode_tokens_per_second: dict[str, AveragedValue] = field(default_factory=dict)
    prefill_tokens_per_second: AveragedValue | None = None
    thinking_chars_per_token: AveragedValue | None = None
    output_chars_per_token: AveragedValue | None = None
    combined_chars_per_token: AveragedValue | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    largest_output_tokens: int | None = None
    max_tokens_honoured: bool | None = None
    updated_at_ms: int = 0

    def with_response(self, samples: ResponseSamples, now_ms: int) -> ModelProfile:
        """This profile with one response's samples folded in."""
        decode = dict(self.decode_tokens_per_second)
        if samples.decode_tokens_per_second is not None:
            bucket = decode_bucket(samples.context_tokens)
            rate = samples.decode_tokens_per_second
            decode[bucket] = averaged(decode.get(bucket), rate.value, rate.tokens, now_ms)

        def fold(current: AveragedValue | None, sample: RateSample | None) -> AveragedValue | None:
            if sample is None:
                return current
            return averaged(current, sample.value, sample.tokens, now_ms)

        return replace(
            self,
            decode_tokens_per_second=decode,
            prefill_tokens_per_second=fold(
                self.prefill_tokens_per_second, samples.prefill_tokens_per_second
            ),
            thinking_chars_per_token=fold(
                self.thinking_chars_per_token, samples.thinking_chars_per_token
            ),
            output_chars_per_token=fold(
                self.output_chars_per_token, samples.output_chars_per_token
            ),
            combined_chars_per_token=fold(
                self.combined_chars_per_token, samples.combined_chars_per_token
            ),
            largest_output_tokens=max(self.largest_output_tokens or 0, samples.output_tokens),
            updated_at_ms=now_ms,
        )

    def with_message(self, message: AssistantMessage, now_ms: int) -> ModelProfile:
        """This profile updated by one response, unchanged when it gives no samples."""
        samples = samples_from_response(message)
        return self if samples is None else self.with_response(samples, now_ms)

    def to_json(self) -> dict[str, Any]:
        """The documented JSON shape (see dev-notes/architecture/model-profiles.md)."""
        return {
            "provider": self.key.provider,
            "endpoint": self.key.endpoint,
            "model": self.key.model,
            "decode_tokens_per_second": {
                bucket: _optional_json(self.decode_tokens_per_second.get(bucket))
                for bucket, _lower in DECODE_BUCKETS
            },
            "prefill_tokens_per_second": _optional_json(self.prefill_tokens_per_second),
            "chars_per_token": {
                "thinking": _optional_json(self.thinking_chars_per_token),
                "output": _optional_json(self.output_chars_per_token),
                "combined": _optional_json(self.combined_chars_per_token),
            },
            "limits": {
                "context_window": self.context_window,
                "max_output_tokens": self.max_output_tokens,
                "largest_output_tokens": self.largest_output_tokens,
            },
            "request_parameters": {"max_tokens_honoured": self.max_tokens_honoured},
            "updated_at_ms": self.updated_at_ms,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ModelProfile:
        decode = data.get("decode_tokens_per_second") or {}
        chars = data.get("chars_per_token") or {}
        limits = data.get("limits") or {}
        parameters = data.get("request_parameters") or {}
        honoured = parameters.get("max_tokens_honoured")
        return cls(
            key=ProfileKey(
                provider=str(data["provider"]),
                endpoint=str(data["endpoint"]),
                model=str(data["model"]),
            ),
            decode_tokens_per_second={
                bucket: AveragedValue.from_json(value)
                for bucket, value in decode.items()
                if isinstance(value, Mapping)
            },
            prefill_tokens_per_second=_optional(data, "prefill_tokens_per_second"),
            thinking_chars_per_token=_optional(chars, "thinking"),
            output_chars_per_token=_optional(chars, "output"),
            combined_chars_per_token=_optional(chars, "combined"),
            context_window=_optional_int(limits.get("context_window")),
            max_output_tokens=_optional_int(limits.get("max_output_tokens")),
            largest_output_tokens=_optional_int(limits.get("largest_output_tokens")),
            max_tokens_honoured=honoured if isinstance(honoured, bool) else None,
            updated_at_ms=int(data.get("updated_at_ms", 0)),
        )


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
