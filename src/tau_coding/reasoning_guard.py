"""Stops a model's streamed reasoning when it loops or runs past a budget.

A reasoning model can go round the same lines until the output limit and end
with no reply and no tool call. `RequestGuard` sits in front of a run's
provider and ends such a response early, as a length stop, so the session's
output-limit follow-up asks the model to act.

The guard steers with that follow-up only. It never edits the prompt or the
thinking level, so the request prefix stays cacheable. All the thresholds here
are heuristics.
"""

from __future__ import annotations

import re
from collections import Counter, deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from os import environ

from tau_agent.messages import (
    REASONING_BUDGET_CUT,
    REASONING_LOOP_CUT,
    AgentMessage,
    AssistantMessage,
    AssistantMessageDiagnostic,
    ThinkingContent,
    Usage,
    UserMessage,
)
from tau_agent.provider import CancellationToken, ModelProvider
from tau_agent.provider_events import (
    AssistantDoneEvent,
    AssistantMessageEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ToolCallStartEvent,
)
from tau_agent.tools import AgentTool
from tau_agent.types import JSONValue
from tau_ai.model_profile import ModelProfile
from tau_coding.context_window import (
    CHARS_PER_TOKEN,
    estimate_context_tokens,
    estimate_message_tokens,
)
from tau_coding.guards import BUDGET_CUT_NUDGE, BUDGET_CUT_WRITE_NUDGE
from tau_coding.model_profiles import trusted_ratio

#: The setting a session starts with: `on`, `off` or a budget in characters.
REASONING_GUARD_ENV = "TAU_REASONING_GUARD"
#: Reasoning loops when some line has recurred REASONING_LOOP_REPEATS times while
#: distinct lines are under REASONING_LOOP_DISTINCT_FRACTION of the lines, over
#: all the reasoning so far (from REASONING_LOOP_MIN_LINES lines on) or over its
#: last REASONING_LOOP_WINDOW lines, which catches a loop after a long sound
#: start. A line that is one unit of at most REASONING_LOOP_RUN_UNIT chars
#: repeated for REASONING_LOOP_RUN_CHARS chars is a loop too, and so is a line
#: whose sentences loop as lines would. Reasoning that repeats a short line a
#: few times among varied ones meets none of these.
REASONING_LOOP_REPEATS = 8
REASONING_LOOP_DISTINCT_FRACTION = 0.25
REASONING_LOOP_MIN_LINES = 40
REASONING_LOOP_WINDOW = 64
REASONING_LOOP_RUN_CHARS = 2_000
REASONING_LOOP_RUN_UNIT = 40
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

#: Characters a turn may think before its first tool call, when the environment
#: sets it. Characters can be counted live from the stream, while
#: servers report tokens only at the end. A time budget would give a slow model
#: less thinking than a fast one.
TURN_REASONING_CHARS_ENV = "TAU_TURN_REASONING_CHARS"
#: The same budget in tokens, turned into characters at the model's thinking
#: ratio (`thinking_chars_per_token`). `TURN_REASONING_CHARS_ENV` wins when both
#: are set.
TURN_REASONING_TOKENS_ENV = "TAU_TURN_REASONING_TOKENS"
#: The thinking budget per level in characters, for a model whose thinking format
#: carries no level or budget to the server (`UNBOUNDED_THINKING_FORMATS`), so
#: that "medium" does not think as long as "high" would. A level not here (xhigh,
#: max) has no budget.
LEVEL_REASONING_CHARS: dict[str, int] = {
    "minimal": 4096,
    "low": 8192,
    "medium": 32768,
    "high": 65536,
}
#: Characters a thinking token when the served model's profile has measured
#: neither its thinking nor its combined ratio. Thinking tends to run well below
#: the usual 4 characters a token; this errs low, so an unmeasured budget stops
#: short of the server's output cap.
DEFAULT_THINKING_CHARS_PER_TOKEN = 2.5
#: The share of the server's output cap (in tokens) kept free for the reply after
#: a budget cut, and the least room kept: a budget that reaches the cap ends the
#: reply with nothing written.
BUDGET_CUT_REPLY_ROOM_SHARE = 0.25
MIN_BUDGET_CUT_REPLY_ROOM_TOKENS = 2048
#: The share of the budget a budget cut's follow-up reply may think. A reply that
#: will act tends to do so early, so short tries cost little.
BUDGET_CUT_REPLY_SHARE = 0.25


@dataclass(frozen=True, slots=True)
class ReasoningGuardSetting:
    """Whether the guard runs, and a per-turn budget that beats the others."""

    enabled: bool = True
    chars: int | None = None


def startup_reasoning_guard() -> ReasoningGuardSetting:
    """The setting `REASONING_GUARD_ENV` asks for; an unreadable value is ignored."""
    try:
        return parse_reasoning_guard(environ.get(REASONING_GUARD_ENV, "on"))
    except ValueError:
        return ReasoningGuardSetting()


def parse_reasoning_guard(value: str) -> ReasoningGuardSetting:
    """The setting `value` asks for: `on`, `off`, or a budget in characters.

    `on` also goes back to the default budget.
    """
    text = value.strip().lower()
    if text == "on":
        return ReasoningGuardSetting()
    if text == "off":
        return ReasoningGuardSetting(enabled=False)
    if text.isascii() and text.isdigit() and int(text) > 0:
        return ReasoningGuardSetting(chars=int(text))
    raise ValueError(f"Expected on, off or a positive number of characters, got {value.strip()!r}.")


class ReasoningLoopDetector:
    """Report, as reasoning streams in, when it has started looping."""

    def __init__(self) -> None:
        self._line: list[str] = []
        self._line_chars = 0
        self._line_checked_at = 0
        self._tail = ""
        self._sentence: list[str] = []
        self._sentence_counts: Counter[str] = Counter()
        self._sentences = 0
        self._counts: Counter[str] = Counter()
        self._lines = 0
        self._top = 0
        self._window: deque[str] = deque()
        self._window_counts: Counter[str] = Counter()
        self.details: dict[str, JSONValue] = {}

    def feed(self, delta: str) -> bool:
        """Take the next reasoning delta; return whether the reasoning loops."""
        *ended, rest = delta.split("\n")
        if ended:
            ended[0] = "".join((*self._line, ended[0]))
            self._line, self._line_chars, self._line_checked_at = [], 0, 0
            self._tail = ""
            self._sentence, self._sentences = [], 0
            self._sentence_counts.clear()
            for line in ended:
                if self._add(line.strip()):
                    return True
        if rest:
            self._line.append(rest)
            self._line_chars += len(rest)
            self._tail = (self._tail + rest)[-REASONING_LOOP_RUN_CHARS:]
        return self._run_loops() or self._sentences_loop(rest)

    def _add(self, line: str) -> bool:
        if not line:
            return False
        self._lines += 1
        count = self._counts[line] + 1
        self._counts[line] = count
        self._top = max(self._top, count)
        self._window.append(line)
        self._window_counts[line] += 1
        if len(self._window) > REASONING_LOOP_WINDOW:
            dropped = self._window.popleft()
            self._window_counts[dropped] -= 1
            if not self._window_counts[dropped]:
                del self._window_counts[dropped]
        if (
            self._lines >= REASONING_LOOP_MIN_LINES
            and self._top >= REASONING_LOOP_REPEATS
            and len(self._counts) < REASONING_LOOP_DISTINCT_FRACTION * self._lines
        ):
            repeated, count = self._counts.most_common(1)[0]
            return self._found(repeated, count, lines=self._lines, distinct=len(self._counts))
        if (
            len(self._window) == REASONING_LOOP_WINDOW
            and len(self._window_counts) < REASONING_LOOP_DISTINCT_FRACTION * REASONING_LOOP_WINDOW
        ):
            repeated, count = self._window_counts.most_common(1)[0]
            if count >= REASONING_LOOP_REPEATS:
                return self._found(
                    repeated, count, lines=REASONING_LOOP_WINDOW, distinct=len(self._window_counts)
                )
        return False

    def _run_loops(self) -> bool:
        """Whether the line being written is one short unit over and over."""
        if (
            self._line_chars < REASONING_LOOP_RUN_CHARS
            or self._line_chars - self._line_checked_at < REASONING_LOOP_RUN_UNIT
        ):
            return False
        self._line_checked_at = self._line_chars
        tail = self._tail
        for unit in range(1, REASONING_LOOP_RUN_UNIT + 1):
            if tail[unit:] == tail[:-unit]:
                self.details = {"kind": "run", "unit": tail[:unit], "chars": self._line_chars}
                return True
        return False

    def _sentences_loop(self, text: str) -> bool:
        """Whether the line being written loops in its sentences as lines would."""
        last = self._sentence[-1][-1:] if self._sentence else ""
        *ended, rest = _SENTENCE_END.split(last + text)
        if not ended:
            if text:
                self._sentence.append(text)
            return False
        ended[0] = "".join(self._sentence) + ended[0][len(last) :]
        self._sentence = [rest] if rest else []
        for sentence in ended:
            if sentence := sentence.strip():
                self._sentences += 1
                self._sentence_counts[sentence] += 1
        if (
            self._sentences < REASONING_LOOP_MIN_LINES
            or len(self._sentence_counts) >= REASONING_LOOP_DISTINCT_FRACTION * self._sentences
        ):
            return False
        repeated, count = self._sentence_counts.most_common(1)[0]
        if count < REASONING_LOOP_REPEATS:
            return False
        self.details = {
            "kind": "sentences",
            "sentence": repeated[:200],
            "repeats": count,
            "sentences": self._sentences,
            "distinct": len(self._sentence_counts),
        }
        return True

    def _found(self, line: str, repeats: int, *, lines: int, distinct: int) -> bool:
        self.details = {
            "kind": "lines",
            "line": line[:200],
            "repeats": repeats,
            "lines": lines,
            "distinct": distinct,
        }
        return True


@dataclass(frozen=True, slots=True)
class ServedModelLimits:
    """What a request's thinking budget needs from the served model.

    `thinking_chars_per_token` turns a limit in tokens into characters for
    thinking; `context_chars_per_token` is the ratio the context estimates use;
    `max_output_tokens` is the server's output cap, None when unknown.
    """

    thinking_chars_per_token: float = DEFAULT_THINKING_CHARS_PER_TOKEN
    context_chars_per_token: float = CHARS_PER_TOKEN
    max_output_tokens: int | None = None


def thinking_chars_per_token(profile: ModelProfile | None) -> float:
    """Characters a thinking token for the model `profile` describes: its
    measured thinking ratio, else its combined one, else
    `DEFAULT_THINKING_CHARS_PER_TOKEN`. The visible-output ratio is never used:
    it runs well above what thinking does."""
    if profile is not None:
        for value in (profile.thinking_chars_per_token, profile.combined_chars_per_token):
            if trusted_ratio(value):
                return value.value
    return DEFAULT_THINKING_CHARS_PER_TOKEN


def budget_cut_reply_room_tokens(max_output_tokens: int) -> int:
    """Tokens of the output cap kept free for the reply after a budget cut."""
    return max(
        MIN_BUDGET_CUT_REPLY_ROOM_TOKENS, int(max_output_tokens * BUDGET_CUT_REPLY_ROOM_SHARE)
    )


def _env_budget(name: str) -> tuple[bool, float | None]:
    """Whether `name` sets a budget, and its value: None when it is set to zero
    or less (no budget). An unset or unreadable value sets none."""
    raw = environ.get(name, "").strip()
    try:
        number = float(raw) if raw else None
    except ValueError:
        number = None
    if number is None:
        return False, None
    return True, number if number > 0 else None


def turn_reasoning_chars(
    default: int | None = None,
    *,
    chars_per_token: float = DEFAULT_THINKING_CHARS_PER_TOKEN,
) -> int | None:
    """The run's thinking budget per turn in characters: `TURN_REASONING_CHARS_ENV`
    when set, else `TURN_REASONING_TOKENS_ENV` at `chars_per_token`, else
    `default` (the thinking level's). Either set to zero or less means none."""
    is_set, chars = _env_budget(TURN_REASONING_CHARS_ENV)
    if is_set:
        return None if chars is None else max(1, int(chars))
    is_set, tokens = _env_budget(TURN_REASONING_TOKENS_ENV)
    if is_set:
        return None if tokens is None else max(1, int(int(tokens) * chars_per_token))
    return default


def _capped_reasoning_chars(
    budget: int | None, max_output_tokens: int | None, chars_per_token: float
) -> int | None:
    """`budget`, at most the characters the server's output cap leaves for
    thinking once the reply's room is kept: `(cap - room) * chars_per_token`.
    A cap no larger than its room keeps a quarter of it for thinking."""
    if budget is None or max_output_tokens is None or max_output_tokens <= 0:
        return budget
    thinking_tokens = max(
        max_output_tokens - budget_cut_reply_room_tokens(max_output_tokens),
        max_output_tokens // 4,
    )
    return min(budget, max(1, int(thinking_tokens * chars_per_token)))


def request_reasoning_chars(
    messages: list[AgentMessage],
    default: int | None = None,
    limits: ServedModelLimits | None = None,
    *,
    chars: int | None = None,
) -> int | None:
    """The thinking budget in characters of a request on `messages`: `chars` when
    given, else the environment's or `default`, clamped under the served model's
    output cap (`limits`). A budget cut's nudge reply has its share
    (`BUDGET_CUT_REPLY_SHARE`) of that."""
    limits = limits or ServedModelLimits()
    ratio = limits.thinking_chars_per_token
    wanted = chars if chars is not None else turn_reasoning_chars(default, chars_per_token=ratio)
    budget = _capped_reasoning_chars(wanted, limits.max_output_tokens, ratio)
    last = messages[-1] if messages else None
    if (
        budget is not None
        and isinstance(last, UserMessage)
        and last.text in (BUDGET_CUT_NUDGE, BUDGET_CUT_WRITE_NUDGE)
    ):
        return max(1, int(budget * BUDGET_CUT_REPLY_SHARE))
    return budget


class RequestGuard:
    """The provider a harness run's loop sends its requests through.

    Each request's reasoning is watched, and one that loops, or thinks
    past its character budget before a tool call, is stopped and ends as a
    length stop, so the session follows up. `setting` is read for every request,
    so a change reaches the run's next request; while it is off, requests go
    straight to `inner`. `level_chars` is the budget from the session's thinking
    level, for a format that cannot send one. `limits` gives the served model's
    ratios and output cap.
    """

    def __init__(
        self,
        inner: ModelProvider,
        *,
        setting: Callable[[], ReasoningGuardSetting] = ReasoningGuardSetting,
        level_chars: Callable[[], int | None] | None = None,
        limits: Callable[[], ServedModelLimits] | None = None,
    ) -> None:
        self.inner = inner
        self.setting = setting
        self.level_chars = level_chars
        self.limits = limits

    def __getattr__(self, name: str) -> object:
        return getattr(self.inner, name)

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        stream = self.inner.stream_response(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            signal=signal,
            session_id=session_id,
        )
        setting = self.setting()
        if not setting.enabled:
            return stream
        limits = self.limits() if self.limits is not None else ServedModelLimits()
        return cut_looping_reasoning(
            stream,
            request_reasoning_chars(
                messages,
                self.level_chars() if self.level_chars else None,
                limits,
                chars=setting.chars,
            ),
            chars_per_token=limits.thinking_chars_per_token,
            request_tokens=lambda: estimate_context_tokens(
                system=system,
                messages=tuple(messages),
                tools=tuple(tools),
                chars_per_token=limits.context_chars_per_token,
            ),
            context_chars_per_token=limits.context_chars_per_token,
        )


async def cut_looping_reasoning(
    source: AsyncIterator[AssistantMessageEvent],
    budget: int | None = None,
    *,
    chars_per_token: float = DEFAULT_THINKING_CHARS_PER_TOKEN,
    request_tokens: Callable[[], int] | None = None,
    context_chars_per_token: float = CHARS_PER_TOKEN,
) -> AsyncIterator[AssistantMessageEvent]:
    """Pass a response's events on, ending it as a length stop if its reasoning
    loops, or if it thinks more than `budget` characters before a tool call.

    The characters are counted from the streamed thinking deltas, since
    servers report usage only at the end. A budget cut reports the budget in
    characters and the tokens they stand for at `chars_per_token`. Time never
    cuts reasoning.

    A cut response never gets the provider's closing usage, so what it is
    missing is estimated: the request from `request_tokens`, the output from
    the text streamed so far at `context_chars_per_token`.
    """
    detector = ReasoningLoopDetector()
    watching = True
    thought_chars = 0
    try:
        async for event in source:
            yield event
            if isinstance(event, ToolCallStartEvent):
                watching = False
            if not (watching and isinstance(event, ThinkingDeltaEvent)):
                continue
            thought_chars += len(event.delta)
            if detector.feed(event.delta):
                diagnostic = AssistantMessageDiagnostic(
                    type=REASONING_LOOP_CUT, details=detector.details
                )
            elif budget is not None and thought_chars > budget:
                diagnostic = AssistantMessageDiagnostic(
                    type=REASONING_BUDGET_CUT,
                    details={
                        "chars": budget,
                        "tokens": max(1, round(budget / chars_per_token)),
                        "chars_per_token": round(chars_per_token, 3),
                    },
                )
            else:
                continue
            partial = event.partial
            block = (
                partial.content[event.content_index]
                if event.content_index < len(partial.content)
                else None
            )
            if isinstance(block, ThinkingContent):
                yield ThinkingEndEvent(
                    content_index=event.content_index, content=block.thinking, partial=partial
                )
            usage = _cut_usage(partial, request_tokens, context_chars_per_token)
            if usage is not partial.usage:
                diagnostic = diagnostic.model_copy(
                    update={"details": {**(diagnostic.details or {}), "usage_estimated": True}}
                )
            message = partial.model_copy(
                update={
                    "stop_reason": "length",
                    "usage": usage,
                    "diagnostics": [*(partial.diagnostics or []), diagnostic],
                }
            )
            yield AssistantDoneEvent(reason="length", message=message)
            return
    finally:
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            await aclose()


def _cut_usage(
    partial: AssistantMessage,
    request_tokens: Callable[[], int] | None,
    chars_per_token: float = CHARS_PER_TOKEN,
) -> Usage:
    """`partial`'s usage, with the parts a cut stream never reported estimated.

    The request counts as unreported when no input or cache count arrived
    (OpenAI-compatible servers send usage only at the end); the output, when
    it is below the streamed text's estimate. Returns `partial.usage` itself
    when nothing was missing.
    """
    usage = partial.usage
    update: dict[str, int] = {}
    if request_tokens is not None and not (usage.input or usage.cache_read or usage.cache_write):
        update["input"] = request_tokens()
    streamed = estimate_message_tokens(partial, chars_per_token) - estimate_message_tokens(
        AssistantMessage(content=[]), chars_per_token
    )
    if usage.output < streamed:
        update["output"] = streamed
    if not update:
        return usage
    estimated = usage.model_copy(update=update)
    return estimated.model_copy(
        update={
            "total_tokens": estimated.input
            + estimated.output
            + estimated.cache_read
            + estimated.cache_write
        }
    )
