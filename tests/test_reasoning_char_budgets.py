"""Thinking budgets are in characters; tokens only where a limit is in tokens.

Characters can be counted live from the stream. A token budget is turned into
characters at the model's measured thinking ratio, and the budget is clamped
under the server's output cap with room for the reply. At an assumed 4 characters
a token, a model that really runs nearer 2.5 reaches the cap first and ends with
nothing written.
"""

from pathlib import Path

import anyio
import pytest

from tau_agent import AssistantMessage
from tau_agent.messages import ThinkingContent, UserMessage
from tau_ai import FakeProvider
from tau_ai.model_profile import AveragedValue, ModelProfile, ProfileKey
from tau_coding import CodingSession
from tau_coding.context_window import (
    CHARS_PER_TOKEN,
    estimate_context_tokens,
    estimate_text_tokens,
)
from tau_coding.guards import (
    BUDGET_CUT_NUDGE,
    BUDGET_CUT_WRITE_NUDGE,
)
from tau_coding.model_profiles import (
    MIN_TRUSTED_RATIO_WEIGHT,
    ModelProfileStore,
    context_chars_per_token,
)
from tau_coding.reasoning_guard import (
    BUDGET_CUT_REPLY_SHARE,
    DEFAULT_THINKING_CHARS_PER_TOKEN,
    LEVEL_REASONING_CHARS,
    MIN_BUDGET_CUT_REPLY_ROOM_TOKENS,
    TURN_REASONING_CHARS_ENV,
    TURN_REASONING_TOKENS_ENV,
    ServedModelLimits,
    budget_cut_reply_room_tokens,
    request_reasoning_chars,
    thinking_chars_per_token,
)
from test_reasoning_tokens import _reasons_long, _thinking_session
from test_session_output_limit import _answers

#: An output cap, and a model's characters a token with thinking included.
OUTPUT_CAP = 16384
MEASURED_RATIO = 2.7
KEY = ProfileKey(provider="local", endpoint="", model="fake")


def _value(value: float) -> AveragedValue:
    return AveragedValue(value=value, samples=10, weight=10_000.0)


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TURN_REASONING_CHARS_ENV, raising=False)
    monkeypatch.delenv(TURN_REASONING_TOKENS_ENV, raising=False)


def _store_profile(session: CodingSession, **fields: object) -> None:
    """Store a fake profile for the session's served model."""
    key, paths = session._served_profile_key()
    ModelProfileStore.at(paths).put(ModelProfile(key=key, **fields))  # type: ignore[arg-type]


def _cuts(session: CodingSession) -> list[AssistantMessage]:
    return [
        message
        for message in session.messages
        if isinstance(message, AssistantMessage) and message.diagnostics
    ]


def _thought(message: AssistantMessage) -> int:
    return sum(
        len(block.thinking) for block in message.content if isinstance(block, ThinkingContent)
    )


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    responses: list[list[object]],
    *,
    profile: dict[str, object] | None = None,
    thinking_format: str | None = None,
    level: str = "medium",
) -> CodingSession:
    """Prompt a fake-provider session once, its served model's profile `profile`."""

    async def go() -> CodingSession:
        main = FakeProvider(responses)
        session = await _thinking_session(
            monkeypatch, tmp_path, main, thinking_format=thinking_format, level=level
        )
        if profile is not None:
            _store_profile(session, **profile)
        async for _event in session.prompt("Encode it."):
            pass
        return session

    return anyio.run(go)


def test_the_reasoning_budget_is_counted_in_characters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv(TURN_REASONING_CHARS_ENV, "1000")
    # The old token variable is set too: the characters win.
    monkeypatch.setenv(TURN_REASONING_TOKENS_ENV, "50")
    session = _run(monkeypatch, tmp_path, [_reasons_long(), _answers("Acted.")])

    [cut] = _cuts(session)
    assert cut.stop_reason == "length"
    [diagnostic] = cut.diagnostics or []
    assert diagnostic.type == "reasoning_budget_cut"
    assert (diagnostic.details or {})["chars"] == 1000
    # Cut at the first delta past 1000 characters (40-character deltas).
    assert 1000 < _thought(cut) <= 1000 + 40
    assert session.messages[-1].text == "Acted."

    # The levels mean the characters they meant at 4 a token before.
    assert LEVEL_REASONING_CHARS == {
        "minimal": 1024 * 4,
        "low": 2048 * 4,
        "medium": 8192 * 4,
        "high": 16384 * 4,
    }


def test_a_token_budget_is_turned_into_characters_at_the_models_thinking_ratio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv(TURN_REASONING_TOKENS_ENV, "300")

    # The thinking ratio wins over the combined one; the visible-output one
    # is never used for thinking.
    both = ModelProfile(
        key=KEY,
        thinking_chars_per_token=_value(2.0),
        output_chars_per_token=_value(3.96),
        combined_chars_per_token=_value(3.0),
    )
    combined_only = ModelProfile(
        key=KEY,
        output_chars_per_token=_value(3.96),
        combined_chars_per_token=_value(3.0),
    )
    assert thinking_chars_per_token(both) == 2.0
    assert thinking_chars_per_token(combined_only) == 3.0
    for profile, chars in ((both, 600), (combined_only, 900)):
        limits = ServedModelLimits(thinking_chars_per_token=thinking_chars_per_token(profile))
        assert request_reasoning_chars([UserMessage(content="Encode it.")], None, limits) == chars

    # End to end: the session reads the served model's stored profile.
    session = _run(
        monkeypatch,
        tmp_path,
        [_reasons_long(), _answers("Acted.")],
        profile={
            "thinking_chars_per_token": _value(2.0),
            "output_chars_per_token": _value(3.96),
            "combined_chars_per_token": _value(3.0),
        },
    )
    [cut] = _cuts(session)
    [diagnostic] = cut.diagnostics or []
    assert (diagnostic.details or {})["chars"] == 600
    assert 600 < _thought(cut) <= 600 + 40


def test_the_cut_comes_before_the_servers_output_cap_with_room_to_reply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_env(monkeypatch)
    # At "high": 65,536 characters asked for, but the cap of 16,384 tokens
    # keeps 4,096 for the reply, so thinking gets 12,288 tokens, 33,177
    # characters at 2.7.
    assert budget_cut_reply_room_tokens(OUTPUT_CAP) == 4096
    limits = ServedModelLimits(
        thinking_chars_per_token=MEASURED_RATIO, max_output_tokens=OUTPUT_CAP
    )
    request = [UserMessage(content="Encode it.")]
    budget = request_reasoning_chars(request, LEVEL_REASONING_CHARS["high"], limits)
    assert budget == int((OUTPUT_CAP - 4096) * MEASURED_RATIO) == 33177
    # At the real ratio the cut lands under the cap with the room left.
    assert budget / MEASURED_RATIO + 4096 <= OUTPUT_CAP
    # A budget already under the clamp is left alone.
    assert request_reasoning_chars(request, LEVEL_REASONING_CHARS["low"], limits) == 8192
    # A small cap keeps at least MIN_BUDGET_CUT_REPLY_ROOM_TOKENS for the reply.
    assert budget_cut_reply_room_tokens(4096) == MIN_BUDGET_CUT_REPLY_ROOM_TOKENS == 2048
    small = ServedModelLimits(thinking_chars_per_token=2.0, max_output_tokens=4096)
    assert request_reasoning_chars(request, 100_000, small) == (4096 - 2048) * 2
    # No cap known: no clamp.
    assert request_reasoning_chars(request, 100_000, ServedModelLimits()) == 100_000

    # End to end, the profile's measured cap clamps the environment's budget.
    monkeypatch.setenv(TURN_REASONING_CHARS_ENV, "100000")
    session = _run(
        monkeypatch,
        tmp_path,
        [_reasons_long(), _answers("Acted.")],
        profile={"thinking_chars_per_token": _value(2.0), "max_output_tokens": 4096},
    )
    [cut] = _cuts(session)
    [diagnostic] = cut.diagnostics or []
    assert (diagnostic.details or {})["chars"] == 4096
    assert 4096 < _thought(cut) <= 4096 + 40


def test_an_unmeasured_model_uses_a_conservative_ratio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_env(monkeypatch)
    assert DEFAULT_THINKING_CHARS_PER_TOKEN == 2.5
    assert thinking_chars_per_token(None) == 2.5
    # Only a visible-output ratio measured: still the conservative default.
    output_only = ModelProfile(key=KEY, output_chars_per_token=_value(3.96))
    assert thinking_chars_per_token(output_only) == 2.5

    monkeypatch.setenv(TURN_REASONING_TOKENS_ENV, "300")
    session = _run(monkeypatch, tmp_path, [_reasons_long(), _answers("Acted.")])
    [cut] = _cuts(session)
    [diagnostic] = cut.diagnostics or []
    details = diagnostic.details or {}
    assert details["chars"] == 750
    assert details["chars_per_token"] == 2.5


def test_the_context_estimate_uses_the_models_measured_ratio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_env(monkeypatch)
    assert estimate_text_tokens("x" * 270, 2.7) == 100
    assert estimate_text_tokens("x" * 270) == 68  # 4 a token, as before
    assert context_chars_per_token(None) == CHARS_PER_TOKEN
    measured = ModelProfile(
        key=KEY,
        thinking_chars_per_token=_value(1.5),
        combined_chars_per_token=_value(2.0),
    )
    assert context_chars_per_token(measured) == 2.0

    async def usage(where: Path, profile: dict[str, object] | None) -> int:
        session = await _thinking_session(monkeypatch, where, FakeProvider([]))
        if profile is not None:
            _store_profile(session, **profile)
        return session.context_usage.total_tokens

    plain = anyio.run(usage, tmp_path / "plain", None)
    dense = anyio.run(usage, tmp_path / "dense", {"combined_chars_per_token": _value(2.0)})
    # The session's own estimate reads the served model's ratio.
    assert dense > plain
    # Half the characters a token is twice the tokens, past a message's overhead.
    long = (UserMessage(content="x" * 4000),)
    assert estimate_context_tokens(system="", messages=long, tools=()) == 4 + 1000
    assert estimate_context_tokens(system="", messages=long, tools=(), chars_per_token=2.0) == (
        4 + 2000
    )


def test_a_cut_reports_characters_and_the_tokens_they_stand_for(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv(TURN_REASONING_CHARS_ENV, "1000")
    session = _run(
        monkeypatch,
        tmp_path,
        [_reasons_long(), _answers("Acted.")],
        profile={"thinking_chars_per_token": _value(2.0)},
    )
    [cut] = _cuts(session)
    [diagnostic] = cut.diagnostics or []
    details = diagnostic.details or {}
    assert details["chars"] == 1000
    assert details["tokens"] == 500
    assert details["chars_per_token"] == 2.0
    for nudge in (BUDGET_CUT_NUDGE, BUDGET_CUT_WRITE_NUDGE):
        assert "thinking budget" in nudge
        assert "(in tokens)" not in nudge


def test_a_cut_replay_keeps_its_share_in_characters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv(TURN_REASONING_CHARS_ENV, "8000")
    limits = ServedModelLimits(
        thinking_chars_per_token=MEASURED_RATIO, max_output_tokens=OUTPUT_CAP
    )
    for nudge in (BUDGET_CUT_NUDGE, BUDGET_CUT_WRITE_NUDGE):
        assert request_reasoning_chars([UserMessage(content=nudge)], None, limits) == int(
            8000 * BUDGET_CUT_REPLY_SHARE
        )
    # The share is of the clamped budget.
    monkeypatch.setenv(TURN_REASONING_CHARS_ENV, "100000")
    assert request_reasoning_chars([UserMessage(content=BUDGET_CUT_NUDGE)], None, limits) == int(
        33177 * BUDGET_CUT_REPLY_SHARE
    )

    monkeypatch.setenv(TURN_REASONING_CHARS_ENV, "2000")
    session = _run(monkeypatch, tmp_path, [_reasons_long(), _reasons_long(), _answers("Wrote it.")])
    first, replay = _cuts(session)
    assert [(c.diagnostics or [])[0].details["chars"] for c in (first, replay)] == [2000, 500]  # type: ignore[index]
    assert 2000 < _thought(first) <= 2040
    assert 500 < _thought(replay) <= 540
    assert session.messages[-1].text == "Wrote it."


def test_a_lightly_measured_ratio_is_not_trusted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_env(monkeypatch)
    # A profile with a combined ratio from two samples sets neither ratio.
    light = AveragedValue(value=3.49, samples=2, weight=2000.0)
    warmed = ModelProfile(key=KEY, thinking_chars_per_token=light, combined_chars_per_token=light)
    assert thinking_chars_per_token(warmed) == DEFAULT_THINKING_CHARS_PER_TOKEN
    assert context_chars_per_token(warmed) == CHARS_PER_TOKEN
    # A light thinking ratio falls through to a trusted combined one.
    mixed = ModelProfile(
        key=KEY,
        thinking_chars_per_token=light,
        combined_chars_per_token=AveragedValue(
            value=2.7, samples=20, weight=MIN_TRUSTED_RATIO_WEIGHT
        ),
    )
    assert thinking_chars_per_token(mixed) == 2.7
    assert context_chars_per_token(mixed) == 2.7

    monkeypatch.setenv(TURN_REASONING_TOKENS_ENV, "300")
    session = _run(
        monkeypatch,
        tmp_path,
        [_reasons_long(), _answers("Acted.")],
        profile={"thinking_chars_per_token": light, "combined_chars_per_token": light},
    )
    [cut] = _cuts(session)
    [diagnostic] = cut.diagnostics or []
    assert (diagnostic.details or {})["chars_per_token"] == DEFAULT_THINKING_CHARS_PER_TOKEN
    assert (diagnostic.details or {})["chars"] == 750
