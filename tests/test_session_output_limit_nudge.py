"""The output-limit nudge with another message queued beside it.

A steering message queued beside a nudge goes first and takes the nudge's
request. Without care the nudge is delivered a turn late, and each cut-off after
it queues another, past the bound.
"""

from pathlib import Path

import pytest

from conftest import isolate_home
from tau_agent import AssistantMessage, UserMessage
from tau_agent.messages import ThinkingContent, Usage
from tau_ai import FakeProvider
from tau_coding import CodingSession, OpenAICompatibleProviderConfig, ProviderSettings
from tau_coding import reasoning_guard as reasoning_guard_module
from tau_coding import session as coding_session_module
from tau_coding.guards import (
    BUDGET_CUT_NUDGE,
    BUDGET_CUT_WRITE_NUDGE,
    LOOP_CUT_NUDGE,
    MAX_BUDGET_CUT_CONTINUATIONS,
    MAX_OUTPUT_LIMIT_CONTINUATIONS,
    MAX_OUTPUT_LIMIT_CONTINUATIONS_PER_RUN,
    OUTPUT_LIMIT_NUDGE,
)
from tau_coding.reasoning_guard import (
    BUDGET_CUT_REPLY_SHARE,
    DEFAULT_THINKING_CHARS_PER_TOKEN,
    TURN_REASONING_TOKENS_ENV,
    request_reasoning_chars,
)
from test_session_output_limit import (
    LOOP_OPENING,
    LOOP_ROUND,
    _acts,
    _answers,
    _nudges,
    _session,
    _streamed_reasoning,
    _truncated,
)

REPORT_REQUEST = "Your last message was empty. Reply now with your final report."


async def _run_with_report_request(session: CodingSession) -> None:
    """Run a prompt, steering a report request in at the first turn's end."""
    requested = False
    async for event in session.prompt("Design it."):
        if getattr(event, "type", None) == "turn_end" and not requested:
            requested = True
            session.queue_steering_message(REPORT_REQUEST)


@pytest.mark.anyio
async def test_nudges_in_a_row_stay_bounded_with_a_steering_message_beside_one(
    tmp_path: Path,
) -> None:
    provider = FakeProvider([_truncated() for _ in range(8)])
    session = await _session(tmp_path, provider)

    await _run_with_report_request(session)

    # The reply to the report request is cut off while the nudge is still
    # queued: no second nudge goes in behind it, and the bound holds.
    assert len(_nudges(session)) == MAX_OUTPUT_LIMIT_CONTINUATIONS
    assert len(provider.calls) == 4


async def _thinking_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, main: FakeProvider, built: list[object]
) -> CodingSession:
    """A session on `main` at thinking medium that records each provider built."""
    isolate_home(monkeypatch, tmp_path)
    provider_config = OpenAICompatibleProviderConfig(
        name="local",
        models=("fake",),
        default_model="fake",
        thinking_levels=("off", "low", "medium"),
        thinking_models=("fake",),
        thinking_default="medium",
        thinking_parameter="reasoning_effort",
    )

    def create(*args: object, thinking_level: object, **kwargs: object) -> FakeProvider:
        built.append(thinking_level)
        return main

    monkeypatch.setattr(coding_session_module, "_create_runtime_provider", create)
    session = await _session(
        tmp_path,
        main,
        provider_name="local",
        provider_settings=ProviderSettings(providers=(provider_config,)),
        runtime_provider_config=provider_config,
    )
    built.clear()
    return session


@pytest.mark.anyio
async def test_nudge_after_a_steering_reply_keeps_the_session_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    main = FakeProvider([_truncated(), _answers("Reported."), _answers("Acted on the nudge.")])
    built: list[object] = []
    session = await _thinking_session(monkeypatch, tmp_path, main, built)

    await _run_with_report_request(session)

    # The report request's reply goes first and settles the turn, so the nudge
    # is withdrawn; the reply used the session's provider at its thinking level.
    assert [call[2][-1].text for call in main.calls] == ["Design it.", REPORT_REQUEST]
    assert built == []
    assert session.messages[-1].text == "Reported."
    assert session.thinking_level == "medium"


@pytest.mark.anyio
async def test_loop_cut_nudge_keeps_the_session_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Changing the thinking level for a nudge would re-render the prompt prefix.
    looping = LOOP_OPENING + LOOP_ROUND * 40
    never_reached = AssistantMessage(content="The loop ran to its end.")
    main = FakeProvider(
        [_streamed_reasoning(looping, then=never_reached), _answers("Acted on the nudge.")]
    )
    built: list[object] = []
    session = await _thinking_session(monkeypatch, tmp_path, main, built)

    async for _event in session.prompt("Design it."):
        pass

    cut = session.messages[1]
    assert isinstance(cut, AssistantMessage)
    assert [diagnostic.type for diagnostic in cut.diagnostics or []] == ["reasoning_loop_cut"]
    assert [call[2][-1].text for call in main.calls] == ["Design it.", LOOP_CUT_NUDGE]
    assert built == []
    assert session.messages[-1].text == "Acted on the nudge."
    assert session.thinking_level == "medium"


@pytest.mark.anyio
async def test_output_limit_cut_nudge_keeps_the_session_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    main = FakeProvider([_truncated(), _answers("Acted on the nudge.")])
    built: list[object] = []
    session = await _thinking_session(monkeypatch, tmp_path, main, built)

    async for _event in session.prompt("Design it."):
        pass

    cut = session.messages[1]
    assert isinstance(cut, AssistantMessage)
    assert cut.stop_reason == "length"
    assert not cut.diagnostics
    assert [call[2][-1].text for call in main.calls] == ["Design it.", OUTPUT_LIMIT_NUDGE]
    assert built == []
    assert session.messages[-1].text == "Acted on the nudge."
    assert session.thinking_level == "medium"


@pytest.mark.anyio
async def test_reasoning_past_the_turn_budget_is_cut_and_asked_to_act(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # With a budget set, the turn is cut and its
    # reasoning kept for the nudge to build on.
    monkeypatch.setenv(TURN_REASONING_TOKENS_ENV, "5")
    long = " ".join(f"Byte {n} holds bits {n * 7} to {n * 7 + 6}." for n in range(40))
    never_reached = AssistantMessage(content="The reasoning ran to its end.")
    main = FakeProvider(
        [_streamed_reasoning(long, then=never_reached), _answers("Acted on the nudge.")]
    )
    built: list[object] = []
    session = await _thinking_session(monkeypatch, tmp_path, main, built)

    async for _event in session.prompt("Encode it."):
        pass

    cut = session.messages[1]
    assert isinstance(cut, AssistantMessage)
    assert cut.stop_reason == "length"
    assert [diagnostic.type for diagnostic in cut.diagnostics or []] == ["reasoning_budget_cut"]
    assert [call[2][-1].text for call in main.calls] == ["Encode it.", BUDGET_CUT_NUDGE]
    replayed = main.calls[1][2][1]
    assert isinstance(replayed, AssistantMessage)
    assert any(isinstance(block, ThinkingContent) for block in replayed.content)
    assert session.messages[-1].text == "Acted on the nudge."


@pytest.mark.anyio
async def test_reasoning_has_no_token_budget_unless_the_run_sets_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(TURN_REASONING_TOKENS_ENV, raising=False)
    long = " ".join(f"Byte {n} holds bits {n * 7} to {n * 7 + 6}." for n in range(40))
    main = FakeProvider([_streamed_reasoning(long, then=AssistantMessage(content="Done."))])
    built: list[object] = []
    session = await _thinking_session(monkeypatch, tmp_path, main, built)

    async for _event in session.prompt("Encode it."):
        pass

    assert session.messages[-1].text == "Done."
    assert len(main.calls) == 1


@pytest.mark.anyio
async def test_a_budget_cut_nudges_reply_reasons_on_a_quarter_budget_four_times_in_a_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A reply that will act tends to do so early, so a nudge's reply has a quarter
    # of the budget, and a budget cut allows four nudges in a row. The nudges
    # after the first in a row ask for the write.
    monkeypatch.setenv(TURN_REASONING_TOKENS_ENV, "40")
    long = " ".join(f"Byte {n} holds bits {n * 7} to {n * 7 + 6}." for n in range(40))
    never_reached = AssistantMessage(content="The reasoning ran to its end.")
    main = FakeProvider(
        [*[_streamed_reasoning(long, then=never_reached) for _ in range(4)], _answers("Acted.")]
    )
    built: list[object] = []
    session = await _thinking_session(monkeypatch, tmp_path, main, built)

    async for _event in session.prompt("Encode it."):
        pass

    assert [call[2][-1].text for call in main.calls] == [
        "Encode it.",
        BUDGET_CUT_NUDGE,
        *[BUDGET_CUT_WRITE_NUDGE] * 3,
    ]
    cuts = [m for m in session.messages if isinstance(m, AssistantMessage) and m.diagnostics]
    reasoned = [
        sum(len(b.thinking) for b in m.content if isinstance(b, ThinkingContent)) for m in cuts
    ]
    assert len(reasoned) == 4 and all(r * 3 < reasoned[0] for r in reasoned[1:]), reasoned
    assert session.messages[-1].text == "Acted."


LONG_REASONING = " ".join(f"Byte {n} holds bits {n * 7} to {n * 7 + 6}." for n in range(40))


def _reasons_to_the_cut() -> list[object]:
    """A response that reasons past any budget the tests set, never acting."""
    never_reached = AssistantMessage(content="The reasoning ran to its end.")
    return _streamed_reasoning(LONG_REASONING, then=never_reached)


async def _timed_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, main: FakeProvider
) -> CodingSession:
    """A session on `main` with a 40-token reasoning budget."""
    monkeypatch.setenv(TURN_REASONING_TOKENS_ENV, "40")
    return await _thinking_session(monkeypatch, tmp_path, main, [])


def _sent_nudges(main: FakeProvider) -> list[str]:
    """The nudges the session sent, in order."""
    nudges = {BUDGET_CUT_NUDGE, BUDGET_CUT_WRITE_NUDGE, OUTPUT_LIMIT_NUDGE, LOOP_CUT_NUDGE}
    return [call[2][-1].text for call in main.calls if call[2][-1].text in nudges]


def _reasoned(session: CodingSession) -> list[int]:
    """The reasoning length of each budget-cut turn, in order."""
    return [
        sum(len(block.thinking) for block in message.content if isinstance(block, ThinkingContent))
        for message in session.messages
        if isinstance(message, AssistantMessage) and message.diagnostics
    ]


@pytest.mark.anyio
async def test_write_nudge_follows_the_first_budget_cut_nudge_in_a_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The first nudge in a row stays; the next asks for the write.
    main = FakeProvider([_reasons_to_the_cut(), _reasons_to_the_cut(), _answers("Wrote it.")])
    session = await _timed_session(monkeypatch, tmp_path, main)

    async for _event in session.prompt("Encode it."):
        pass

    assert _sent_nudges(main) == [BUDGET_CUT_NUDGE, BUDGET_CUT_WRITE_NUDGE]
    assert session.messages[-1].text == "Wrote it."


@pytest.mark.anyio
async def test_write_nudge_reply_reasons_on_the_budget_cut_reply_share(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(TURN_REASONING_TOKENS_ENV, "40")
    # 40 tokens are 100 characters at the unmeasured thinking ratio.
    chars = int(40 * DEFAULT_THINKING_CHARS_PER_TOKEN)
    for nudge in (BUDGET_CUT_NUDGE, BUDGET_CUT_WRITE_NUDGE):
        assert request_reasoning_chars([UserMessage(content=nudge)]) == int(
            chars * BUDGET_CUT_REPLY_SHARE
        )
    assert request_reasoning_chars([UserMessage(content="Encode it.")]) == chars

    main = FakeProvider([_reasons_to_the_cut() for _ in range(3)] + [_answers("Wrote it.")])
    session = await _timed_session(monkeypatch, tmp_path, main)
    async for _event in session.prompt("Encode it."):
        pass

    # The full turn, then the BUDGET_CUT_NUDGE reply and the write nudge's reply
    # on the same reduced share.
    first, after_budget_cut, after_write = _reasoned(session)
    assert after_budget_cut * 3 < first
    assert after_write == after_budget_cut


@pytest.mark.anyio
async def test_write_nudge_keeps_the_budget_cut_bound_in_a_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cuts = MAX_BUDGET_CUT_CONTINUATIONS + 1
    main = FakeProvider([_reasons_to_the_cut() for _ in range(cuts)] + [_answers("Unsent.")])
    session = await _timed_session(monkeypatch, tmp_path, main)

    async for _event in session.prompt("Encode it."):
        pass

    assert MAX_BUDGET_CUT_CONTINUATIONS == 4
    assert _sent_nudges(main) == [
        BUDGET_CUT_NUDGE,
        *[BUDGET_CUT_WRITE_NUDGE] * (MAX_BUDGET_CUT_CONTINUATIONS - 1),
    ]
    assert len(main.calls) == cuts


@pytest.mark.anyio
async def test_write_nudge_keeps_the_bound_per_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Each row: a cut, two nudges whose replies are cut, and one whose reply
    # acts. The sixth reply without a tool call, in the third row, ends the run.
    row = [_reasons_to_the_cut(), _reasons_to_the_cut(), _reasons_to_the_cut(), _acts()]
    main = FakeProvider([*row * 3, _answers("Unsent.")])
    session = await _timed_session(monkeypatch, tmp_path, main)

    async for _event in session.prompt("Encode it."):
        pass

    assert MAX_OUTPUT_LIMIT_CONTINUATIONS_PER_RUN == 6
    assert _sent_nudges(main) == [
        *[BUDGET_CUT_NUDGE, BUDGET_CUT_WRITE_NUDGE, BUDGET_CUT_WRITE_NUDGE] * 2,
        BUDGET_CUT_NUDGE,
        BUDGET_CUT_WRITE_NUDGE,
    ]
    assert len(main.calls) == 11


@pytest.mark.anyio
async def test_write_nudge_row_starts_over_after_a_turn_that_acts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    main = FakeProvider(
        [
            _reasons_to_the_cut(),
            _reasons_to_the_cut(),
            _acts(),
            _reasons_to_the_cut(),
            _answers("Wrote it."),
        ]
    )
    session = await _timed_session(monkeypatch, tmp_path, main)

    async for _event in session.prompt("Encode it."):
        pass

    assert _sent_nudges(main) == [BUDGET_CUT_NUDGE, BUDGET_CUT_WRITE_NUDGE, BUDGET_CUT_NUDGE]
    assert session.messages[-1].text == "Wrote it."


@pytest.mark.anyio
async def test_write_nudge_needs_two_budget_cuts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Only budget cuts count toward the write nudge: it needs two of them in a row.
    main = FakeProvider(
        [_truncated(), _reasons_to_the_cut(), _reasons_to_the_cut(), _answers("Wrote it.")]
    )
    session = await _timed_session(monkeypatch, tmp_path, main)

    async for _event in session.prompt("Encode it."):
        pass

    assert _sent_nudges(main) == [OUTPUT_LIMIT_NUDGE, BUDGET_CUT_NUDGE, BUDGET_CUT_WRITE_NUDGE]
    assert session.messages[-1].text == "Wrote it."


def test_write_nudge_asks_for_a_partial_write_and_run_in_neutral_words() -> None:
    words = BUDGET_CUT_WRITE_NUDGE.lower()
    # Stop working it out; the next reply is a tool call that writes what it
    # has, partial or not, to the file it is working on, then runs it.
    for asks in (
        "do not think it through again",
        "no more reasoning",
        "reply now with only a tool call",
    ):
        assert asks in words
    for asks in ("writes what you have so far", "partial", "the file you are working on", "run it"):
        assert asks in words
    # Task- and language-neutral: no language, tool, codec or test framework.
    for named in ("python", "pytest", "bash", "encode", "module", "test", ".py"):
        assert named not in words, named


@pytest.mark.anyio
async def test_a_budget_cut_turn_records_estimated_usage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A cut stream never gets the server's closing usage, so the turn read as
    # free: its request and reasoning are estimated instead.
    monkeypatch.setenv(TURN_REASONING_TOKENS_ENV, "5")
    long = " ".join(f"Byte {n} holds bits {n * 7} to {n * 7 + 6}." for n in range(40))
    never_reached = AssistantMessage(content="The reasoning ran to its end.")
    main = FakeProvider(
        [_streamed_reasoning(long, then=never_reached), _answers("Acted on the nudge.")]
    )
    session = await _thinking_session(monkeypatch, tmp_path, main, [])

    async for _event in session.prompt("Encode it."):
        pass

    cut = session.messages[1]
    assert isinstance(cut, AssistantMessage)
    assert cut.usage.input > 0 and cut.usage.output > 0
    assert cut.usage.total_tokens == cut.usage.input + cut.usage.output
    [diagnostic] = cut.diagnostics or []
    assert diagnostic.details is not None and diagnostic.details["usage_estimated"] is True


def test_a_cut_keeps_the_usage_the_provider_reported() -> None:
    reported = Usage(input=0, cache_read=900, output=400, total_tokens=1300)
    partial = AssistantMessage(content=[ThinkingContent(thinking="short")], usage=reported)

    assert reasoning_guard_module._cut_usage(partial, lambda: 5_000) is reported
