"""A reply cut off at the output limit before it acted is asked to act."""

from pathlib import Path

import pytest

from pi_event_helpers import assistant_done, assistant_start
from tau_agent import (
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    UserMessage,
)
from tau_agent.loop import provider_context
from tau_agent.session import JsonlSessionStorage
from tau_ai import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig
from tau_coding.guards import MAX_OUTPUT_LIMIT_CONTINUATIONS, OUTPUT_LIMIT_NUDGE

CALL = ToolCall(id="call-1", name="fetch", arguments={})
OTHER_REQUEST = "Your last message was empty. Reply now with your final report."


def _fetch_tool() -> AgentTool:
    async def fetch(
        tool_call_id: object, arguments: object, signal: object = None, on_update: object = None
    ) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="fetched")])

    return AgentTool(
        name="fetch",
        label="Fetch",
        description="Fetch a document.",
        parameters={"type": "object"},
        execute_fn=fetch,
    )


def _acts() -> list[object]:
    return [assistant_start(), assistant_done(message=AssistantMessage(content=[CALL]))]


def _answers() -> list[object]:
    return [assistant_start(), assistant_done(message=AssistantMessage(content="Done."))]


def _truncated() -> list[object]:
    return [
        assistant_start(),
        assistant_done(
            message=AssistantMessage(content=[ThinkingContent(thinking="Hmm. " * 50)]),
            finish_reason="length",
        ),
    ]


async def _session(tmp_path: Path, provider: FakeProvider, **config: object) -> CodingSession:
    return await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            tools=[_fetch_tool()],
            **config,  # type: ignore[arg-type]
        )
    )


async def _run(session: CodingSession) -> None:
    async for _event in session.prompt("Design it."):
        pass


def _nudges(session: CodingSession) -> list[UserMessage]:
    return [
        message
        for message in session.messages
        if isinstance(message, UserMessage) and message.text == OUTPUT_LIMIT_NUDGE
    ]


@pytest.mark.anyio
async def test_a_cut_off_reply_is_asked_to_act(tmp_path: Path) -> None:
    provider = FakeProvider([_truncated(), _acts(), _answers()])
    session = await _session(tmp_path, provider)

    await _run(session)

    assert len(_nudges(session)) == 1
    assert session.messages[-1].text == "Done."


@pytest.mark.anyio
async def test_the_in_a_row_bound_stops_a_model_that_never_acts(tmp_path: Path) -> None:
    provider = FakeProvider([_truncated() for _ in range(5)])
    session = await _session(tmp_path, provider)

    await _run(session)

    assert len(_nudges(session)) == MAX_OUTPUT_LIMIT_CONTINUATIONS
    assert len(provider.calls) == 3


@pytest.mark.anyio
async def test_the_per_run_bound_skips_nudges_whose_reply_acted(tmp_path: Path) -> None:
    provider = FakeProvider([*[_truncated(), _acts()] * 8, _answers()])
    session = await _session(tmp_path, provider)

    await _run(session)

    assert len(_nudges(session)) == 8
    assert session.messages[-1].text == "Done."


@pytest.mark.anyio
async def test_the_bound_holds_with_a_steering_message_beside_a_nudge(tmp_path: Path) -> None:
    # The steering message takes the nudge's request, and the reply to it is
    # cut off while the nudge still waits: no second nudge goes in behind it.
    provider = FakeProvider([_truncated() for _ in range(8)])
    session = await _session(tmp_path, provider)

    requested = False
    async for event in session.prompt("Design it."):
        if getattr(event, "type", None) == "turn_end" and not requested:
            requested = True
            session.queue_steering_message(OTHER_REQUEST)

    assert len(_nudges(session)) == MAX_OUTPUT_LIMIT_CONTINUATIONS
    assert len(provider.calls) == 4


@pytest.mark.anyio
async def test_a_nudge_the_run_ended_on_is_withdrawn(tmp_path: Path) -> None:
    provider = FakeProvider([_acts(), _truncated()])
    session = await _session(tmp_path, provider, max_turns=1)

    await _run(session)

    assert session.queued_message_count == 0
    assert _nudges(session) == []


def test_an_empty_cut_off_reply_is_not_replayed() -> None:
    empty = AssistantMessage(content=[], stop_reason="length")
    kept = AssistantMessage(content=[ThinkingContent(thinking="so far")], stop_reason="length")

    assert provider_context([empty, kept]) == [kept]


@pytest.mark.anyio
async def test_the_per_run_bound_stops_asks_that_go_unanswered(tmp_path: Path) -> None:
    # Each cycle has one reply to a nudge that is cut off again; the sixth such
    # reply ends the asking, even though the in-a-row count has started over.
    provider = FakeProvider([*[_truncated(), _truncated(), _acts()] * 6, _answers()])
    session = await _session(tmp_path, provider)

    await _run(session)

    assert len(_nudges(session)) == 11
    assert len(provider.calls) == 17


@pytest.mark.anyio
async def test_a_nudge_is_withdrawn_when_another_reply_settles_the_turn(tmp_path: Path) -> None:
    provider = FakeProvider([_truncated(), _acts(), _answers()])
    session = await _session(tmp_path, provider)

    requested = False
    async for event in session.prompt("Design it."):
        if getattr(event, "type", None) == "turn_end" and not requested:
            requested = True
            session.queue_steering_message(OTHER_REQUEST)

    assert _nudges(session) == []
    assert session.messages[-1].text == "Done."
    assert len(provider.calls) == 3
