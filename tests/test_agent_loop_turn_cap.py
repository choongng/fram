from collections.abc import AsyncIterator, Mapping

import pytest

from pi_event_helpers import assistant_done, assistant_start, text_delta, tool_call_end
from tau_agent import (
    AgentEvent,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    SimpleCancellationToken,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from tau_agent.loop import run_agent_loop
from tau_agent.types import JSONValue
from tau_ai import FakeProvider


async def _collect(stream: AsyncIterator[AgentEvent]) -> list[AgentEvent]:
    return [event async for event in stream]


async def _noop(
    _call_id: str, _args: Mapping[str, JSONValue], _signal: object, _update: object
) -> AgentToolResult:
    return AgentToolResult(content="ok")


def _tool() -> AgentTool:
    return AgentTool(
        name="poke",
        label="Poke",
        description="Run poke.",
        parameters={"type": "object"},
        execute_fn=_noop,
    )


def _tool_turn(index: int) -> list:
    call = ToolCall(id=f"call-{index}", name="poke", arguments={})
    assistant = AssistantMessage(content=[call], model="fake")
    return [assistant_start(), tool_call_end(call), assistant_done(assistant, "toolUse")]


def _text_turn(text: str) -> list:
    assistant = AssistantMessage(content=text, model="fake")
    return [assistant_start(), text_delta(text), assistant_done(assistant)]


def _user_texts(messages: list[AgentMessage]) -> list[str]:
    return [message.text for message in messages if isinstance(message, UserMessage)]


@pytest.mark.anyio
async def test_turn_cap_warns_before_last_turn() -> None:
    max_turns = 4
    provider = FakeProvider([_tool_turn(i) for i in range(max_turns)] + [_text_turn("report")])
    messages: list[AgentMessage] = [UserMessage(content="work")]

    await _collect(
        run_agent_loop(
            provider=provider,
            model="fake",
            system="You are Tau.",
            messages=messages,
            tools=[_tool()],
            max_turns=max_turns,
            report_at_max_turns=True,
        )
    )

    # The warning reaches the request for turn 3 of 4 (two turns left), not before.
    warning = "2 turns left: finish what you are doing and write your report."
    seen = [_user_texts(call[2]) for call in provider.calls]
    assert [warning in texts for texts in seen] == [False, False, True, True, True]
    assert sum(text.startswith("2 turns left") for text in _user_texts(messages)) == 1


@pytest.mark.anyio
async def test_turn_cap_ends_with_report_not_error() -> None:
    max_turns = 3
    # The model keeps calling tools, even in the final turn; that call is not run.
    provider = FakeProvider([_tool_turn(i) for i in range(max_turns + 1)])
    executed: list[str] = []

    async def record(
        call_id: str, _args: Mapping[str, JSONValue], _signal: object, _update: object
    ) -> AgentToolResult:
        executed.append(call_id)
        return AgentToolResult(content="ok")

    tool = AgentTool(
        name="poke",
        label="Poke",
        description="Run poke.",
        parameters={"type": "object"},
        execute_fn=record,
    )
    messages: list[AgentMessage] = [UserMessage(content="work")]

    await _collect(
        run_agent_loop(
            provider=provider,
            model="fake",
            system="You are Tau.",
            messages=messages,
            tools=[tool],
            max_turns=max_turns,
            report_at_max_turns=True,
        )
    )

    assert len(provider.calls) == max_turns + 1
    # Tool definitions stay on the report request: some providers reject tool
    # history without them.
    assert [len(call[3]) for call in provider.calls] == [1, 1, 1, 1]
    assert executed == ["call-0", "call-1", "call-2"]
    assert _user_texts(provider.calls[-1][2])[-1].startswith("Turn limit reached (3 used)")
    # The unrun call is answered, so the history replays cleanly on resume.
    refusal = messages[-1]
    assert isinstance(refusal, ToolResultMessage)
    assert refusal.is_error is True
    assert refusal.text == "Not run: the turn limit for this run was reached."
    last = messages[-2]
    assert isinstance(last, AssistantMessage)
    assert last.stop_reason != "error"
    assert last.error_message is None

    # A text report in the final turn is what the run hands back.
    provider = FakeProvider([_tool_turn(0), _text_turn("did X; Y is left")])
    messages = [UserMessage(content="work")]
    await _collect(
        run_agent_loop(
            provider=provider,
            model="fake",
            system="You are Tau.",
            messages=messages,
            tools=[_tool()],
            max_turns=1,
            report_at_max_turns=True,
        )
    )
    last = messages[-1]
    assert isinstance(last, AssistantMessage)
    assert last.text == "did X; Y is left"
    assert last.stop_reason == "stop"


@pytest.mark.anyio
async def test_turn_cap_without_report_mode_still_stops_with_an_error() -> None:
    provider = FakeProvider([_tool_turn(0), _tool_turn(1)])
    messages: list[AgentMessage] = [UserMessage(content="work")]

    await _collect(
        run_agent_loop(
            provider=provider,
            model="fake",
            system="You are Tau.",
            messages=messages,
            tools=[_tool()],
            max_turns=1,
        )
    )

    assert len(provider.calls) == 1
    last = messages[-1]
    assert isinstance(last, AssistantMessage)
    assert last.stop_reason == "error"
    assert _user_texts(messages) == ["work"]


@pytest.mark.anyio
async def test_turn_cap_of_two_has_no_warning_turn() -> None:
    provider = FakeProvider([_tool_turn(0), _tool_turn(1), _text_turn("report")])
    messages: list[AgentMessage] = [UserMessage(content="work")]

    await _collect(
        run_agent_loop(
            provider=provider,
            model="fake",
            system="You are Tau.",
            messages=messages,
            tools=[_tool()],
            max_turns=2,
            report_at_max_turns=True,
        )
    )

    notes = _user_texts(messages)[1:]
    assert len(notes) == 1
    assert notes[0].startswith("Turn limit reached")


@pytest.mark.anyio
async def test_cancel_at_the_cap_adds_no_turn_limit_note() -> None:
    signal = SimpleCancellationToken()

    async def cancel(
        _call_id: str, _args: Mapping[str, JSONValue], _signal: object, _update: object
    ) -> AgentToolResult:
        signal.cancel()
        return AgentToolResult(content="ok")

    tool = AgentTool(
        name="poke",
        label="Poke",
        description="Run poke.",
        parameters={"type": "object"},
        execute_fn=cancel,
    )
    provider = FakeProvider([_tool_turn(0), _text_turn("report")])
    messages: list[AgentMessage] = [UserMessage(content="work")]

    await _collect(
        run_agent_loop(
            provider=provider,
            model="fake",
            system="You are Tau.",
            messages=messages,
            tools=[tool],
            max_turns=1,
            report_at_max_turns=True,
            signal=signal,
        )
    )

    assert len(provider.calls) == 1
    assert _user_texts(messages) == ["work"]
