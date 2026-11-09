"""A tool call cut off at the output limit is answered, not run.

A model can fall into repeating one block of text inside a tool call's
arguments. The stream is stopped once that is plain, and the call is answered
as cut off instead of being sent on.
"""

import json
from collections.abc import Mapping

import httpx
import pytest

from pi_event_helpers import assistant_start, text_delta, tool_call_end
from tau_agent import (
    AgentEvent,
    AgentMessage,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from tau_agent.loop import run_agent_loop
from tau_agent.messages import TOOL_CALL_LOOP_CUT
from tau_agent.provider_events import AssistantDoneEvent
from tau_agent.types import JSONValue
from tau_ai import CancellationToken, FakeProvider, OpenAICompatibleConfig, OpenAICompatibleProvider
from tau_ai.openai_compatible import repeating_tail

# A real start, then one block of text over and over.
REAL_START = '{"edits":[{"newText":"import pytest\\nfrom pkg import thing\\n' + "".join(
    f"\\ndef test_case_{n}():\\n    assert thing({n}) == {n * n}\\n" for n in range(60)
)
REPORT = (
    '```json\\n{\\n  \\"status\\": \\"done\\",\\n  \\"files_changed\\": '
    '[\\"tests/test_thing.py\\"],\\n  \\"issues_found\\": [\\n    \\"The code takes '
    'positional arguments instead of parsing the string.\\"\\n  ]\\n}\\n```\\n'
    "**Cases written:** one for each input, named after what it checks.\\n"
    + "".join(
        f"- `test_case_{n}`: checks that input {n} gives its expected value.\\n" for n in range(14)
    )
)


def _sse(chunks: list[Mapping[str, object]]) -> str:
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


def _tool_call_chunks(
    name: str, argument_parts: list[str], finish_reason: str
) -> list[Mapping[str, object]]:
    head = {"index": 0, "id": "call-1", "type": "function", "function": {"name": name}}
    chunks: list[Mapping[str, object]] = [{"choices": [{"delta": {"tool_calls": [head]}}]}]
    chunks += [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": part}}]}}]}
        for part in argument_parts
    ]
    chunks.append({"choices": [{"delta": {}, "finish_reason": finish_reason}]})
    return chunks


def _text_chunks(text: str) -> list[Mapping[str, object]]:
    return [{"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]}]


class _Server:
    """Answers each request with the next SSE body and counts what it sent."""

    def __init__(self, bodies: list[list[Mapping[str, object]]]) -> None:
        self.bodies = bodies
        self.requests: list[dict[str, JSONValue]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            text=_sse(self.bodies.pop(0)),
            headers={"content-type": "text/event-stream"},
        )


def _recording_tool(name: str, ran: list[Mapping[str, JSONValue]]) -> AgentTool:
    async def execute(
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: CancellationToken | None = None,
        on_update=None,  # noqa: ANN001
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        ran.append(dict(arguments))
        return AgentToolResult(content=[TextContent(text="ok")], details={})

    return AgentTool(
        name=name,
        label=name.title(),
        description=f"{name} a file",
        parameters={"type": "object"},
        execute_fn=execute,
    )


async def _run(
    server: _Server, tools: list[AgentTool]
) -> tuple[list[AgentEvent], list[AgentMessage]]:
    messages: list[AgentMessage] = [UserMessage(content="Write the cases.")]
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="k", base_url="https://example.test/v1"),
            client=client,
        )
        events = [
            event
            async for event in run_agent_loop(
                provider=provider,
                model="m",
                system="You are Tau.",
                messages=messages,
                tools=tools,
            )
        ]
    return events, messages


def _cut_parts(text: str, size: int = 64) -> list[str]:
    return [text[at : at + size] for at in range(0, len(text), size)]


@pytest.mark.anyio
async def test_output_cap_tool_call_is_answered_with_next_step_not_run() -> None:
    """A call cut at the output limit is not run, and the next request tells the
    model what happened and to write in smaller parts."""
    ran: list[Mapping[str, JSONValue]] = []
    cut = REAL_START[:3000]  # ends inside a string: what the limit leaves
    server = _Server([_tool_call_chunks("edit", _cut_parts(cut), "length"), _text_chunks("ok")])

    events, messages = await _run(server, [_recording_tool("edit", ran)])

    assert ran == []
    assistant = messages[1]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.stop_reason == "length"
    result = messages[2]
    assert isinstance(result, ToolResultMessage)
    assert result.is_error
    assert result.text == (
        "This edit call was cut off: the reply reached the output limit after 3000 "
        "characters of its arguments. It did not run, and nothing it would have done "
        "was done. Send less in one call: to write or edit a file, write it in smaller "
        "parts, one call per part."
    )
    # The note reaches the model in the next request, answering the cut call.
    sent = server.requests[1]["messages"]
    assert isinstance(sent, list)
    assert any(isinstance(m, dict) and m.get("content") == result.text for m in sent)
    assert any(event.type == "tool_execution_end" for event in events)


@pytest.mark.anyio
async def test_output_cap_tool_call_repeating_arguments_end_the_stream_early() -> None:
    """Arguments that repeat one block end the response there, as a cut
    at the limit, long before the cap."""
    ran: list[Mapping[str, JSONValue]] = []
    runaway = REAL_START + REPORT * 80
    server = _Server([_tool_call_chunks("edit", _cut_parts(runaway), "length"), _text_chunks("ok")])

    _, messages = await _run(server, [_recording_tool("edit", ran)])

    assert ran == []
    assistant = messages[1]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.stop_reason == "length"
    assert [d.type for d in assistant.diagnostics or []] == [TOOL_CALL_LOOP_CUT]
    sent = assistant.tool_calls[0].arguments["_raw_arguments"]
    assert isinstance(sent, str)
    assert len(REAL_START) < len(sent) < len(REAL_START) + 20 * len(REPORT) < len(runaway) // 3
    result = messages[2]
    assert isinstance(result, ToolResultMessage)
    assert result.text.startswith(
        "This edit call was cut off: its arguments started repeating the same text "
        f"over and over, so the reply was stopped after {len(sent)} characters. "
        "It did not run"
    )
    assert result.text.endswith(
        "If the file really holds this much repeated text, generate it with a short "
        "script instead of writing it out. Otherwise send it again without the "
        "repetition, and send less in one call: to write or edit a file, write it in "
        "smaller parts, one call per part."
    )


@pytest.mark.anyio
async def test_output_cap_tool_call_long_real_arguments_run_unchanged() -> None:
    """A long call that ends normally, its text varied, runs as before."""
    ran: list[Mapping[str, JSONValue]] = []
    content = "".join(f"line {n}: value {n * 7919 % 10007}\n" for n in range(1500))
    arguments = json.dumps({"path": "a.txt", "content": content})
    assert repeating_tail(arguments) is None
    server = _Server(
        [_tool_call_chunks("write", _cut_parts(arguments), "tool_calls"), _text_chunks("ok")]
    )

    _, messages = await _run(server, [_recording_tool("write", ran)])

    assert ran == [{"path": "a.txt", "content": content}]
    assistant = messages[1]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.stop_reason == "toolUse"
    assert not assistant.diagnostics


def test_output_cap_tool_call_repeating_tail_needs_a_repeated_block() -> None:
    """The detector names the repeated block; a real start alone, or a block seen
    fewer than sixteen times, is not a loop."""
    assert repeating_tail(REAL_START) is None
    assert repeating_tail(REAL_START + REPORT * 15) is None
    found = repeating_tail(REAL_START + REPORT * 16)
    assert found is not None
    assert found["period"] == len(REPORT)


@pytest.mark.parametrize(
    "content",
    [
        "x = 1\n" + " " * 4_100 + "\n",
        "id,name,qty\n" + "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n" * 150,
        "| a | b |\n|---|---|\n" + "| 0 | 0 |\n" * 420,
        "rows = [\n" + "    (0, 0, 0, 0, 0, 0, 0, 0),\n" * 400 + "]\n",
    ],
    ids=["padding", "csv-rows", "table-rows", "literal-rows"],
)
def test_output_cap_tool_call_repetitive_files_are_not_loops(content: str) -> None:
    """Files that repeat for a few thousand characters are written, not cut."""
    assert repeating_tail(json.dumps({"path": "f", "content": content})[:-2]) is None


@pytest.mark.anyio
async def test_output_cap_tool_call_earlier_complete_calls_still_run() -> None:
    """Only the call the limit cut is held back; complete calls before it run."""
    ran: list[Mapping[str, JSONValue]] = []
    done = ToolCall(id="call-0", name="read", arguments={"path": "a.txt"})
    cut = ToolCall(id="call-1", name="read", arguments={"_raw_arguments": '{"path": "b.t'})
    first = AssistantMessage(content=[done, cut], model="fake", stop_reason="length")
    provider = FakeProvider(
        [
            [
                assistant_start(),
                tool_call_end(done),
                tool_call_end(cut),
                AssistantDoneEvent(reason="length", message=first),
            ],
            [assistant_start(), text_delta("ok")],
        ]
    )
    messages: list[AgentMessage] = [UserMessage(content="Read both.")]
    _ = [
        event
        async for event in run_agent_loop(
            provider=provider,
            model="fake",
            system="You are Tau.",
            messages=messages,
            tools=[_recording_tool("read", ran)],
        )
    ]

    assert ran == [{"path": "a.txt"}]
    results = [m for m in messages if isinstance(m, ToolResultMessage)]
    assert [(r.tool_call_id, r.is_error) for r in results] == [
        ("call-0", False),
        ("call-1", True),
    ]
    assert "cut off" in results[1].text and "13 characters" in results[1].text
