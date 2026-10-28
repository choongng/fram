"""tau_agent checks tool-call arguments against the tool's declared schema."""

import asyncio
from collections.abc import AsyncIterator, Mapping

from pi_event_helpers import assistant_done, assistant_start, tool_call_end
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
from tau_agent.types import JSONValue
from tau_ai import FakeProvider

READ_SCHEMA: dict[str, JSONValue] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "limit": {"type": "integer"},
        "mode": {"type": "string", "enum": ["text", "bytes"]},
        "ranges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"start": {"type": "integer"}},
                "required": ["start"],
            },
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}


async def _run(
    schema: Mapping[str, JSONValue], arguments: dict[str, JSONValue]
) -> tuple[ToolResultMessage, list[Mapping[str, JSONValue]]]:
    """Run one `read` call with `arguments`; return its result and what the tool got."""
    seen: list[Mapping[str, JSONValue]] = []

    async def execute(tool_call_id, args, signal=None, on_update=None):  # noqa: ANN001, ANN202
        del tool_call_id, signal, on_update
        seen.append(args)
        return AgentToolResult(content=[TextContent(text="ran")], details={})

    tool = AgentTool(
        name="read",
        label="Read",
        description="Read a file.",
        parameters=schema,
        execute_fn=execute,
    )
    call = ToolCall(id="call-1", name="read", arguments=arguments)
    assistant = AssistantMessage(content=[call], model="fake")
    provider = FakeProvider(
        [[assistant_start(), tool_call_end(call), assistant_done(assistant, "toolUse")]]
    )
    messages: list[AgentMessage] = [UserMessage(content="Read it")]
    events: AsyncIterator[AgentEvent] = run_agent_loop(
        provider=provider,
        model="fake",
        system="You are Tau.",
        messages=messages,
        tools=[tool],
        max_turns=1,
    )
    _ = [event async for event in events]
    result = next(m for m in messages if isinstance(m, ToolResultMessage))
    return result, seen


def test_missing_required_argument_is_refused_before_the_tool_runs() -> None:
    result, seen = asyncio.run(_run(READ_SCHEMA, {"limit": 5}))

    assert seen == []
    assert result.is_error is True
    assert result.text == (
        'Invalid arguments for read: missing required "path". Send the call again with these fixed.'
    )


def test_one_error_lists_every_problem() -> None:
    result, seen = asyncio.run(
        _run(
            READ_SCHEMA,
            {"limit": "10", "mode": "lines", "colour": "red", "ranges": [{"start": 1}, {}]},
        )
    )

    assert seen == []
    assert result.is_error is True
    assert result.text == (
        "Invalid arguments for read: "
        'missing required "path"; '
        '"limit" must be integer, got string; '
        '"mode" must be one of "text", "bytes", got "lines"; '
        'unknown argument "colour"; '
        'missing required "ranges[1].start". '
        "Send the call again with these fixed."
    )


def test_wrong_type_and_unknown_enum_value_are_named() -> None:
    result, seen = asyncio.run(
        _run(READ_SCHEMA, {"path": "a.py", "limit": True, "mode": "pages", "ranges": "1-3"})
    )

    assert seen == []
    assert '"limit" must be integer, got boolean' in result.text
    assert '"mode" must be one of "text", "bytes", got "pages"' in result.text
    assert '"ranges" must be array, got string' in result.text


def test_valid_arguments_reach_the_tool_unchanged() -> None:
    arguments: dict[str, JSONValue] = {
        "path": "a.py",
        "limit": 10.0,
        "mode": "bytes",
        "ranges": [{"start": 2}],
    }

    result, seen = asyncio.run(_run(READ_SCHEMA, arguments))

    assert result.is_error is False
    assert result.text == "ran"
    assert seen == [arguments]
    assert isinstance(seen[0]["limit"], float)


def test_a_tool_with_no_schema_is_not_checked() -> None:
    result, seen = asyncio.run(_run({}, {"anything": [1, "two"], "path": 3}))

    assert result.is_error is False
    assert seen == [{"anything": [1, "two"], "path": 3}]


def test_unparsed_arguments_are_left_to_the_tool() -> None:
    result, seen = asyncio.run(_run(READ_SCHEMA, {"_raw_arguments": '{"path": "a.py'}))

    assert result.is_error is False
    assert seen == [{"_raw_arguments": '{"path": "a.py'}]
