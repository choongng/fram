"""Chat-template tool-call tokens: refused in arguments, followed up in text."""

from dataclasses import replace
from pathlib import Path

import pytest

from pi_event_helpers import assistant_done, assistant_start
from tau_agent import (
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    TextContent,
    ToolCall,
    UserMessage,
)
from tau_agent.messages import assistant_content
from tau_agent.session import JsonlSessionStorage
from tau_agent.types import JSONValue
from tau_ai import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig, TauResourcePaths, create_write_tool
from tau_coding.guards import TEXT_TOOL_CALL_NUDGE
from tau_coding.tools import create_bash_tool

#: The reply's next call and its reasoning, run into this call's last string.
LEAKED_SUMMARY = (
    "Add a command that reports disk use per directory and per user on this machine; "
    'it is tested where the output varies.",confirmed:false,urgent:true}'
    "<tool_call|><|channel>thought\nNow that the note is saved, I will "
    "start the first part.<channel|><|tool_call>call:bash{command:"
)
LEAKED_ESCAPED_DESCRIPTION = (
    'Run the known-answers test.\\"}<tool_call|><|channel>thought\n'
    "<channel|><|tool_call>call:bash{command:"
)
LEAKED_DESCRIPTION = (
    'Writing the marker file."}<tool_call|><|channel>thought\n'
    "<channel|><|tool_call>call:bash{command:"
)
#: The XML-style form, left at the end of a call's text.
XML_SPILL = "Report written.</tool_call>\n<tool_call>\n<function=bash>\n<parameter=command>"
#: A whole reply that wrote its call as text.
TEXT_CALL = (
    "<tool_call>\n<function=bash>\n<parameter=command>\ncd /tmp/project && "
    "python3 -m pytest -q tests/test_data.py --collect-only\n</parameter>\n"
    "</function>\n</tool_call>"
)
#: The same call with no opening `<tool_call>`.
UNOPENED_CALL = (
    "<function=bash><parameter=command>ls -la /tmp/project/</parameter>"
    "<parameter=description>List project structure</parameter></function></tool_call>"
)


def _turns(name: str, *arguments: dict[str, JSONValue]) -> list[list[object]]:
    turns: list[list[object]] = [
        [
            assistant_start(model="fake"),
            assistant_done(
                message=AssistantMessage(
                    content=assistant_content(
                        "Calling.", [ToolCall(id=f"call-{i}", name=name, arguments=args)]
                    ),
                    stop_reason="toolUse",
                )
            ),
        ]
        for i, args in enumerate(arguments)
    ]
    turns.append(
        [assistant_start(model="fake"), assistant_done(message=AssistantMessage(content="Done"))]
    )
    return turns


def _recording_tool(ran: list[object]) -> AgentTool:
    async def record(
        tool_call_id: object, arguments: object, signal: object = None, on_update: object = None
    ) -> AgentToolResult:
        ran.append(arguments)
        return AgentToolResult(content=[TextContent(text=f"ran {len(ran)}")])

    return AgentTool(
        name="note",
        label="Note",
        description="Record a note.",
        parameters={"type": "object"},
        execute_fn=record,
    )


async def _run(session: CodingSession) -> list[str]:
    async for _ in session.prompt("Go."):
        pass
    return [m.text for m in session.messages if m.role == "toolResult"]


async def _run_note(
    tmp_path: Path, *arguments: dict[str, JSONValue]
) -> tuple[list[object], list[str]]:
    ran: list[object] = []
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider(_turns("note", *arguments)),
            model="fake",
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            tools=[_recording_tool(ran)],
        )
    )
    return ran, await _run(session)


@pytest.mark.anyio
async def test_tool_args_with_template_tokens_refused(tmp_path: Path) -> None:
    plan: dict[str, JSONValue] = {"summary": LEAKED_SUMMARY, "urgent": True}
    nested: dict[str, JSONValue] = {
        "items": [{"title": "first", "description": LEAKED_ESCAPED_DESCRIPTION}]
    }

    ran, results = await _run_note(tmp_path, plan, nested, {"text": "plain"})

    assert ran == [{"text": "plain"}]
    assert results[0].startswith(
        "Not run: its arguments carry the chat template's own tool-call tokens "
        "(`<tool_call|>` in `summary`), so the reply's next call ran into this one; "
        "send one call per reply, with the text alone. If the text must really hold that "
        "token, split it across strings in a script that joins them. 7 more refused calls "
        "in a row stop the run."
    )
    # A string at any depth is read, and the refusals count toward the stop.
    assert "(`<tool_call|>` in `items[0].description`)" in results[1]
    assert "6 more refused calls in a row" in results[1]


@pytest.mark.anyio
async def test_tool_args_with_a_spilled_xml_call_refused(tmp_path: Path) -> None:
    json_form = 'Done.<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}'

    ran, results = await _run_note(tmp_path, {"text": XML_SPILL}, {"text": json_form})

    assert ran == []
    assert "(`<tool_call>` in `text`)" in results[0]
    assert "(`<tool_call>` in `text`)" in results[1]


@pytest.mark.anyio
async def test_extension_tool_args_with_template_tokens_refused(tmp_path: Path) -> None:
    # Extension tools take the same path.
    paths = TauResourcePaths(
        root=tmp_path / "home-tau", cwd=tmp_path / "project", agents_root=tmp_path / "agents"
    )
    assert paths.cwd is not None
    paths.cwd.mkdir(parents=True)
    log = tmp_path / "ran.log"
    extensions = paths.root / "extensions"
    extensions.mkdir(parents=True)
    (extensions / "notes.py").write_text(
        f"""
from pathlib import Path

from tau_agent.tools import AgentTool, AgentToolResult


async def _run(tool_call_id, arguments, signal=None, on_update=None):
    with Path({str(log)!r}).open("a") as log:
        log.write(arguments["summary"] + "\\n")
    return AgentToolResult(content="saved")


def setup(tau):
    tau.register_tool(
        AgentTool(
            name="save_note",
            label="Save",
            description="Save a note.",
            parameters={{
                "type": "object",
                "properties": {{"summary": {{"type": "string"}}}},
                "required": ["summary"],
            }},
            execute_fn=_run,
        )
    )
""",
        encoding="utf-8",
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider(
                _turns("save_note", {"summary": LEAKED_SUMMARY}, {"summary": "Clean goal."})
            ),
            model="fake",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=paths.cwd,
            resource_paths=paths,
        )
    )
    assert "save_note" in [tool.name for tool in session.tools]

    results = await _run(session)

    assert log.read_text() == "Clean goal.\n"
    assert results[0].startswith("Not run: its arguments carry the chat template's own")
    assert results[1] == "saved"


@pytest.mark.anyio
async def test_tool_args_with_ordinary_angle_brackets_run(tmp_path: Path) -> None:
    ordinary: list[str] = [
        '<html><body><p class="x">a</p></body></html>',
        '<?xml version="1.0"?><config><item name="a"/></config>',
        "List<String> names = new ArrayList<>();",
        "if a < b > c: pass",
        "The tool_call field names the channel the reply came on.",
        "Some templates wrap a call in <tool_call> tags; others use their own tokens.",
        "<|endoftext|> and <tool_call/> and <channel> are not the forms refused.",
    ]

    ran, results = await _run_note(tmp_path, *({"text": text} for text in ordinary))

    assert ran == [{"text": text} for text in ordinary]
    assert not any(r.startswith("Not run") for r in results)


@pytest.mark.anyio
async def test_argument_scan_accepts_unopened_call_markup_in_a_write(tmp_path: Path) -> None:
    # A file about XML-style call syntax (a parser, its docs or tests) holds the
    # reply-only shapes; only the scan's own tokens refuse a write.
    unopened = "<function=bash><parameter=command>ls</parameter></function></tool_call>"
    closing = "</function></tool_call>"
    writes: list[dict[str, JSONValue]] = [
        {"path": "parser_test.py", "content": unopened},
        {"path": "parser_test.py", "content": closing},
        {"path": "parser_test.py", "content": UNOPENED_CALL},
        {"path": "parser_test.py", "content": XML_SPILL},
    ]

    ran, results = await _run_note(tmp_path, *writes)

    assert ran == writes[:3]
    assert not any(r.startswith("Not run") for r in results[:3])
    assert "(`<tool_call>` in `content`)" in results[3]


def _says(text: str) -> list[object]:
    return [
        assistant_start(model="fake"),
        assistant_done(message=AssistantMessage(content=text, stop_reason="stop")),
    ]


def _calls_note(arguments: dict[str, JSONValue]) -> list[object]:
    call = ToolCall(id="call-real", name="note", arguments=arguments)
    return [
        assistant_start(model="fake"),
        assistant_done(
            message=AssistantMessage(
                content=assistant_content("Calling.", [call]), stop_reason="toolUse"
            )
        ),
    ]


async def _text_session(
    tmp_path: Path, turns: list[list[object]]
) -> tuple[CodingSession, FakeProvider, list[object]]:
    ran: list[object] = []
    provider = FakeProvider(turns)
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            tools=[_recording_tool(ran)],
        )
    )
    return session, provider, ran


def _follow_ups(session: CodingSession) -> list[UserMessage]:
    return [
        message
        for message in session.messages
        if isinstance(message, UserMessage) and message.text == TEXT_TOOL_CALL_NUDGE
    ]


@pytest.mark.anyio
async def test_text_reply_with_tool_call_markup_is_followed_up(tmp_path: Path) -> None:
    session, provider, ran = await _text_session(
        tmp_path, [_says(TEXT_CALL), _calls_note({"text": "run it"}), _says("Report.")]
    )

    await _run(session)

    assert provider.calls[1][2][-1].text == TEXT_TOOL_CALL_NUDGE
    assert ran == [{"text": "run it"}]
    assert len(provider.calls) == 3
    assert len(_follow_ups(session)) == 1


@pytest.mark.anyio
async def test_text_tool_call_follow_ups_are_bounded(tmp_path: Path) -> None:
    # Two in a row, then the run ends; a real call resets the streak, and the
    # per-run cap of four stops a model that acts just enough in between.
    session, provider, _ = await _text_session(tmp_path, [_says(TEXT_CALL) for _ in range(5)])
    await _run(session)
    assert len(provider.calls) == 3
    assert len(_follow_ups(session)) == 2

    (tmp_path / "unopened").mkdir()
    session, provider, _ = await _text_session(
        tmp_path / "unopened", [_says(UNOPENED_CALL) for _ in range(5)]
    )
    await _run(session)
    assert len(provider.calls) == 3
    assert len(_follow_ups(session)) == 2

    piped = '<|tool_call>call:bash{command:<|"|>ls<|"|>}<tool_call|>'
    cycle = [_says(TEXT_CALL), _says(piped), _calls_note({"text": "x"})]
    (tmp_path / "per-run").mkdir()
    session, provider, ran = await _text_session(tmp_path / "per-run", [*cycle * 3, _says("End.")])
    await _run(session)
    assert len(_follow_ups(session)) == 4
    # The fifth markup reply, after the second real call, is not followed up.
    assert len(ran) == 2
    assert len(provider.calls) == 7


@pytest.mark.anyio
async def test_text_tool_call_without_opening_tag_is_followed_up(tmp_path: Path) -> None:
    session, provider, ran = await _text_session(
        tmp_path, [_says(UNOPENED_CALL), _calls_note({"text": "run it"}), _says("Report.")]
    )

    await _run(session)

    assert provider.calls[1][2][-1].text == TEXT_TOOL_CALL_NUDGE
    assert ran == [{"text": "run it"}]
    assert len(_follow_ups(session)) == 1

    for i, prose in enumerate(
        (
            "I will call <function=bash> to list the files.",
            "The bash function runs `ls`; </function> closes a call.",
        )
    ):
        (tmp_path / f"prose-{i}").mkdir()
        session, provider, _ = await _text_session(
            tmp_path / f"prose-{i}", [_says(prose), _says("Unreached.")]
        )
        await _run(session)
        assert len(provider.calls) == 1
        assert _follow_ups(session) == []


@pytest.mark.anyio
async def test_reply_naming_the_tag_in_prose_is_not_followed_up(tmp_path: Path) -> None:
    prose = "Some templates wrap a call in <tool_call> tags; this reply only names them."
    session, provider, _ = await _text_session(tmp_path, [_says(prose), _says("Unreached.")])

    await _run(session)

    assert len(provider.calls) == 1
    assert _follow_ups(session) == []


def _seen_bash(cwd: Path, seen: list[dict[str, JSONValue]]) -> AgentTool:
    """tau's own bash, noting the arguments each call runs with."""
    bash = create_bash_tool(cwd=cwd)

    async def run(
        tool_call_id: str, arguments: object, signal: object = None, on_update: object = None
    ) -> AgentToolResult:
        assert isinstance(arguments, dict)
        seen.append(dict(arguments))
        return await bash.execute(tool_call_id, arguments, signal, on_update)  # type: ignore[arg-type]

    return replace(bash, execute_fn=run)


async def _run_bash(
    tmp_path: Path, *arguments: dict[str, JSONValue]
) -> tuple[list[dict[str, JSONValue]], list[str], FakeProvider]:
    seen: list[dict[str, JSONValue]] = []
    provider = FakeProvider(_turns("bash", *arguments))
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            tools=[_seen_bash(tmp_path, seen)],
        )
    )
    return seen, await _run(session), provider


@pytest.mark.anyio
async def test_leak_in_optional_argument_runs_the_clean_call(tmp_path: Path) -> None:
    # bash's schema requires `description`, but only to label the call: a
    # leak there is cut like one in an optional argument.
    marker = tmp_path / "marker.txt"
    first = {"command": f"echo one > {marker}", "description": LEAKED_DESCRIPTION}
    second = {"command": f"echo two >> {marker}", "description": LEAKED_ESCAPED_DESCRIPTION}

    seen, results, provider = await _run_bash(tmp_path, first, second)

    assert marker.read_text() == "one\ntwo\n"
    assert seen == [
        {"command": f"echo one > {marker}", "description": "Writing the marker file."},
        {"command": f"echo two >> {marker}", "description": "Run the known-answers test."},
    ]
    assert not any(result.startswith("Not run") for result in results)
    for result in results:
        assert result.endswith(
            "Note: the text after the chat template's own tool-call token (`<tool_call|>` in "
            "`description`) was the reply's next call run into this one. It was dropped, and "
            "the call ran without it. Send each call in its own reply."
        )
    # The next request carries the call as it ran, not as it was sent.
    sent = [
        block.arguments
        for message in provider.calls[1][2]
        if isinstance(message, AssistantMessage)
        for block in message.content
        if isinstance(block, ToolCall)
    ]
    assert sent == [seen[0]]


@pytest.mark.anyio
async def test_leak_inside_required_argument_still_refused(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    in_command = {
        "command": f'echo one > {marker}"}}<tool_call|><|channel>thought\n<channel|>',
        "description": "Writing the marker file.",
    }
    # A leak in `description` too does not let the command's leak through.
    in_both = {**in_command, "description": LEAKED_DESCRIPTION}

    seen, results, _ = await _run_bash(tmp_path, in_command, in_both)

    assert seen == []
    assert not marker.exists()
    assert results[0].startswith(
        "Not run: its arguments carry the chat template's own tool-call tokens "
        "(`<tool_call|>` in `command`)"
    )
    assert "(`<tool_call|>` in `command`)" in results[1]
    assert "6 more refused calls in a row" in results[1]


@pytest.mark.anyio
async def test_refusal_names_the_clean_prefix(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    command = f'echo one > {marker}"}}<tool_call|><|channel>thought\n<channel|>'

    _, results, _ = await _run_bash(tmp_path, {"command": command, "description": "Writing."})

    assert results[0].endswith(
        "stop the run. The call to send is this same call with `command` ending before "
        f"`<tool_call|>`, as `echo one > {marker}`."
    )

    # A long value is quoted by its two ends. This one absorbed its sibling
    # arguments, so its clean prefix keeps them: the model sees what it sent.
    ran: list[object] = []
    note = replace(
        _recording_tool(ran),
        name="submit",
        parameters={
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    )
    (tmp_path / "submit").mkdir()
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider(_turns("submit", {"summary": LEAKED_SUMMARY})),
            model="fake",
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "submit" / "session.jsonl"),
            cwd=tmp_path / "submit",
            tools=[note],
        )
    )
    results = await _run(session)

    assert ran == []
    assert results[0].endswith(
        "The call to send is this same call with `summary` ending before `<tool_call|>`, "
        "as `Add a command that reports disk use per directory and per us…"
        'ested where the output varies.",confirmed:false,urgent:true}`.'
    )


#: A write that sent `content` alone, whose file text ran on into its reasoning
#: before the leaked token.
SHORT_WRITE_CONTENT = (
    "from __future__ import annotations\n\nVERSION = 1\n\n"
    "I'll write the code to `src/core.py`.\n<channel|><|tool_call>call:write{content:"
)


@pytest.mark.anyio
async def test_leak_in_a_call_short_of_required_arguments_names_them(tmp_path: Path) -> None:
    ran: list[object] = []
    write = replace(
        _recording_tool(ran),
        name="write",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider(_turns("write", {"content": SHORT_WRITE_CONTENT})),
            model="fake",
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            tools=[write],
        )
    )
    [result] = await _run(session)

    assert ran == []
    assert result.startswith(
        "Not run: its arguments carry the chat template's own tool-call tokens "
        "(`<channel|>` in `content`), so the reply's next call ran into this one, and "
        "it has no `path`, which `write` needs; send one call per reply"
    )
    assert result.endswith(
        "Send `write` again with `path` set, and `content` holding only its own text: "
        "it ends before `<channel|>`, and before any reasoning that ran on into it. "
        "Reason before a call, not inside it."
    )
    # The prefix that holds the reasoning is not offered as the call to send.
    assert "I'll write the code" not in result


@pytest.mark.anyio
async def test_leak_in_arguments_that_did_not_parse_claims_none_missing(tmp_path: Path) -> None:
    # With the arguments only `_raw_arguments`, the refusal said the call had
    # no `path` and `content`, though it carried `path`, and named
    # `_raw_arguments` as an argument to fill.
    ran: list[object] = []
    write = replace(
        _recording_tool(ran),
        name="write",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    )
    raw = '{"path": "src/a.py", "content": "x = 1\n<tool_call|><|channel>thought'
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider(_turns("write", {"_raw_arguments": raw})),
            model="fake",
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            tools=[write],
        )
    )
    [result] = await _run(session)

    assert ran == []
    assert result.startswith(
        "Not run: its arguments are not valid JSON and carry the chat template's own "
        "tool-call token (`<tool_call|>`)"
    )
    assert "has no" not in result and "_raw_arguments" not in result
    assert result.endswith(
        "Send `write` again with its arguments as JSON, each string ending before "
        "`<tool_call|>` and before any reasoning that ran on into it. Reason before a "
        "call, not inside it."
    )


#: A write with no `path`, whose `content` ran from the file into reasoning and
#: the reply's next call.
RUN_ON_CONTENT = (
    "from dataclasses import dataclass\n```\n\nActually, the definition is:\n"
    "`split(0) = (0, 0)`\n\nLet me do this carefully.\n\n"
    "<channel|><|tool_call>call:write{content:"
)


async def _run_write(tmp_path: Path, *arguments: dict[str, JSONValue]) -> list[str]:
    turns: list[list[object]] = [
        [
            assistant_start(model="fake"),
            assistant_done(
                message=AssistantMessage(
                    content=assistant_content(
                        "Calling.", [ToolCall(id=f"call-{i}", name="write", arguments=args)]
                    ),
                    stop_reason="toolUse",
                )
            ),
        ]
        for i, args in enumerate(arguments)
    ]
    turns.append(
        [assistant_start(model="fake"), assistant_done(message=AssistantMessage(content="Done"))]
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider(turns),
            model="fake",
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            tools=[create_write_tool(cwd=tmp_path)],
        )
    )
    async for _ in session.prompt("Go."):
        pass
    return [m.text for m in session.messages if m.role == "toolResult"]


@pytest.mark.anyio
async def test_channel_token_in_a_write_without_a_path_names_both(tmp_path: Path) -> None:
    results = await _run_write(tmp_path, {"content": RUN_ON_CONTENT})

    assert "(`<channel|>` in `content`)" in results[0]
    assert "it ends before `<channel|>`" in results[0]
    assert "with `path` set" in results[0]
    assert list(tmp_path.rglob("*.py")) == []


@pytest.mark.anyio
@pytest.mark.parametrize("token", ["<channel|>", "<|channel|>", "<|channel>", "<|tool_call|>"])
async def test_channel_tokens_never_reach_a_file(tmp_path: Path, token: str) -> None:
    content = f"x = 1\n{token}thought\nNow the next call."
    results = await _run_write(tmp_path, {"path": "a.py", "content": content})

    assert f"(`{token}` in `content`)" in results[0]
    assert not (tmp_path / "a.py").exists()


@pytest.mark.anyio
async def test_plain_pipe_text_is_written(tmp_path: Path) -> None:
    content = 'STATS = "<|stats|>"\nbar = "a|b"\n'
    results = await _run_write(tmp_path, {"path": "a.py", "content": content})

    assert results[0].startswith("Successfully wrote")
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == content


@pytest.mark.anyio
async def test_a_follow_up_the_run_ended_on_does_not_reach_the_next_run(tmp_path: Path) -> None:
    # The report turn at the budget is the run's last, so a call written as text
    # there gets no follow-up, and none is left queued for the next prompt.
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider([_calls_note({"text": "x"}), _says(TEXT_CALL)]),
            model="fake",
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            tools=[_recording_tool([])],
            max_turns=1,
        )
    )

    await _run(session)

    assert session.queued_message_count == 0
    assert _follow_ups(session) == []
