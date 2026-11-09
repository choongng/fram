"""The repeat breaker refuses tool calls that keep giving the same result."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from pi_event_helpers import assistant_done, assistant_start
from tau_agent import AgentTool, AgentToolResult, AssistantMessage, TextContent, ToolCall
from tau_agent.messages import assistant_content
from tau_agent.provider_events import AssistantDoneEvent
from tau_agent.session import JsonlSessionStorage
from tau_agent.types import JSONValue
from tau_ai import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig
from tau_coding.guards import ToolCallGuards, call_key, command_heads

Args = dict[str, JSONValue]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _tool(
    name: str, ran: list[str], *, changes_workspace: bool = False, text: str | None = None
) -> AgentTool:
    async def execute(
        tool_call_id: object, arguments: object, signal: object = None, on_update: object = None
    ) -> AgentToolResult:
        ran.append(name)
        return AgentToolResult(
            content=[TextContent(text=f"{name}: same" if text is None else text)]
        )

    return AgentTool(
        name=name,
        label=name,
        description=f"{name} something.",
        parameters={"type": "object"},
        execute_fn=execute,
        changes_workspace=changes_workspace,
    )


def _changing_tool(name: str, ran: list[str]) -> AgentTool:
    async def execute(
        tool_call_id: object, arguments: object, signal: object = None, on_update: object = None
    ) -> AgentToolResult:
        ran.append(name)
        return AgentToolResult(content=[TextContent(text=f"found {len(ran)}")])

    return AgentTool(
        name=name,
        label=name,
        description="Look.",
        parameters={"type": "object"},
        execute_fn=execute,
    )


def _turns(calls: list[tuple[str, Args]]) -> list[list[object]]:
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
        for i, (name, args) in enumerate(calls)
    ]
    done = [assistant_start(model="fake"), assistant_done(message=AssistantMessage(content="Done"))]
    return [*turns, done]


async def _run(
    tmp_path: Path, tools: list[AgentTool], calls: list[tuple[str, Args]]
) -> tuple[CodingSession, FakeProvider]:
    provider = FakeProvider(_turns(calls))
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            tools=tools,
        )
    )
    async for _ in session.prompt("Go."):
        pass
    return session, provider


def _results(session: CodingSession) -> list[str]:
    return [m.text for m in session.messages if m.role == "toolResult"]


@pytest.mark.anyio
async def test_refuses_a_call_repeated_with_the_same_result(tmp_path: Path) -> None:
    ran: list[str] = []
    same: Args = {"where": "logs"}
    other: Args = {"where": "tests"}

    session, _ = await _run(
        tmp_path,
        [_tool("look", ran)],
        [("look", same), ("look", same), ("look", same), ("look", other), ("look", same)],
    )

    # The third is refused, and a different call in between does not reset it.
    assert len(ran) == 3
    results = _results(session)
    assert results[2].startswith("Not run: this `look` call ran 2 times in the last 2")
    assert results[4].startswith("Not run: this `look` call ran 2 times in the last 3")
    assert session.messages[-1].text == "Done"


@pytest.mark.anyio
async def test_refuses_two_calls_taken_in_turn(tmp_path: Path) -> None:
    ran: list[str] = []
    a: Args = {"where": "a"}
    b: Args = {"where": "b"}

    session, _ = await _run(tmp_path, [_tool("look", ran)], [("look", a), ("look", b)] * 3)

    assert len(ran) == 4
    assert all(text.startswith("Not run:") for text in _results(session)[4:])


@pytest.mark.anyio
async def test_runs_a_repeated_call_whose_result_changes(tmp_path: Path) -> None:
    ran: list[str] = []
    same: Args = {"where": "next"}

    await _run(tmp_path, [_changing_tool("look", ran)], [("look", same)] * 4)

    assert len(ran) == 4


@pytest.mark.anyio
async def test_a_workspace_change_forgets_earlier_results(tmp_path: Path) -> None:
    ran: list[str] = []
    build: Args = {"command": "make"}
    change: Args = {"path": "a.py"}
    calls = [("check", build), ("check", build), ("change", change), ("check", build)]
    calls += [("check", build), ("check", build)]

    session, _ = await _run(
        tmp_path, [_tool("check", ran), _tool("change", ran, changes_workspace=True)], calls
    )

    # Only the third call after the change, with the same result twice since
    # the change, is a repeat.
    assert ran == ["check", "check", "change", "check", "check"]
    assert _results(session)[5].startswith("Not run: this `check` call ran 2 times")


@pytest.mark.anyio
async def test_a_new_bash_result_forgets_earlier_results(tmp_path: Path) -> None:
    ran: list[str] = []
    test: Args = {"command": "pytest -q"}
    fixes: list[Args] = [{"command": f"sed -i 's/a/b/' f{n}.py"} for n in range(3)]
    calls = [("check", test)]
    for fix in fixes:
        calls += [("bash", fix), ("check", test)]
    counter = iter(range(100))
    bash = _tool("bash", ran)

    async def execute(
        tool_call_id: object, arguments: object, signal: object = None, on_update: object = None
    ) -> AgentToolResult:
        ran.append("bash")
        return AgentToolResult(content=[TextContent(text=f"fix {next(counter)}")])

    session, _ = await _run(
        tmp_path, [_tool("check", ran), replace(bash, execute_fn=execute)], calls
    )

    # Each fix through bash gave a result of its own, so the same failing
    # test after it is run again.
    assert ran.count("check") == 4
    assert not any(text.startswith("Not run:") for text in _results(session))


@pytest.mark.anyio
async def test_two_bash_commands_taken_in_turn_are_still_caught(tmp_path: Path) -> None:
    ran: list[str] = []
    a: Args = {"command": "cat a.txt"}
    b: Args = {"command": "cat b.txt"}

    session, _ = await _run(tmp_path, [_tool("bash", ran)], [("bash", a), ("bash", b)] * 4)

    # The first run of each is new, so the pair is caught one round later
    # than two calls that change nothing.
    assert len(ran) == 5
    assert all(text.startswith("Not run:") for text in _results(session)[5:])


@pytest.mark.anyio
async def test_a_repeated_change_is_still_caught_and_told_to_check_it(tmp_path: Path) -> None:
    ran: list[str] = []
    change: Args = {"path": "a.py"}

    session, _ = await _run(
        tmp_path, [_tool("change", ran, changes_workspace=True)], [("change", change)] * 3
    )

    assert ran == ["change", "change"]
    assert "That change is already made" in _results(session)[2]


@pytest.mark.anyio
async def test_a_repeated_failed_change_quotes_the_failure(tmp_path: Path) -> None:
    ran: list[str] = []
    change: Args = {"path": "a.py"}

    async def fail(
        tool_call_id: object, arguments: object, signal: object = None, on_update: object = None
    ) -> AgentToolResult:
        ran.append("change")
        raise ValueError("old text not found")

    tool = AgentTool(
        name="change",
        label="change",
        description="Change something.",
        parameters={"type": "object"},
        execute_fn=fail,
        changes_workspace=True,
    )

    session, _ = await _run(tmp_path, [tool], [("change", change)] * 3)

    assert ran == ["change", "change"]
    assert "It said: old text not found" in _results(session)[2]
    assert "already made" not in _results(session)[2]


@pytest.mark.anyio
async def test_a_repeated_empty_result_asks_for_a_different_step(tmp_path: Path) -> None:
    ran: list[str] = []
    same: Args = {"where": "logs"}

    session, _ = await _run(tmp_path, [_tool("look", ran, text="")], [("look", same)] * 3)

    assert "It gave no output each time" in _results(session)[2]
    assert "It said:" not in _results(session)[2]


@pytest.mark.anyio
async def test_refusal_quotes_the_earlier_result_once(tmp_path: Path) -> None:
    head = "save: nothing written because that name is not allowed here. " * 6
    ran: list[str] = []
    args: Args = {"names": ["x", "y"]}

    session, _ = await _run(
        tmp_path, [_tool("save", ran, text=f"{head}\nLeave it out.")], [("save", args)] * 4
    )

    assert ran == ["save", "save"]
    results = _results(session)
    assert f"It said: {head.strip()[:300]}" in results[2]
    assert head.strip()[:301] not in results[2] and "Leave it out." not in results[2]
    assert "It said:" not in results[3]
    assert "quoted in this call's first refusal" in results[3]


@pytest.mark.anyio
async def test_refusal_quotes_the_sentence_holding_the_requested_call(tmp_path: Path) -> None:
    why = "apply: not changed because nothing differs from the file on disk. " * 5
    ask = 'Send apply target="a" mode="full" to go on.'
    ran: list[str] = []
    args: Args = {"target": "a"}

    session, _ = await _run(
        tmp_path, [_tool("apply", ran, text=f"{why}{ask}\nThen stop.")], [("apply", args)] * 3
    )

    assert f"It said: {ask} Do what that result asks" in _results(session)[2]
    assert why.strip()[:300] not in _results(session)[2]


@pytest.mark.anyio
async def test_refuses_a_tool_run_through_bash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # util-linux ships a `look` program; this test's `look` is only a tool.
    which = shutil.which
    monkeypatch.setattr(
        shutil, "which", lambda name, *a, **k: None if name == "look" else which(name, *a, **k)
    )
    ran: list[str] = []
    calls: list[tuple[str, Args]] = [
        ("bash", {"command": "cd /tmp && look"}),
        ("bash", {"command": "ls && echo hi"}),
    ]

    session, _ = await _run(tmp_path, [_tool("bash", ran), _tool("look", [])], calls)

    assert ran == ["bash"]
    assert _results(session)[0].startswith("Not run: `look` is one of your tools")


@pytest.mark.parametrize("command", ["history | tail", "echo x | read v"])
def test_a_shell_builtin_named_like_a_tool_is_left_alone(command: str) -> None:
    tools = [_tool("bash", []), _tool("read", []), _tool("history", [])]
    guards = ToolCallGuards(tools=lambda: tools, stop_run=lambda: None)

    assert guards.problem(ToolCall(id="c", name="bash", arguments={"command": command})) is None


def test_a_repeat_of_a_cut_read_quotes_the_offset_to_go_on_from() -> None:
    guards = ToolCallGuards(tools=lambda: [], stop_run=lambda: None)
    call = ToolCall(id="c", name="read", arguments={"path": "a.py"})
    result = AgentToolResult(content=[TextContent(text="first line\nsecond line")])
    note = "[Read cut: lines 1-2. Use offset=3 to continue.]"
    shown = AgentToolResult(content=[TextContent(text=f"first line\n{note}")])

    guards.record(call, result, shown, False)
    guards.record(call, result, shown, False)

    problem = guards.problem(call)
    assert problem is not None and f"It said: {note}" in problem


@pytest.mark.parametrize(
    ("command", "heads"),
    [
        ("cd /tmp && look", ["cd", "look"]),
        # A commit message's lines do not run, whatever word they begin with.
        ("git commit -F - <<'EOF'\nfix: x\n\nlook at it\nEOF\nlook", ["git", "look"]),
        ("cat <<-END\n\tlook\n\tEND\necho ok", ["cat", "echo"]),
        ('git commit -m "first\nlook again" && ls', ["git", "ls"]),
        ("echo 'a; look'; ls | grep x", ["echo", "ls", "grep"]),
        ("cat <<< look", ["cat"]),
    ],
)
def test_command_heads_skip_heredoc_bodies_and_quotes(command: str, heads: list[str]) -> None:
    assert command_heads(command) == heads


def test_call_key_ignores_duplicate_list_entries() -> None:
    def key(arguments: Args, name: str = "apply") -> str:
        return call_key(ToolCall(id="c", name=name, arguments=arguments))

    base = key({"target": "a", "paths": ["a.py", "b.py"]})
    assert key({"target": "a", "paths": ["a.py", "b.py", "a.py", "a.py"]}) == base
    assert key({"target": "a", "paths": ["a.py", "a.py", "b.py", "b.py"]}) == base
    nested = key({"opts": {"paths": ["x", "y"]}})
    assert key({"opts": {"paths": ["x", "x", "y", "x"]}}) == nested

    assert key({"target": "a", "paths": ["b.py", "a.py"]}) != base
    assert key({"target": "a", "paths": ["a.py"]}) != base
    assert key({"target": "b", "paths": ["a.py", "b.py"]}) != base
    assert key({"flags": [1, True]}) != key({"flags": [1]})
    assert key({"target": "a", "paths": ["a.py", "b.py"]}, name="save") != base


@pytest.mark.anyio
async def test_a_run_that_keeps_repeating_ends_aborted_with_the_reason(tmp_path: Path) -> None:
    ran: list[str] = []
    same: Args = {"where": "logs"}

    session, provider = await _run(tmp_path, [_tool("look", ran)], [("look", same)] * 20)

    # Two run, eight are refused, and no request follows the eighth refusal.
    assert len(ran) == 2
    assert len(provider.calls) == 10
    final = session.messages[-1]
    assert isinstance(final, AssistantMessage)
    assert final.stop_reason == "aborted"
    assert final.error_message == (
        "Not run, and the run is stopped: 8 tool calls in a row were refused."
    )


@pytest.mark.anyio
async def test_a_call_cut_off_at_the_output_limit_does_not_count_as_run(tmp_path: Path) -> None:
    ran: list[str] = []
    same: Args = {"where": "logs"}
    turns: list[list[object]] = [
        [
            assistant_start(model="fake"),
            AssistantDoneEvent(
                reason="length",
                message=AssistantMessage(
                    content=assistant_content(
                        "Calling.", [ToolCall(id=f"call-{i}", name="look", arguments=same)]
                    ),
                    stop_reason="length",
                ),
            ),
        ]
        for i in range(3)
    ]
    done = [assistant_start(model="fake"), assistant_done(message=AssistantMessage(content="Done"))]
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider([*turns, done]),
            model="fake",
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            tools=[_tool("look", ran)],
        )
    )

    async for _ in session.prompt("Go."):
        pass

    assert ran == []
    assert not any("call ran" in text for text in _results(session))


@pytest.mark.anyio
async def test_each_run_starts_with_a_clean_record(tmp_path: Path) -> None:
    ran: list[str] = []
    same: Args = {"where": "logs"}
    provider = FakeProvider([*_turns([("look", same)] * 2), *_turns([("look", same)] * 2)])
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            tools=[_tool("look", ran)],
        )
    )

    for prompt in ("First.", "Second."):
        async for _ in session.prompt(prompt):
            pass

    assert len(ran) == 4
