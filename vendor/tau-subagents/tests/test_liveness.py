"""A live run's phase: queued, awaiting the model, streaming, or in a tool."""

import asyncio
from pathlib import Path

import pytest
from tau_agent.events import (
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    TurnStartEvent,
)
from tau_agent.messages import AssistantMessage, Usage
from tau_agent.provider_events import TextDeltaEvent
from tau_agent.tools import AgentToolResult
from test_extension import (
    BlockingProvider,
    RecordingSession,
    _agent_tool,
    _extension_module,
    _load_runtime,
    _menu_run,
    _patch_provider_factory,
    _submodule,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _tool_start(call_id: str, name: str) -> ToolExecutionStartEvent:
    return ToolExecutionStartEvent(tool_call_id=call_id, tool_name=name, args={})


def _tool_end(call_id: str, name: str) -> ToolExecutionEndEvent:
    return ToolExecutionEndEvent(
        tool_call_id=call_id,
        tool_name=name,
        result=AgentToolResult(content="ok"),
        is_error=False,
    )


def test_observe_moves_through_the_phases(tmp_path: Path) -> None:
    _load_runtime(tmp_path)
    module = _extension_module()
    manager = module.SubagentManager(None)  # type: ignore[attr-defined]
    run = _menu_run(module, status="running", started_at=0.0)
    assert run.phase == "queued"

    def observe(event: object) -> None:
        manager._observe(run, event, [])  # noqa: SLF001

    observe(TurnStartEvent())
    partial = AssistantMessage(model="fake")
    observe(MessageStartEvent(message=partial))
    assert run.phase == "streaming"
    observe(
        MessageUpdateEvent(
            message=partial,
            assistant_message_event=TextDeltaEvent(
                content_index=0, delta="x" * 400, partial=partial
            ),
        )
    )
    assert run.streamed_chars == 400
    assert module.tokens_so_far(run) == 100  # type: ignore[attr-defined]

    done = AssistantMessage(
        content="calling tools", usage=Usage(input=50, output=10, cache_write=0)
    )
    observe(MessageEndEvent(message=done))
    assert run.phase == "awaiting_model"
    # The estimate is dropped once the response's billed usage arrives.
    assert module.tokens_so_far(run) == 60  # type: ignore[attr-defined]

    # Two parallel tools: ending one keeps the phase on the other.
    observe(_tool_start("a", "bash"))
    observe(_tool_start("b", "grep"))
    assert (run.phase, run.current_tool) == ("in_tool", "grep")
    observe(_tool_end("b", "grep"))
    assert (run.phase, run.current_tool) == ("in_tool", "bash")
    observe(_tool_end("a", "bash"))
    assert (run.phase, run.current_tool) == ("awaiting_model", None)


def test_phase_text_for_each_phase(tmp_path: Path) -> None:
    _load_runtime(tmp_path)
    module = _extension_module()
    phase = _submodule("phase")
    fmt = phase.format_phase  # type: ignore[attr-defined]

    queued = _menu_run(module, status="queued")
    assert fmt(queued, 10.0) == "queued"
    waiting = _menu_run(
        module, status="running", phase="awaiting_model", phase_since=0.0
    )
    assert fmt(waiting, 75.0) == "awaiting model 1m15s"
    streaming = _menu_run(
        module,
        status="running",
        phase="streaming",
        phase_since=0.0,
        streamed_chars=2000,
    )
    assert fmt(streaming, 4.0) == "streaming 4s, ~500 tokens"
    in_tool = _menu_run(
        module,
        status="running",
        phase="in_tool",
        phase_since=0.0,
        current_tool="bash",
        current_tool_started_at=2.0,
    )
    assert fmt(in_tool, 32.0) == "in bash 30s"
    assert fmt(_menu_run(module, status="completed"), 5.0) == ""

    liveness = module.format_liveness(  # type: ignore[attr-defined]
        _menu_run(
            module,
            status="running",
            started_at=0.0,
            last_event_at=1.0,
            phase="awaiting_model",
            phase_since=1.0,
            tokens_input=900,
            tokens_output=100,
        ),
        now=61.0,
    )
    assert liveness == (
        "Running for 1m01s; awaiting the model for 1m00s; last event 1m00s ago;"
        " ~1000 tokens so far."
    )


async def test_get_result_reports_phase_details_while_running(tmp_path: Path) -> None:
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))
    module = _extension_module()
    release = asyncio.Event()
    _patch_provider_factory(module, BlockingProvider(release, "done"))

    await _agent_tool(runtime).execute(
        "call-1", {"prompt": "p", "description": "d", "run_in_background": True}
    )
    get_result = next(
        tool for tool in runtime.extension_tools if tool.name == "get_subagent_result"
    )
    fetched = await get_result.execute("c", {"agent_id": "agent-1"})
    assert "Still running." in fetched.text
    assert "awaiting the model for" in fetched.text
    details = fetched.details
    assert details["status"] == "running"
    assert details["phase"] == "awaiting_model"
    assert details["current_tool"] is None
    assert details["tokens_so_far"] == 0
    assert isinstance(details["phase_elapsed_s"], float)

    release.set()
    fetched = await get_result.execute("c", {"agent_id": "agent-1", "wait": True})
    assert "done" in fetched.text


def test_strip_row_shows_the_phase_of_a_live_run(tmp_path: Path) -> None:
    _load_runtime(tmp_path)
    module = _extension_module()
    from tau_subagents.ui.strip import _row_detail

    live = _menu_run(
        module,
        status="running",
        phase="in_tool",
        current_tool="bash",
        current_tool_started_at=0.0,
    )
    assert _row_detail(live).startswith("task · in bash ")
    assert _row_detail(_menu_run(module, status="queued")) == "task"
    assert _row_detail(_menu_run(module, status="completed")) == "task"


def test_tool_log_keeps_the_last_calls_and_their_outcomes(tmp_path: Path) -> None:
    from tau_subagents.tool_log import format_tool_log, summarize_args

    _load_runtime(tmp_path)
    module = _extension_module()
    manager = module.SubagentManager(None)  # type: ignore[attr-defined]
    run = _menu_run(module, status="running", started_at=0.0)

    def observe(event: object) -> None:
        manager._observe(run, event, [])  # noqa: SLF001

    for index in range(25):
        observe(
            ToolExecutionStartEvent(
                tool_call_id=f"c{index}", tool_name="read", args={"path": f"f{index}"}
            )
        )
        observe(_tool_end(f"c{index}", "read"))
    observe(
        ToolExecutionStartEvent(
            tool_call_id="last", tool_name="bash", args={"command": "sleep  30\nx"}
        )
    )
    assert len(run.recent_tools) == 20
    text = format_tool_log(run.recent_tools, 3, run.tool_calls, now=1e12)
    lines = text.splitlines()
    assert lines[0] == "Recent tool calls (last 3 of 26):"
    assert lines[1].startswith("- read `f23` — ok in ")
    assert lines[3].startswith("- bash `sleep 30 x` — running ")

    cut_off = format_tool_log(run.recent_tools, 1, 26, now=1e12, live=False)
    assert "- bash `sleep 30 x` — never finished; ran " in cut_off

    assert summarize_args({}) == ""
    assert summarize_args({"a": 1}) == '{"a": 1}'
    assert summarize_args({"command": "x" * 200}).endswith("…")
    assert len(summarize_args({"command": "x" * 200})) == 80


async def test_get_result_peeks_at_recent_tool_calls(tmp_path: Path) -> None:
    from tau_ai import FakeProvider
    from test_extension import _bash_call_stream, _stop_tool, _text_stream

    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))
    module = _extension_module()
    _patch_provider_factory(
        module,
        FakeProvider([_bash_call_stream("sleep 30", "c1"), _text_stream("never")]),
    )
    await _agent_tool(runtime).execute(
        "call-1", {"prompt": "p", "description": "d", "run_in_background": True}
    )
    get_result = next(
        tool for tool in runtime.extension_tools if tool.name == "get_subagent_result"
    )
    fetched = None
    for _ in range(500):
        fetched = await get_result.execute("c", {"agent_id": "agent-1", "tools": 5})
        if "inside `bash`" in fetched.text:
            break
        await asyncio.sleep(0.01)
    assert fetched is not None
    assert "Recent tool calls (last 1 of 1):\n- bash `sleep 30` — running" in (
        fetched.text
    )
    assert fetched.details["recent_tools"][0]["tool"] == "bash"
    assert fetched.details["recent_tools"][0]["running"] is True

    plain = await get_result.execute("c", {"agent_id": "agent-1"})
    assert "Recent tool calls" not in plain.text
    assert "recent_tools" not in plain.details

    await _stop_tool(runtime).execute("s", {"agent_id": "agent-1"})
    finished = await get_result.execute("c", {"agent_id": "agent-1", "tools": 99})
    assert "- bash `sleep 30` — " in finished.text
    assert "running" not in finished.text.split("Recent tool calls", 1)[1]
    assert finished.details["recent_tools"][0]["running"] is False
