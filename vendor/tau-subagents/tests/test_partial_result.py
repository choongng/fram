"""A stopped or cancelled run keeps what it got done as a partial result."""

import asyncio
from pathlib import Path

import pytest
from tau_agent.messages import AssistantMessage, TextContent, ToolCall
from tau_ai import AssistantDoneEvent, AssistantStartEvent, FakeProvider
from tau_coding.extensions.api import NullUiBridge
from test_extension import (
    RecordingSession,
    _agent_tool,
    _extension_module,
    _load_runtime,
    _patch_provider_factory,
    _stop_tool,
    _text_stream,
    _wait_for,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _ToastUi(NullUiBridge):
    def __init__(self) -> None:
        self.toasts: list[tuple[str, str]] = []

    def notify(self, message: str, level: str = "info") -> None:  # type: ignore[override]
        self.toasts.append((message, level))


def _say_then_hang() -> list[object]:
    """A response that says something, then runs a command that never ends."""
    return [
        AssistantStartEvent(partial=AssistantMessage(model="fake")),
        AssistantDoneEvent(
            reason="toolUse",
            message=AssistantMessage(
                content=[
                    TextContent(text="Checking the logs first."),
                    ToolCall(
                        id="c1",
                        name="bash",
                        arguments={"command": "sleep 30", "description": "Wait"},
                    ),
                ],
                stop_reason="toolUse",
            ),
        ),
    ]


async def _spawn_hanging_agent(tmp_path: Path):  # noqa: ANN202
    runtime = _load_runtime(tmp_path)
    session = RecordingSession(tmp_path)
    runtime.bind(session)
    module = _extension_module()
    _patch_provider_factory(
        module, FakeProvider([_say_then_hang(), _text_stream("never")])
    )
    await _agent_tool(runtime).execute(
        "call-1", {"prompt": "p", "description": "logs", "run_in_background": True}
    )
    get_result = next(
        tool for tool in runtime.extension_tools if tool.name == "get_subagent_result"
    )

    async def in_bash() -> bool:
        fetched = await get_result.execute("c", {"agent_id": "agent-1"})
        return "inside `bash`" in fetched.text

    for _ in range(500):
        if await in_bash():
            break
        await asyncio.sleep(0.01)
    return runtime, session, get_result


async def test_stop_reports_the_last_message_and_tool_tail(tmp_path: Path) -> None:
    runtime, _session, get_result = await _spawn_hanging_agent(tmp_path)

    stopped = await _stop_tool(runtime).execute("s", {"agent_id": "agent-1"})

    assert "Stopped.\n\nPartial result: the run was aborted" in stopped.text
    assert "Last message:\nChecking the logs first." in stopped.text
    assert "- bash `sleep 30` — " in stopped.text
    assert "Transcript: " in stopped.text
    fetched = await get_result.execute("c", {"agent_id": "agent-1"})
    assert "Checking the logs first." in fetched.text


async def test_session_end_cancels_with_a_partial_record_and_a_warning(
    tmp_path: Path,
) -> None:
    runtime, session, _get_result = await _spawn_hanging_agent(tmp_path)
    ui = _ToastUi()
    runtime.set_ui_bridge(ui)

    await runtime.emit_session_shutdown("new")

    assert ui.toasts
    message, level = ui.toasts[0]
    assert level == "warning"
    assert "Session new: stopping 1 running subagent(s): agent-1 (logs)" in message
    await _wait_for(
        lambda: any(ns == "subagents:record" for ns, _data in session.custom_entries)
    )
    record = next(
        data for ns, data in session.custom_entries if ns == "subagents:record"
    )
    assert record["status"] == "cancelled"
    assert "Partial result: the run was cancelled" in str(record["result"])
    assert "Checking the logs first." in str(record["result"])


async def test_no_warning_when_nothing_is_running(tmp_path: Path) -> None:
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))
    ui = _ToastUi()
    runtime.set_ui_bridge(ui)
    await runtime.emit_session_shutdown("quit")
    assert ui.toasts == []
