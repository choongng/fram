"""A terminal notice reaches a busy parent at its next turn boundary, as steering."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from tau_ai import FakeProvider
from tau_coding.extensions.api import NullUiBridge
from test_extension import (
    RecordingSession,
    _agent_tool,
    _extension_module,
    _load_runtime,
    _patch_provider_factory,
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


async def _finish_in_background(tmp_path: Path, *, running: bool):  # noqa: ANN202
    """Spawn one background agent that finishes at once; wait for its toast."""
    runtime = _load_runtime(tmp_path)
    session = RecordingSession(tmp_path, running=running)
    runtime.bind(session)
    ui = _ToastUi()
    runtime.set_ui_bridge(ui)
    _patch_provider_factory(_extension_module(), FakeProvider([_text_stream("done")]))
    await _agent_tool(runtime).execute(
        "call-1", {"prompt": "p", "description": "d", "run_in_background": True}
    )
    await _wait_for(lambda: ui.toasts)
    assert "agent-1 (d) completed" in ui.toasts[0][0]
    return runtime, session


async def test_busy_parent_gets_the_notice_as_steering_at_turn_end(
    tmp_path: Path,
) -> None:
    runtime, session = await _finish_in_background(tmp_path, running=True)
    # The user saw the toast; the model's notice waits for a turn boundary.
    assert session.followed_up == []
    assert session.steered == []

    await runtime.emit_event(SimpleNamespace(type="turn_end"))
    assert len(session.steered) == 1
    assert "<status>completed</status>" in session.steered[0]
    assert session.followed_up == []

    await runtime.emit_event(SimpleNamespace(type="turn_end"))
    assert len(session.steered) == 1  # sent once


async def test_result_read_before_the_boundary_drops_the_notice(
    tmp_path: Path,
) -> None:
    runtime, session = await _finish_in_background(tmp_path, running=True)
    get_result = next(
        tool for tool in runtime.extension_tools if tool.name == "get_subagent_result"
    )
    fetched = await get_result.execute("c", {"agent_id": "agent-1"})
    assert "done" in fetched.text

    await runtime.emit_event(SimpleNamespace(type="turn_end"))
    await runtime.emit_event(SimpleNamespace(type="agent_settled"))
    assert session.steered == []
    assert session.followed_up == []


async def test_notice_held_past_the_last_turn_goes_out_when_settled(
    tmp_path: Path,
) -> None:
    runtime, session = await _finish_in_background(tmp_path, running=True)
    session.is_running = False
    await runtime.emit_event(SimpleNamespace(type="agent_settled"))
    assert session.steered == []
    assert len(session.followed_up) == 1
    assert "<status>completed</status>" in session.followed_up[0]


async def test_idle_parent_gets_the_notice_at_once(tmp_path: Path) -> None:
    _runtime, session = await _finish_in_background(tmp_path, running=False)
    await _wait_for(lambda: session.followed_up)
    assert len(session.followed_up) == 1
    assert session.steered == []
