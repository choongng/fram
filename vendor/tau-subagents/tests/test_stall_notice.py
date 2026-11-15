"""The parent hears once when a background agent goes quiet for too long."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from tau_ai import FakeProvider
from test_extension import (
    RecordingSession,
    _agent_tool,
    _bash_call_stream,
    _extension_module,
    _load_runtime,
    _menu_run,
    _patch_provider_factory,
    _stop_tool,
    _text_stream,
    _wait_for,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _Api:
    def __init__(self) -> None:
        self.toasts: list[str] = []
        self.messages: list[tuple[str, str, dict]] = []

    def notify(self, message: str, level: str = "info") -> None:
        self.toasts.append(message)

    def send_custom_message(self, content, *, custom_type, details, deliver_as):  # noqa: ANN001, ANN201
        self.messages.append((content, custom_type, details))


def _manager_with(module, api: _Api, seconds: int):  # noqa: ANN001, ANN202
    manager = module.SubagentManager(api)
    manager._settings = module.SubagentSettings(stall_notice_seconds=seconds)  # noqa: SLF001
    return manager


def test_stall_setting_defaults_on_and_can_be_turned_off(tmp_path: Path) -> None:
    from tau_subagents.settings import load_subagent_settings

    home, cwd = tmp_path / "home", tmp_path / "cwd"
    assert load_subagent_settings(cwd, home=home).stall_notice_seconds == 900
    (cwd / ".tau").mkdir(parents=True)
    config = cwd / ".tau" / "subagents.json"
    config.write_text('{"stallNoticeSeconds": 0}')
    assert load_subagent_settings(cwd, home=home).stall_notice_seconds == 0
    config.write_text('{"stallNoticeSeconds": -5}')
    assert load_subagent_settings(cwd, home=home).stall_notice_seconds == 900


async def test_one_notice_per_quiet_stretch(tmp_path: Path) -> None:
    _load_runtime(tmp_path)
    module = _extension_module()
    api = _Api()
    manager = _manager_with(module, api, 60)
    now = module.time.monotonic()  # type: ignore[attr-defined]
    quiet = _menu_run(
        module,
        status="running",
        started_at=now - 200,
        last_event_at=now - 100,
        phase="in_tool",
        current_tool="bash",
        current_tool_started_at=now - 100,
    )
    recent = _menu_run(
        module, agent_id="agent-2", status="running", started_at=now, last_event_at=now
    )
    foreground = _menu_run(
        module,
        agent_id="agent-3",
        status="running",
        background=False,
        started_at=now - 200,
        last_event_at=now - 200,
    )
    for run in (quiet, recent, foreground):
        manager._runs[run.agent_id] = run  # noqa: SLF001

    manager._check_stalls()  # noqa: SLF001
    manager._check_stalls()  # noqa: SLF001
    assert len(api.messages) == 1
    content, custom_type, details = api.messages[0]
    assert custom_type == "subagent-stall"
    assert "Background subagent agent-1 (task) has sent no events for 1m4" in content
    assert "inside `bash`" in content
    assert "stop_subagent" in content
    assert details["agent_id"] == "agent-1"
    assert details["phase"].startswith("in bash")
    assert api.toasts == ["Subagent agent-1 (task) has been quiet for " + details["quiet"] + "."]

    # The agent speaks, then goes quiet again: a new stretch, a new notice.
    quiet.last_event_at = now - 70
    manager._check_stalls()  # noqa: SLF001
    assert len(api.messages) == 2
    manager._stall_timer.cancel()  # noqa: SLF001

    rendered = module.render_stall_notice(  # type: ignore[attr-defined]
        SimpleNamespace(details=details), SimpleNamespace(expanded=False)
    )
    assert "quiet for" in rendered
    assert "in bash" in rendered


async def test_zero_turns_the_notice_off(tmp_path: Path) -> None:
    _load_runtime(tmp_path)
    module = _extension_module()
    manager = _manager_with(module, _Api(), 0)
    manager._watch_for_stalls()  # noqa: SLF001
    assert manager._stall_timer is None  # noqa: SLF001


async def test_a_wedged_background_agent_triggers_the_notice(tmp_path: Path) -> None:
    runtime = _load_runtime(tmp_path)
    session = RecordingSession(tmp_path)
    runtime.bind(session)
    module = _extension_module()
    module.STALL_CHECK_SECONDS = 0.05  # type: ignore[attr-defined]
    module.load_subagent_settings = (  # type: ignore[attr-defined]
        lambda cwd, home=None: module.SubagentSettings(stall_notice_seconds=1)
    )
    _patch_provider_factory(
        module,
        FakeProvider([_bash_call_stream("sleep 30", "c1"), _text_stream("never")]),
    )
    await _agent_tool(runtime).execute(
        "call-1", {"prompt": "p", "description": "probe", "run_in_background": True}
    )

    await _wait_for(lambda: session.followed_up_custom, tries=400)
    assert session.followed_up_custom[0][0] == "subagent-stall"
    assert "may be stalled" in session.followed_up[0]
    await _stop_tool(runtime).execute("s", {"agent_id": "agent-1"})
    await asyncio.sleep(0.2)
    assert len(session.followed_up_custom) == 1
