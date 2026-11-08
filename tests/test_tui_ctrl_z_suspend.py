"""Ctrl+Z suspends tau to the shell; fg brings it back redrawn and intact.

Textual's headless test driver cannot suspend, so these tests stand in for the
terminal: the driver reports it can suspend, its suspend/resume of application
mode is recorded, and ``os.kill`` is replaced by a recorder. No SIGTSTP is ever
sent to the test process. The recorder returning plays the part of ``fg``.
"""

import asyncio
import os
import signal
import sys
from collections.abc import AsyncIterator

import pytest
from textual.app import App, ComposeResult
from textual.drivers.headless_driver import HeadlessDriver
from textual.screen import ModalScreen
from textual.widgets import Input

from tau_agent import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    AssistantMessage,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    TextContent,
    UserMessage,
)
from tau_agent.provider_events import TextDeltaEvent
from tau_coding.commands import create_default_command_registry
from tau_coding.tui import app as tui_app
from tau_coding.tui.app import EditorTextArea, PromptInput, TauTuiApp
from tau_coding.tui.config import TuiKeybindings, TuiSettings, tui_settings_from_json
from test_commands import FakeSession as CommandFakeSession
from test_tui_app import FakeSession

if sys.platform == "win32":
    pytest.skip("Windows has no SIGTSTP", allow_module_level=True)

RESUMED_NOTE = "Resumed; the turn in flight was paused while suspended."
UNSUPPORTED_NOTE = "Suspend (ctrl+z) is not supported in this terminal."


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """A suspendable stand-in terminal; returns the ordered log of what happened."""
    log: list[object] = []
    monkeypatch.setattr(HeadlessDriver, "can_suspend", property(lambda self: True))
    monkeypatch.setattr(
        HeadlessDriver, "suspend_application_mode", lambda self: log.append("terminal restored")
    )
    monkeypatch.setattr(
        HeadlessDriver, "resume_application_mode", lambda self: log.append("terminal re-entered")
    )

    def kill(pid: int, sig: int) -> None:
        # The default action is in place, so the signal itself stops tau.
        log.append(("kill", pid, sig, signal.getsignal(signal.SIGTSTP)))

    monkeypatch.setattr(os, "kill", kill)
    return log


# Terminal restored before the stop, re-entered after fg.
STOPPED_GROUP: list[object] = [
    "terminal restored",
    ("kill", 0, signal.SIGTSTP, signal.SIG_DFL),
    "terminal re-entered",
]


def _rows(app: TauTuiApp) -> list[tuple[str, str]]:
    return [(item.role, item.text) for item in app.state.items]


def _turn(index: int) -> list[AgentEvent]:
    answer = f"answer {index}"
    partial = AssistantMessage()
    return [
        MessageEndEvent(message=UserMessage(content=f"prompt {index}")),
        AgentStartEvent(),
        MessageStartEvent(message=partial),
        MessageUpdateEvent(
            message=partial,
            assistant_message_event=TextDeltaEvent(content_index=0, delta=answer, partial=partial),
        ),
        MessageEndEvent(message=AssistantMessage(content=[TextContent(text=answer)])),
        AgentEndEvent(),
    ]


class _Dialog(ModalScreen[None]):
    def compose(self) -> ComposeResult:
        yield Input(id="dialog-input")


@pytest.mark.anyio
async def test_ctrl_z_suspend_from_the_prompt_stops_the_group_and_resumes_intact(
    terminal: list[object],
) -> None:
    session = FakeSession()
    app = TauTuiApp(session)
    tstp_handler = signal.getsignal(signal.SIGTSTP)

    async with app.run_test(size=(100, 30)) as pilot:
        session.events = tuple(_turn(0))
        await app._run_prompt("prompt 0")
        await pilot.pause()
        rows = _rows(app)
        assert ("assistant", "answer 0") in rows
        prompt = app.query_one(PromptInput)
        prompt.focus()
        await pilot.press(*"half a thought")
        redraws: list[tuple[bool, bool]] = []
        refresh = app.refresh

        def spy(*args: object, repaint: bool = True, layout: bool = False, **kw: object):
            redraws.append((repaint, layout))
            return refresh(*args, repaint=repaint, layout=layout, **kw)

        app.refresh = spy  # type: ignore[method-assign]

        await pilot.press("ctrl+z")
        await pilot.pause()

        # Terminal restored before the stop, re-entered after fg, then redrawn.
        assert terminal == STOPPED_GROUP
        assert (True, True) in redraws
        assert prompt.text == "half a thought"
        assert _rows(app) == rows
        assert app.focused is prompt
    # Textual's own SIGTSTP handler is back once tau continues.
    assert signal.getsignal(signal.SIGTSTP) == tstp_handler


@pytest.mark.anyio
@pytest.mark.parametrize("focus", ["transcript", "dialog"])
async def test_ctrl_z_suspend_whichever_widget_has_focus(
    terminal: list[object], focus: str
) -> None:
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        if focus == "dialog":
            await app.push_screen(_Dialog())
            await pilot.pause()
            app.screen.query_one("#dialog-input", Input).focus()
            await pilot.press(*"typed")
        else:
            app.query_one("#transcript").focus()
        await pilot.pause()

        await pilot.press("ctrl+z")
        await pilot.pause()

        assert terminal == STOPPED_GROUP
        if focus == "dialog":
            assert app.screen.query_one("#dialog-input", Input).value == "typed"


@pytest.mark.anyio
async def test_ctrl_z_suspend_mid_turn_pauses_it_and_it_finishes_after_fg(
    terminal: list[object],
) -> None:
    gate = asyncio.Event()

    class GatedSession(FakeSession):
        async def prompt(self, text: str, **kwargs: object) -> AsyncIterator[AgentEvent]:
            del kwargs
            self.prompt_texts.append(text)
            partial = AssistantMessage()
            yield MessageEndEvent(message=UserMessage(content=text))
            yield AgentStartEvent()
            yield MessageStartEvent(message=partial)
            yield MessageUpdateEvent(
                message=partial,
                assistant_message_event=TextDeltaEvent(
                    content_index=0, delta="first half", partial=partial
                ),
            )
            await gate.wait()
            yield MessageEndEvent(
                message=AssistantMessage(content=[TextContent(text="first half, second half")])
            )
            yield AgentEndEvent()

    app = TauTuiApp(GatedSession())

    async with app.run_test(size=(100, 30), notifications=True) as pilot:
        await pilot.pause()
        turn = asyncio.create_task(app._run_prompt("go"))
        for _ in range(50):
            await pilot.pause()
            if app.state.running:
                break
        assert app.state.running

        await pilot.press("ctrl+z")
        await pilot.pause()

        assert terminal == STOPPED_GROUP
        assert RESUMED_NOTE in [note.message for note in app._notifications]
        assert app.state.running

        gate.set()
        await turn
        await pilot.pause()
        assert ("assistant", "first half, second half") in _rows(app)
        assert not app.state.running


@pytest.mark.anyio
@pytest.mark.parametrize("why", ["driver cannot suspend", "windows"])
async def test_ctrl_z_suspend_unsupported_is_a_harmless_note(
    monkeypatch: pytest.MonkeyPatch, why: str
) -> None:
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: sent.append((pid, sig)))
    if why == "windows":
        monkeypatch.setattr(HeadlessDriver, "can_suspend", property(lambda self: True))
        monkeypatch.setattr(tui_app, "_WINDOWS", True)
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30), notifications=True) as pilot:
        prompt = app.query_one(PromptInput)
        prompt.focus()
        await pilot.press(*"keep me")

        await pilot.press("ctrl+z")
        await pilot.pause()

        assert sent == []
        # Not taken as undo either: the draft is untouched.
        assert prompt.text == "keep me"
        assert [note.message for note in app._notifications] == [UNSUPPORTED_NOTE]


@pytest.mark.anyio
async def test_ctrl_z_suspend_moves_prompt_undo_to_a_named_key(terminal: list[object]) -> None:
    app = TauTuiApp(FakeSession())

    async with app.run_test(size=(100, 30)) as pilot:
        prompt = app.query_one(PromptInput)
        prompt.focus()
        await pilot.press(*"oops")
        await pilot.press("ctrl+z")
        await pilot.pause()
        assert prompt.text == "oops"
        assert terminal == STOPPED_GROUP

        for undo_key in TuiKeybindings().undo.split(","):
            prompt.text = ""
            await pilot.press(*"oops")
            await pilot.press(undo_key)
            await pilot.pause()
            assert prompt.text == "", undo_key


@pytest.mark.anyio
async def test_ctrl_z_suspend_key_is_configurable(terminal: list[object]) -> None:
    settings = tui_settings_from_json({"keybindings": {"suspend": "f9"}})
    assert settings.keybindings.suspend == "f9"
    assert TuiSettings().to_json()["keybindings"]["suspend"] == "ctrl+z"
    app = TauTuiApp(FakeSession(), tui_settings=settings)

    async with app.run_test(size=(100, 30)) as pilot:
        prompt = app.query_one(PromptInput)
        prompt.focus()
        await pilot.press("f9")
        await pilot.pause()
        assert terminal == STOPPED_GROUP

        # With suspend elsewhere, ctrl+z is the TextArea's undo again.
        await pilot.press(*"abc")
        await pilot.press("ctrl+z")
        await pilot.pause()
        assert prompt.text == ""
        assert len(terminal) == 3


def test_ctrl_z_suspend_and_undo_keys_are_in_hotkeys(tmp_path) -> None:
    result = create_default_command_registry().execute(CommandFakeSession(tmp_path), "/hotkeys")

    assert result.message is not None
    assert "Ctrl+Z: suspend to the shell (fg resumes it)" in result.message
    assert "Ctrl+_ or Ctrl+/: undo in the prompt and editors (Ctrl+Y redoes)" in result.message


class _EditorHost(App[None]):
    def __init__(self, settings: TuiSettings) -> None:
        super().__init__()
        self.tui_settings = settings

    def compose(self) -> ComposeResult:
        yield EditorTextArea("", id="editor")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("keybindings", "key"),
    [(TuiKeybindings(), "ctrl+slash"), (TuiKeybindings(undo="f9"), "f9")],
)
async def test_editors_undo_with_the_configured_key(keybindings: TuiKeybindings, key: str) -> None:
    app = _EditorHost(TuiSettings(keybindings=keybindings))

    async with app.run_test() as pilot:
        editor = app.query_one("#editor", EditorTextArea)
        editor.focus()
        await pilot.press("a")
        await pilot.pause()
        assert editor.text == "a"

        await pilot.press(key)
        await pilot.pause()

        assert editor.text == ""
