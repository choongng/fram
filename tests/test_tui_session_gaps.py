"""TUI prompt handling against a real ``CodingSession``.

The fake sessions in ``test_tui_app.py`` say when a run can queue a message.
These tests let the session say it: the provider is scripted, and everything
between it and the app is the real thing.
"""

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytest

from pi_event_helpers import assistant_done, assistant_start
from tau_agent import (
    AgentMessage,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from tau_agent.messages import assistant_content
from tau_agent.session import JsonlSessionStorage, MessageEntry
from tau_ai import CancellationToken
from tau_ai.events import AssistantMessageEvent
from tau_coding import (
    CodingSession,
    CodingSessionConfig,
)
from tau_coding.extensions import ExtensionAPI, InputHookResult
from tau_coding.extensions.loader import LoadedExtension
from tau_coding.tui.app import PromptInput, TauTuiApp


@dataclass
class _Reply:
    """One scripted provider response, which can wait before it is sent."""

    events: Sequence[AssistantMessageEvent]
    gate: asyncio.Event | None = None
    started: asyncio.Event = field(default_factory=asyncio.Event)


def _says(text: str, *, gated: bool = False) -> _Reply:
    return _Reply(
        [assistant_start(model="fake"), assistant_done(message=AssistantMessage(content=text))],
        gate=asyncio.Event() if gated else None,
    )


class _ScriptedProvider:
    """Replays replies in order and records the messages of each request."""

    def __init__(self, replies: Sequence[_Reply]) -> None:
        self._replies = list(replies)
        self.calls: list[list[AgentMessage]] = []

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        del model, system, tools, signal, session_id
        self.calls.append(list(messages))
        reply = self._replies.pop(0)

        async def iterator() -> AsyncIterator[AssistantMessageEvent]:
            reply.started.set()
            if reply.gate is not None:
                await reply.gate.wait()
            for event in reply.events:
                yield event

        return iterator()


async def _load_session(
    tmp_path: Path, provider: _ScriptedProvider, **config: Any
) -> CodingSession:
    return await CodingSession.load(
        CodingSessionConfig(
            provider=provider,  # type: ignore[arg-type]
            model="fake",
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            **config,
        )
    )


def _extension_api(session: CodingSession) -> ExtensionAPI:
    """Register an empty extension with the session and return its API."""
    captured: list[ExtensionAPI] = []
    session.extension_runtime._setup_extension(  # noqa: SLF001 - test seam
        LoadedExtension(
            name="probe",
            path=Path("/virtual/probe.py"),
            source_id="extension:test:probe",
            setup=captured.append,
        )
    )
    return captured[0]


def _count_input_hooks(api: ExtensionAPI) -> list[str]:
    """Record every text the session's `input` hooks are run on."""
    hooked: list[str] = []

    def hook(event: Any, context: object) -> InputHookResult | None:
        del context
        hooked.append(event.text)
        return None

    api.on("input", hook)
    return hooked


async def _submit(
    app: TauTuiApp, text: str, streaming_behavior: Literal["steer", "follow_up"] = "steer"
) -> None:
    """Submit ``text`` like Enter does, failing if the handler waits on a run."""
    app.query_one("#prompt", PromptInput).text = text
    await asyncio.wait_for(
        app._submit_prompt_from_editor(streaming_behavior=streaming_behavior), timeout=5
    )


async def _pause_until(pilot: Any, condition: Callable[[], bool]) -> None:
    for _ in range(400):
        if condition():
            return
        await pilot.pause()
    raise AssertionError("condition was not met")


def _is_idle(app: TauTuiApp) -> bool:
    return (
        app._active_prompt_run_id is None
        and not app._has_held_prompts()
        and not app.session.is_running
        and not app.state.running
    )


def _user_texts(session: CodingSession) -> list[str]:
    return [message.text for message in session.messages if isinstance(message, UserMessage)]


async def _stored_user_texts(tmp_path: Path) -> list[str]:
    entries = await JsonlSessionStorage(tmp_path / "session.jsonl").read_all()
    return [
        entry.message.text
        for entry in entries
        if isinstance(entry, MessageEntry) and isinstance(entry.message, UserMessage)
    ]


class _BlockingHook:
    """An extension handler that waits, once it has started, until released."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self.finished = 0

    async def __call__(self, event: object, context: object) -> None:
        del event, context
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        self.finished += 1


async def _interrupt_into_settle(
    app: TauTuiApp, pilot: Any, reply: _Reply, settle: _BlockingHook
) -> None:
    """Start a run, interrupt it mid-reply and wait for its settle hook to block."""
    await _submit(app, "first")
    await asyncio.wait_for(reply.started.wait(), timeout=5)
    await pilot.pause()
    await pilot.press("escape")
    await asyncio.wait_for(settle.entered.wait(), timeout=5)


async def _pause(pilot: Any, times: int = 10) -> None:
    for _ in range(times):
        await pilot.pause()


@pytest.mark.anyio
async def test_prompt_after_escape_waits_for_the_interrupted_run_to_settle(
    tmp_path: Path,
) -> None:
    first = _says("First", gated=True)
    provider = _ScriptedProvider([first, _says("Second")])
    session = await _load_session(tmp_path, provider)
    api = _extension_api(session)
    hooked = _count_input_hooks(api)
    settle = _BlockingHook()
    api.on("agent_settled", settle)
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await _interrupt_into_settle(app, pilot, first, settle)
        # The interrupted run is still settling: a second run now would call
        # the provider and settle while the first one's hooks are running.
        await _submit(app, "second")
        await _pause(pilot)

        assert len(provider.calls) == 1
        assert settle.calls == 1
        assert app._waiting_prompt_texts() == ("second",)

        settle.release.set()
        await _pause_until(pilot, lambda: len(provider.calls) == 2 and _is_idle(app))

        assert [message.text for message in provider.calls[1]].count("second") == 1
        assert hooked == ["first", "second"]
        assert _user_texts(session).count("second") == 1
        assert (await _stored_user_texts(tmp_path)).count("second") == 1
        assert settle.calls == 2
        assert app._waiting_prompt_texts() == ()


@pytest.mark.anyio
async def test_second_escape_while_an_interrupted_run_settles_frees_the_session(
    tmp_path: Path,
) -> None:
    first = _says("First", gated=True)
    provider = _ScriptedProvider([first, _says("Third")])
    session = await _load_session(tmp_path, provider)
    api = _extension_api(session)
    settle = _BlockingHook()
    api.on("agent_settled", settle)
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await _interrupt_into_settle(app, pilot, first, settle)
        await _submit(app, "second")
        # The hook never returns by itself. The second interrupt has to end
        # the wait, or nothing could run in this session again.
        await pilot.press("escape")
        await _pause_until(pilot, lambda: _is_idle(app))

        assert settle.finished == 0
        assert app._waiting_prompt_texts() == ()
        assert len(provider.calls) == 1

        settle.release.set()
        await _submit(app, "third")
        await _pause_until(pilot, lambda: len(provider.calls) == 2 and _is_idle(app))

        assert "second" not in [message.text for message in provider.calls[1]]
        assert _user_texts(session).count("third") == 1


def _hanging_tool(started: asyncio.Event) -> AgentTool:
    async def hang(
        tool_call_id: object,
        arguments: object,
        signal: object = None,
        on_update: object = None,
    ) -> AgentToolResult:
        del tool_call_id, arguments, signal, on_update
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    return AgentTool(
        name="fetch",
        label="Fetch",
        description="Never return.",
        parameters={"type": "object"},
        execute_fn=hang,
    )


@pytest.mark.anyio
async def test_second_escape_while_an_interrupted_tool_result_is_saved_lets_the_save_finish(
    tmp_path: Path,
) -> None:
    tool_started = asyncio.Event()
    calling = AssistantMessage(
        content=assistant_content("Fetching.", [ToolCall(id="call-1", name="fetch", arguments={})]),
        stop_reason="toolUse",
    )
    provider = _ScriptedProvider(
        [_Reply([assistant_start(model="fake"), assistant_done(message=calling)]), _says("Next")]
    )
    session = await _load_session(tmp_path, provider, tools=[_hanging_tool(tool_started)])
    api = _extension_api(session)
    settled: list[str] = []
    api.on("agent_settled", lambda event, context: settled.append("settled"))
    app = TauTuiApp(session)
    slow = False
    saving = asyncio.Event()
    disk_ready = asyncio.Event()
    cut_short: list[BaseException] = []
    append_entry = session._append_session_entry

    async def append_slowly(entry: Any, *args: Any, **kwargs: Any) -> Any:
        if slow:
            saving.set()
            try:
                await disk_ready.wait()
            except BaseException as exc:
                cut_short.append(exc)
                raise
        return await append_entry(entry, *args, **kwargs)

    session._append_session_entry = append_slowly  # type: ignore[method-assign]

    async with app.run_test(size=(120, 30)) as pilot:
        await _submit(app, "first")
        await asyncio.wait_for(tool_started.wait(), timeout=5)
        await pilot.pause()
        slow = True
        await pilot.press("escape")
        # The session is saving the result that says the tool was interrupted.
        await asyncio.wait_for(saving.wait(), timeout=5)
        await pilot.press("escape")
        await _pause(pilot)

        assert cut_short == []

        slow = False
        disk_ready.set()
        await _pause_until(pilot, lambda: _is_idle(app))

        assert cut_short == []
        # The second interrupt gave up the settle hooks, and nothing else.
        assert settled == []
        entries = await JsonlSessionStorage(tmp_path / "session.jsonl").read_all()
        assert [
            entry.message.text
            for entry in entries
            if isinstance(entry, MessageEntry) and isinstance(entry.message, ToolResultMessage)
        ] == ["Tool call interrupted by user"]

        await _submit(app, "next")
        await _pause_until(pilot, lambda: len(provider.calls) == 2 and _is_idle(app))

        assert app.state.error is None
        assert settled == ["settled"]


@pytest.mark.anyio
async def test_interrupt_while_the_app_applies_an_event_finishes_the_run_before_the_next(
    tmp_path: Path,
) -> None:
    # The worker is not inside the session when it is cancelled, so the
    # session's side of the run is still suspended and has to be closed.
    provider = _ScriptedProvider([_says("First"), _says("Second")])
    session = await _load_session(tmp_path, provider)
    api = _extension_api(session)
    settle = _BlockingHook()
    api.on("agent_settled", settle)
    app = TauTuiApp(session)
    applying = asyncio.Event()
    apply_event = app._apply_streaming_transcript_event

    async def apply_slowly(event: Any) -> None:
        if not applying.is_set():
            applying.set()
            await asyncio.Event().wait()
        await apply_event(event)

    app._apply_streaming_transcript_event = apply_slowly  # type: ignore[method-assign]

    async with app.run_test(size=(120, 30)) as pilot:
        await _submit(app, "first")
        await asyncio.wait_for(applying.wait(), timeout=5)
        await pilot.press("escape")
        try:
            await asyncio.wait_for(settle.entered.wait(), timeout=5)
            await _submit(app, "second")
            await _pause(pilot)
            waiting = app._waiting_prompt_texts()
            calls_while_settling = len(provider.calls)
        finally:
            # A run left suspended is closed when the event loop shuts down,
            # and would wait in this hook then.
            settle.release.set()

        assert waiting == ("second",)
        assert calls_while_settling <= 1

        await _pause_until(pilot, lambda: _is_idle(app) and "second" in _user_texts(session))

        assert settle.finished == 2
        assert app.state.error is None


@pytest.mark.anyio
async def test_compact_is_refused_while_an_interrupted_run_settles(tmp_path: Path) -> None:
    first = _says("First", gated=True)
    provider = _ScriptedProvider([first, _says("Summary")])
    session = await _load_session(tmp_path, provider)
    api = _extension_api(session)
    settle = _BlockingHook()
    api.on("agent_settled", settle)
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await _interrupt_into_settle(app, pilot, first, settle)
        # Nothing but the run's hold on the session says it is busy now.
        assert not session.is_running
        assert not app.state.running

        await _submit(app, "/compact")
        await _pause(pilot)

        assert app.query_one("#prompt", PromptInput).text == "/compact"
        assert not app._is_compaction_active()
        assert len(provider.calls) == 1

        settle.release.set()
        await _pause_until(pilot, lambda: _is_idle(app))


@pytest.mark.anyio
async def test_app_exit_while_an_interrupted_run_settles_does_not_hang(tmp_path: Path) -> None:
    first = _says("First", gated=True)
    provider = _ScriptedProvider([first, _says("Second")])
    session = await _load_session(tmp_path, provider)
    api = _extension_api(session)
    settle = _BlockingHook()
    api.on("agent_settled", settle)
    app = TauTuiApp(session)

    async def run_and_exit() -> None:
        async with app.run_test(size=(120, 30)) as pilot:
            await _interrupt_into_settle(app, pilot, first, settle)
            await _submit(app, "second")
            await pilot.pause()

    await asyncio.wait_for(run_and_exit(), timeout=20)
    for _ in range(10):
        await asyncio.sleep(0)

    # The prompt that was waiting goes with the app instead of starting a run.
    assert len(provider.calls) == 1
    assert settle.finished == 0


@pytest.mark.anyio
async def test_extension_message_from_the_settle_hook_of_an_interrupted_run_follows_it(
    tmp_path: Path,
) -> None:
    first = _says("First", gated=True)
    provider = _ScriptedProvider([first, _says("Second")])
    session = await _load_session(tmp_path, provider)
    api = _extension_api(session)
    in_hook = asyncio.Event()
    leave_hook = asyncio.Event()
    settled: list[str] = []

    async def settle(event: object, context: object) -> None:
        del event, context
        if not settled:
            api.send_user_message("from the hook")
            in_hook.set()
            await leave_hook.wait()
        settled.append("done")

    api.on("agent_settled", settle)
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await _submit(app, "first")
        await asyncio.wait_for(first.started.wait(), timeout=5)
        await pilot.pause()
        await pilot.press("escape")
        await asyncio.wait_for(in_hook.wait(), timeout=5)
        await _pause(pilot)

        # The hook that sent the message is still running.
        assert len(provider.calls) == 1

        leave_hook.set()
        await _pause_until(pilot, lambda: len(provider.calls) == 2 and _is_idle(app))

        assert [message.text for message in provider.calls[1]].count("from the hook") == 1
        assert _user_texts(session).count("from the hook") == 1
        assert session.queued_message_count == 0
        assert settled == ["done", "done"]
