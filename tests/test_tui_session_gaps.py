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

from pi_event_helpers import assistant_done, assistant_error, assistant_start
from tau_agent import (
    AgentMessage,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from tau_agent.messages import AssistantMessageDiagnostic, assistant_content
from tau_agent.provider_events import AssistantErrorEvent
from tau_agent.session import JsonlSessionStorage, MessageEntry
from tau_ai import CancellationToken
from tau_ai.events import AssistantMessageEvent
from tau_coding import (
    CodingSession,
    CodingSessionConfig,
    OpenAICompatibleProviderConfig,
    ProviderSettings,
    TauPaths,
    TauResourcePaths,
)
from tau_coding import session as coding_session_module
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


@pytest.mark.anyio
@pytest.mark.parametrize(
    "stage",
    ["input", "agent_start", "turn_end", "agent_end", "agent_settled"],
)
async def test_extension_message_sent_at_any_stage_of_a_run_is_delivered_once(
    tmp_path: Path, stage: str
) -> None:
    # `input` is before the run's first event and `agent_settled` after its
    # last. At `agent_end` the agent still reports running but reads its queues
    # no more.
    provider = _ScriptedProvider([_says("First"), _says("Second"), _says("Unwanted")])
    session = await _load_session(tmp_path, provider)
    api = _extension_api(session)
    sent: list[str] = []

    def send(event: object, context: object) -> None:
        del event, context
        if not sent:
            sent.append("from extension")
            api.send_user_message("from extension")

    api.on(stage, send)
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await _submit(app, "first")
        await _pause_until(pilot, lambda: len(provider.calls) == 2 and _is_idle(app))
        await _pause(pilot)

        assert [message.text for message in provider.calls[1]] == [
            "first",
            "First",
            "from extension",
        ]
        assert len(provider.calls) == 2
        assert session.queued_message_count == 0
        assert _user_texts(session) == ["first", "from extension"]
        assert await _stored_user_texts(tmp_path) == ["first", "from extension"]


@pytest.mark.anyio
async def test_settle_hook_message_waits_behind_a_prompt_typed_while_the_run_settles(
    tmp_path: Path,
) -> None:
    provider = _ScriptedProvider([_says("First"), _says("Second"), _says("Third")])
    session = await _load_session(tmp_path, provider)
    api = _extension_api(session)
    hooked = _count_input_hooks(api)
    in_hook = asyncio.Event()
    leave_hook = asyncio.Event()
    settles: list[int] = []

    async def settle(event: object, context: object) -> None:
        del event, context
        settles.append(len(provider.calls))
        if len(settles) == 1:
            in_hook.set()
            await leave_hook.wait()
            api.send_user_message("from the hook")

    api.on("agent_settled", settle)
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await _submit(app, "first")
        await asyncio.wait_for(in_hook.wait(), timeout=5)
        await _submit(app, "typed")
        await _pause(pilot)

        assert len(provider.calls) == 1
        assert app._waiting_prompt_texts() == ("typed",)

        leave_hook.set()
        await _pause_until(pilot, lambda: len(provider.calls) == 3 and _is_idle(app))
        await _pause(pilot)

        # One run for both: the typed prompt starts it and the hook's message
        # follows as that run's follow-up.
        assert _user_texts(session) == ["first", "typed", "from the hook"]
        assert await _stored_user_texts(tmp_path) == ["first", "typed", "from the hook"]
        assert hooked == ["first", "typed"]
        assert settles == [1, 3]
        assert session.queued_message_count == 0


_OVERFLOW_ERROR = "This model's maximum context length was exceeded."


@pytest.mark.anyio
async def test_prompt_and_extension_message_during_overflow_compaction_join_the_retry(
    tmp_path: Path,
) -> None:
    summary = _says("Summary of the earlier work", gated=True)
    provider = _ScriptedProvider(
        [
            _says("First answer"),
            _says("Second answer"),
            _Reply([assistant_error(message=_OVERFLOW_ERROR)]),
            summary,
            _says("Recovered answer"),
            _says("Follow-up answer"),
        ]
    )
    session = await _load_session(tmp_path, provider)
    # Compaction needs earlier turns to summarize.
    async for _ in session.prompt("Collect context.\n" + ("old context " * 12_000)):
        pass
    async for _ in session.prompt("Keep this recent turn."):
        pass
    api = _extension_api(session)
    hooked = _count_input_hooks(api)
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await _submit(app, "Trigger overflow.")
        # The summary request is under way: the failed agent run is over and
        # the retry has not started, so nothing can queue a message.
        await asyncio.wait_for(summary.started.wait(), timeout=10)
        assert not session.is_running
        await _submit(app, "typed during compaction")
        api.send_user_message("sent during compaction")
        await _pause(pilot)

        assert len(provider.calls) == 4
        assert app._waiting_prompt_texts() == ("typed during compaction",)
        assert session.queued_message_count == 0

        summary.gate.set()  # type: ignore[union-attr]
        await _pause_until(pilot, lambda: len(provider.calls) == 6 and _is_idle(app))
        await _pause(pilot)

        late = ["typed during compaction", "sent during compaction"]
        assert [text for text in _user_texts(session) if text in late] == late
        assert [text for text in await _stored_user_texts(tmp_path) if text in late] == late
        # The typed prompt steers the retry from its first request on; the
        # extension's message follows that answer.
        assert [message.text for message in provider.calls[4]][-2:] == [
            "Trigger overflow.",
            "typed during compaction",
        ]
        assert [message.text for message in provider.calls[5]][-2:] == [
            "Recovered answer",
            "sent during compaction",
        ]
        assert hooked == ["Trigger overflow.", "typed during compaction"]
        assert app._waiting_prompt_texts() == ()
        assert session.queued_message_count == 0


@pytest.mark.anyio
async def test_prompt_during_route_failover_joins_the_rerouted_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = "moonshotai/Kimi-K3"
    rate_limited = AssistantMessage(
        stop_reason="error",
        error_message="Rate limit exceeded",
        diagnostics=[
            AssistantMessageDiagnostic(type="provider_error", details={"status_code": 429})
        ],
    )
    failing = _ScriptedProvider([_Reply([AssistantErrorEvent(reason="error", error=rate_limited)])])
    rerouted = _ScriptedProvider([_says("Rerouted answer")])

    def create_provider(
        provider_config: object, *, inference_provider: str | None = None, **options: object
    ) -> _ScriptedProvider:
        del provider_config, options
        return rerouted if inference_provider is None else failing

    monkeypatch.setattr(coding_session_module, "create_model_provider", create_provider)
    provider_config = OpenAICompatibleProviderConfig(
        name="huggingface", models=(model,), default_model=model
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=_ScriptedProvider([]),  # type: ignore[arg-type]
            model=model,
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            provider_name="huggingface",
            inference_provider="deepinfra",
            inference_provider_mode="automatic",
            provider_settings=ProviderSettings(providers=(provider_config,)),
            runtime_provider_config=provider_config,
            resource_paths=TauResourcePaths(
                root=tmp_path / ".tau",
                paths=TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents"),
            ),
        )
    )
    api = _extension_api(session)
    hooked = _count_input_hooks(api)
    # The failed run is over when the retry is announced, and the rerouted one
    # starts only after the handlers of that announcement return.
    rerouting = _BlockingHook()
    api.on("auto_retry_start", rerouting)
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await _submit(app, "first")
        await asyncio.wait_for(rerouting.entered.wait(), timeout=5)
        assert not session.is_running
        await _submit(app, "typed during failover")
        await _pause(pilot)

        assert rerouted.calls == []
        assert app._waiting_prompt_texts() == ("typed during failover",)

        rerouting.release.set()
        await _pause_until(pilot, lambda: len(rerouted.calls) == 1 and _is_idle(app))
        await _pause(pilot)

        assert [message.text for message in rerouted.calls[0]].count("typed during failover") == 1
        assert _user_texts(session) == ["first", "typed during failover"]
        assert await _stored_user_texts(tmp_path) == ["first", "typed during failover"]
        assert hooked == ["first", "typed during failover"]
        assert app._waiting_prompt_texts() == ()
        assert session.queued_message_count == 0


def _fetch_tool(text: str) -> AgentTool:
    async def fetch(
        tool_call_id: object,
        arguments: object,
        signal: object = None,
        on_update: object = None,
    ) -> AgentToolResult:
        del tool_call_id, arguments, signal, on_update
        return AgentToolResult(content=[TextContent(text=text)])

    return AgentTool(
        name="fetch",
        label="Fetch",
        description="Return a large document.",
        parameters={"type": "object"},
        execute_fn=fetch,
    )


@pytest.mark.anyio
async def test_prompt_during_compaction_between_turns_is_queued_with_the_run(
    tmp_path: Path,
) -> None:
    # Not a gap: the agent compacts between two of its own turns and reads its
    # queues again afterwards, so the session takes the prompt directly.
    summary = _says("Mid-run summary", gated=True)
    fetching = AssistantMessage(
        content=assistant_content("Fetching.", [ToolCall(id="call-1", name="fetch", arguments={})]),
        stop_reason="toolUse",
    )
    provider = _ScriptedProvider(
        [
            _says("First answer"),
            _Reply([assistant_start(model="fake"), assistant_done(message=fetching)]),
            summary,
            _says("Done"),
        ]
    )
    session = await _load_session(
        tmp_path,
        provider,
        tools=[_fetch_tool("fetched " * 6_000)],
        auto_compact_token_threshold=37_000,
    )
    async for _ in session.prompt("Explain.\n" + ("old context " * 12_000)):
        pass
    api = _extension_api(session)
    hooked = _count_input_hooks(api)
    app = TauTuiApp(session)

    async with app.run_test(size=(120, 30)) as pilot:
        await _submit(app, "Fetch it.")
        await asyncio.wait_for(summary.started.wait(), timeout=10)
        assert session.is_running
        await _submit(app, "typed during compaction")
        await _pause(pilot)

        assert app._waiting_prompt_texts() == ()
        assert session.queued_steering_messages == ("typed during compaction",)

        summary.gate.set()  # type: ignore[union-attr]
        await _pause_until(pilot, lambda: len(provider.calls) == 4 and _is_idle(app))
        await _pause(pilot)

        assert [message.text for message in provider.calls[3]].count("typed during compaction") == 1
        assert _user_texts(session).count("typed during compaction") == 1
        assert (await _stored_user_texts(tmp_path)).count("typed during compaction") == 1
        assert hooked == ["Fetch it.", "typed during compaction"]
        assert session.queued_message_count == 0
