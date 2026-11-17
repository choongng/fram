"""The reasoning guard can be switched at runtime, per session."""

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from tau_agent import AssistantMessage, UserMessage
from tau_agent.messages import (
    REASONING_BUDGET_CUT,
    REASONING_LOOP_CUT,
    TOOL_CALL_LOOP_CUT,
    ThinkingContent,
)
from tau_agent.provider_events import AssistantDoneEvent
from tau_ai import FakeProvider, OpenAICompatibleConfig, OpenAICompatibleProvider
from tau_ai.loop_cuts import looping_tool_arguments_cut
from tau_coding import CodingSession
from tau_coding.commands import CommandContext, create_default_command_registry
from tau_coding.reasoning_guard import (
    REASONING_GUARD_ENV,
    TURN_REASONING_CHARS_ENV,
    ReasoningGuardSetting,
    RequestGuard,
    parse_reasoning_guard,
    startup_reasoning_guard,
)
from tau_coding.request_gate import ConcurrencyLimitedProvider
from test_output_cap_tool_call import (
    REAL_START,
    REPORT,
    _cut_parts,
    _Server,
    _tool_call_chunks,
)
from test_runtime_autocompact import _rpc
from test_session_output_limit import (
    LOOP_OPENING,
    LOOP_ROUND,
    _acts,
    _answers,
    _run,
    _session,
    _streamed_reasoning,
)

LOOPING = LOOP_OPENING + LOOP_ROUND * 40
LONG = " ".join(f"Entry {n} holds {n * 7}." for n in range(400))
NEVER = AssistantMessage(content="The reasoning ran to its end.")


def _cuts(session: CodingSession) -> list[str]:
    return [
        diagnostic.type
        for message in session.messages
        if isinstance(message, AssistantMessage)
        for diagnostic in message.diagnostics or []
    ]


def test_settings_parse() -> None:
    assert parse_reasoning_guard(" ON ") == ReasoningGuardSetting()
    assert parse_reasoning_guard("off") == ReasoningGuardSetting(enabled=False)
    assert parse_reasoning_guard("20000") == ReasoningGuardSetting(chars=20000)
    for bad in ("", "0", "-5", "1.5", "maybe", "²"):
        with pytest.raises(ValueError):
            parse_reasoning_guard(bad)


def test_startup_setting_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(REASONING_GUARD_ENV, raising=False)
    assert startup_reasoning_guard() == ReasoningGuardSetting()
    monkeypatch.setenv(REASONING_GUARD_ENV, "off")
    assert startup_reasoning_guard().enabled is False
    monkeypatch.setenv(REASONING_GUARD_ENV, "nonsense")
    assert startup_reasoning_guard() == ReasoningGuardSetting()


@pytest.mark.anyio
async def test_off_passes_the_providers_events_on_untouched() -> None:
    events = [object(), object()]
    limits_read: list[int] = []

    class Inner:
        async def stream_response(self, **kwargs: object) -> AsyncIterator[object]:
            for event in events:
                yield event

    guard = RequestGuard(
        Inner(),  # type: ignore[arg-type]
        setting=lambda: ReasoningGuardSetting(enabled=False),
        limits=lambda: limits_read.append(1),  # type: ignore[arg-type,return-value]
    )
    stream = guard.stream_response(model="m", system="s", messages=[], tools=[])
    passed = [event async for event in stream]
    assert all(got is sent for got, sent in zip(passed, events, strict=True))
    assert not limits_read


@pytest.mark.anyio
async def test_a_session_with_the_guard_off_lets_a_loop_run(tmp_path: Path) -> None:
    done = AssistantMessage(content="Finished.")
    provider = FakeProvider([_streamed_reasoning(LOOPING, then=done)])
    session = await _session(
        tmp_path, provider, reasoning_guard=ReasoningGuardSetting(enabled=False)
    )
    await _run(session)
    assert session.messages[-1].text == "Finished."
    assert not _cuts(session)


@pytest.mark.anyio
async def test_a_switch_mid_run_reaches_the_next_request(tmp_path: Path) -> None:
    provider = FakeProvider([_acts(), _streamed_reasoning(LOOPING, then=_done()), _answers("x")])
    session = await _session(tmp_path, provider)
    original = session._tool_guards.record

    def switch_off(*args: object, **kwargs: object) -> object:
        session.set_reasoning_guard(ReasoningGuardSetting(enabled=False))
        return original(*args, **kwargs)  # type: ignore[arg-type]

    session._tool_guards.record = switch_off  # type: ignore[method-assign]
    await _run(session)
    assert REASONING_LOOP_CUT not in _cuts(session)
    assert session.messages[-1].text == "Reasoned to the end."


def _done() -> AssistantMessage:
    return AssistantMessage(content="Reasoned to the end.")


@pytest.mark.anyio
async def test_the_runtime_budget_beats_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(TURN_REASONING_CHARS_ENV, "100000")
    provider = FakeProvider([_streamed_reasoning(LONG, then=NEVER), _answers("Acted.")])
    session = await _session(tmp_path, provider)
    assert session.reasoning_budget_chars == 100000
    session.set_reasoning_guard(ReasoningGuardSetting(chars=200))
    assert session.reasoning_budget_chars == 200
    await _run(session)
    assert REASONING_BUDGET_CUT in _cuts(session)
    thinking = next(
        block
        for message in session.messages
        if isinstance(message, AssistantMessage)
        for block in message.content
        if isinstance(block, ThinkingContent)
    )
    assert len(thinking.thinking) < 300


@pytest.mark.anyio
async def test_reasoning_cut_for_looping_is_not_replayed(tmp_path: Path) -> None:
    provider = FakeProvider([_streamed_reasoning(LOOPING, then=NEVER), _answers("Acted.")])
    session = await _session(tmp_path, provider)
    await _run(session)
    replayed = provider.calls[-1][2]
    assert isinstance(replayed[-1], UserMessage)
    assert not any(
        isinstance(block, ThinkingContent)
        for message in replayed
        if isinstance(message, AssistantMessage)
        for block in message.content
    )
    # The roles still alternate: the empty turn is not dropped.
    roles = [type(message) for message in replayed]
    assert roles[-3:] == [UserMessage, AssistantMessage, UserMessage]


@pytest.mark.anyio
async def test_slash_command_shows_and_sets_the_guard(tmp_path: Path) -> None:
    session = await _session(tmp_path, FakeProvider([]))
    registry = create_default_command_registry()

    async def run(args: str) -> str | None:
        command = registry.get("reasoning-guard")
        assert command is not None
        result = command.handler(
            CommandContext(
                session=session,  # type: ignore[arg-type]
                registry=registry,
                text=f"/reasoning-guard {args}".strip(),
                name="reasoning-guard",
                args=args,
            )
        )
        return result.message

    assert await run("") == "Reasoning guard: on, no thinking budget"
    assert await run("20000") == "Reasoning guard: on, 20000 characters of thinking per turn"
    assert await run("off") == "Reasoning guard: off"
    assert (await run("banana") or "").endswith("Setting unchanged. Reasoning guard: off")
    assert await run("on") == "Reasoning guard: on, no thinking budget"


@pytest.mark.anyio
async def test_rpc_sets_the_guard(tmp_path: Path) -> None:
    session = await _session(tmp_path, FakeProvider([]))
    changed, state = await _rpc(
        session,
        {"id": "set", "type": "set_reasoning_guard", "value": 20000},
        {"id": "state", "type": "get_state"},
    )
    assert changed["success"] is True
    assert changed["data"] == {"reasoningGuardEnabled": True, "reasoningBudgetChars": 20000}
    assert state["data"]["reasoningBudgetChars"] == 20000  # type: ignore[index]


@pytest.mark.anyio
@pytest.mark.parametrize("enabled", [True, False])
async def test_the_switch_governs_the_tool_argument_loop_cut(enabled: bool) -> None:
    """A provider that cuts looping tool-call arguments itself does so only
    while the guard is on, also when reached through another wrapper."""
    runaway = REAL_START + REPORT * 80
    server = _Server([_tool_call_chunks("edit", _cut_parts(runaway), "length")])
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="k", base_url="https://example.test/v1"),
            client=client,
        )
        guard = RequestGuard(
            ConcurrencyLimitedProvider(provider, provider_name="test", limit=1),
            setting=lambda: ReasoningGuardSetting(enabled=enabled),
        )
        events = [
            event
            async for event in guard.stream_response(
                model="m",
                system="s",
                messages=[UserMessage(content="Write the cases.")],
                tools=[],
            )
        ]
    done = events[-1]
    assert isinstance(done, AssistantDoneEvent)
    sent = done.message.tool_calls[0].arguments["_raw_arguments"]
    assert isinstance(sent, str)
    cuts = [diagnostic.type for diagnostic in done.message.diagnostics or []]
    if enabled:
        assert cuts == [TOOL_CALL_LOOP_CUT]
        assert len(sent) < len(runaway) // 3
    else:
        assert cuts == []
        assert sent == runaway


@pytest.mark.anyio
async def test_the_tool_argument_loop_cut_follows_the_switch_between_requests() -> None:
    """A request made after the guard is switched back on is cut again."""
    runaway = REAL_START + REPORT * 80
    chunks = _tool_call_chunks("edit", _cut_parts(runaway), "length")
    server = _Server([chunks, chunks])
    setting = ReasoningGuardSetting(enabled=False)
    cuts: list[list[str]] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="k", base_url="https://example.test/v1"),
            client=client,
        )
        guard = RequestGuard(provider, setting=lambda: setting)
        for enabled in (False, True):
            setting = ReasoningGuardSetting(enabled=enabled)
            async for event in guard.stream_response(
                model="m", system="s", messages=[UserMessage(content="Go.")], tools=[]
            ):
                if isinstance(event, AssistantDoneEvent):
                    cuts.append([d.type for d in event.message.diagnostics or []])
    assert cuts == [[], [TOOL_CALL_LOOP_CUT]]


def _shared_provider(client: httpx.AsyncClient) -> ConcurrencyLimitedProvider:
    return ConcurrencyLimitedProvider(
        OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="k", base_url="https://example.test/v1"),
            client=client,
        ),
        provider_name="shared",
        limit=1,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("first", [False, True])
async def test_two_sessions_on_one_provider_each_keep_their_own_switch(first: bool) -> None:
    """A request's choice is its own: another session's request, made on the
    same provider while this one waits for its slot, does not change it."""
    runaway = REAL_START + REPORT * 80
    chunks = _tool_call_chunks("edit", _cut_parts(runaway), "length")
    server = _Server([chunks, chunks])
    cuts: list[list[str]] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        shared = _shared_provider(client)
        streams = [
            RequestGuard(
                shared, setting=lambda enabled=enabled: ReasoningGuardSetting(enabled=enabled)
            ).stream_response(
                model="m", system="s", messages=[UserMessage(content="Go.")], tools=[]
            )
            for enabled in (first, not first)
        ]
        for stream in streams:
            done = [event async for event in stream][-1]
            assert isinstance(done, AssistantDoneEvent)
            cuts.append([d.type for d in done.message.diagnostics or []])
    expected = {True: [TOOL_CALL_LOOP_CUT], False: []}
    assert cuts == [expected[first], expected[not first]]


@pytest.mark.anyio
async def test_the_switch_does_not_outlive_the_read_of_a_response() -> None:
    """Between the events of a guarded response, and after it, the caller's
    own requests are not marked with that response's choice."""
    seen: list[bool] = []

    class Inner:
        async def stream_response(self, **kwargs: object) -> AsyncIterator[object]:
            seen.append(looping_tool_arguments_cut())
            yield object()
            seen.append(looping_tool_arguments_cut())

    guard = RequestGuard(
        Inner(),  # type: ignore[arg-type]
        setting=lambda: ReasoningGuardSetting(enabled=False),
    )
    async for _event in guard.stream_response(model="m", system="s", messages=[], tools=[]):
        assert looping_tool_arguments_cut() is True
    assert seen == [False, False]
    assert looping_tool_arguments_cut() is True


@pytest.mark.anyio
@pytest.mark.parametrize("command", ["off", "on"])
async def test_the_slash_command_reaches_a_real_providers_parse(
    tmp_path: Path, command: str
) -> None:
    """`/reasoning-guard off` in a session lets a looping tool call's arguments
    through the provider the session sends its requests to; `on` cuts them."""
    runaway = REAL_START + REPORT * 80
    server = _Server(
        [
            _tool_call_chunks("fetch", _cut_parts(runaway), "length"),
            [{"choices": [{"delta": {"content": "Done."}, "finish_reason": "stop"}]}],
        ]
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        session = await _session(tmp_path, _shared_provider(client))
        registry = create_default_command_registry()
        guard_command = registry.get("reasoning-guard")
        assert guard_command is not None
        guard_command.handler(
            CommandContext(
                session=session,  # type: ignore[arg-type]
                registry=registry,
                text=f"/reasoning-guard {command}",
                name="reasoning-guard",
                args=command,
            )
        )
        await _run(session)
    call = session.messages[1]
    assert isinstance(call, AssistantMessage)
    sent = call.tool_calls[0].arguments["_raw_arguments"]
    if command == "off":
        assert _cuts(session) == []
        assert sent == runaway
    else:
        assert _cuts(session) == [TOOL_CALL_LOOP_CUT]
        assert sent != runaway
