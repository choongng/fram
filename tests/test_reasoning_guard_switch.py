"""The reasoning guard can be switched at runtime, per session."""

from pathlib import Path

import pytest

from tau_agent import AssistantMessage, UserMessage
from tau_agent.messages import REASONING_BUDGET_CUT, REASONING_LOOP_CUT, ThinkingContent
from tau_ai import FakeProvider
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


def test_off_hands_the_providers_own_stream_back() -> None:
    sentinel = object()
    limits_read: list[int] = []

    class Inner:
        def stream_response(self, **kwargs: object) -> object:
            return sentinel

    guard = RequestGuard(
        Inner(),  # type: ignore[arg-type]
        setting=lambda: ReasoningGuardSetting(enabled=False),
        limits=lambda: limits_read.append(1),  # type: ignore[arg-type,return-value]
    )
    stream = guard.stream_response(model="m", system="s", messages=[], tools=[])
    assert stream is sentinel
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
