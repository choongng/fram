"""Show and change the auto-compact threshold in a running session.

Criteria under test:

- ``/autocompact`` with no argument shows the current threshold.
- ``/autocompact <tokens>`` or ``/autocompact <percent>%`` (of the context
  window) sets it on the running session, without a restart.
- ``/autocompact off`` disables auto-compaction.
- Out-of-range or malformed values are rejected with a message and leave the
  setting unchanged.
- The session's next compaction decision uses the new value.
- RPC already exposes compaction settings (``set_auto_compaction``,
  ``autoCompactionEnabled``), so RPC gets the same capability: a
  ``set_auto_compaction_threshold`` command whose ``threshold`` is a token
  count, a ``"<percent>%"`` string or ``"off"``, and ``get_state`` reporting
  ``autoCompactionThreshold``.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import pytest

from pi_event_helpers import assistant_done, assistant_start
from tau_agent import AssistantMessage
from tau_agent.session import JsonlSessionStorage
from tau_ai import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig
from tau_coding.commands import create_default_command_registry
from tau_coding.context_window import DEFAULT_CONTEXT_WINDOW_TOKENS
from tau_coding.provider_config import OpenAICompatibleProviderConfig, ProviderSettings
from tau_coding.rpc import RpcServer

LARGE_PROMPT = "Explain sessions.\n" + ("old context " * 12_000)


async def _session(
    tmp_path: Path,
    provider: FakeProvider | None = None,
    *,
    threshold: int | None = None,
    enabled: bool = True,
) -> CodingSession:
    # The catalog declares the fake model's window, so the bounds and
    # percentages below have a window to be measured against.
    local = OpenAICompatibleProviderConfig(
        name="local",
        api_key_env="LOCAL_API_KEY",
        credential_name=None,
        models=("fake",),
        default_model="fake",
        context_windows={"fake": DEFAULT_CONTEXT_WINDOW_TOKENS},
    )
    return await CodingSession.load(
        CodingSessionConfig(
            provider=provider if provider is not None else FakeProvider([]),
            model="fake",
            provider_name="local",
            provider_settings=ProviderSettings(providers=(local,)),
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            auto_compact_token_threshold=threshold,
            auto_compact_enabled=enabled,
        )
    )


def _answer(text: str) -> list[object]:
    return [assistant_start(model="fake"), assistant_done(message=AssistantMessage(content=text))]


async def _drain(stream: object) -> None:
    async for _event in stream:  # type: ignore[attr-defined]
        pass


async def _compactions(tmp_path: Path) -> list[object]:
    entries = await JsonlSessionStorage(tmp_path / "session.jsonl").read_all()
    return [entry for entry in entries if entry.type == "compaction"]


def _mentions(message: str | None, number: int) -> bool:
    assert message is not None
    return str(number) in message.replace(",", "").replace("_", "")


async def _rpc(session: CodingSession, *commands: dict[str, object]) -> list[dict[str, object]]:
    stdin = StringIO("".join(json.dumps(command) + "\n" for command in commands))
    stdout = StringIO()
    await RpcServer(session, stdin=stdin, stdout=stdout).run()
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


# --- one case per operation ---------------------------------------------------


def test_autocompact_is_a_registered_slash_command() -> None:
    assert create_default_command_registry().get("autocompact") is not None


@pytest.mark.anyio
async def test_show_reports_the_current_threshold(tmp_path: Path) -> None:
    session = await _session(tmp_path, threshold=50_000)

    result = session.handle_command("/autocompact")

    assert result.handled is True
    assert _mentions(result.message, 50_000)
    assert session.auto_compact_token_threshold == 50_000


@pytest.mark.anyio
async def test_set_token_count_updates_the_running_session(tmp_path: Path) -> None:
    session = await _session(tmp_path)

    result = session.handle_command("/autocompact 50000")

    assert result.handled is True
    assert result.message
    assert session.auto_compact_token_threshold == 50_000


@pytest.mark.anyio
async def test_set_percentage_of_context_window(tmp_path: Path) -> None:
    session = await _session(tmp_path)
    window = session.context_window_tokens
    assert window % 2 == 0

    result = session.handle_command("/autocompact 50%")

    assert result.handled is True
    assert session.auto_compact_token_threshold == window // 2


@pytest.mark.anyio
async def test_off_disables_auto_compaction(tmp_path: Path) -> None:
    session = await _session(tmp_path, threshold=50_000)

    result = session.handle_command("/autocompact off")

    assert result.handled is True
    assert result.message
    assert session.auto_compaction_enabled is False
    assert session.auto_compact_token_threshold is None


@pytest.mark.anyio
async def test_rpc_sets_threshold_when_rpc_exposes_compaction_settings(tmp_path: Path) -> None:
    session = await _session(tmp_path)

    changed, state = await _rpc(
        session,
        {"id": "set", "type": "set_auto_compaction_threshold", "threshold": 50_000},
        {"id": "state", "type": "get_state"},
    )

    assert changed["success"] is True
    assert session.auto_compact_token_threshold == 50_000
    assert state["data"]["autoCompactionThreshold"] == 50_000  # type: ignore[index]


@pytest.mark.anyio
async def test_rpc_off_disables_auto_compaction(tmp_path: Path) -> None:
    session = await _session(tmp_path, threshold=50_000)

    changed, state = await _rpc(
        session,
        {"id": "off", "type": "set_auto_compaction_threshold", "threshold": "off"},
        {"id": "state", "type": "get_state"},
    )

    assert changed["success"] is True
    assert session.auto_compaction_enabled is False
    assert state["data"]["autoCompactionEnabled"] is False  # type: ignore[index]


@pytest.mark.anyio
async def test_rpc_sets_percentage_of_context_window(tmp_path: Path) -> None:
    session = await _session(tmp_path)
    window = session.context_window_tokens
    assert window % 2 == 0

    (changed,) = await _rpc(
        session, {"id": "pct", "type": "set_auto_compaction_threshold", "threshold": "50%"}
    )

    assert changed["success"] is True
    assert session.auto_compact_token_threshold == window // 2


# --- rules ----------------------------------------------------------------------


@pytest.mark.anyio
async def test_set_then_show_round_trips(tmp_path: Path) -> None:
    session = await _session(tmp_path)

    session.handle_command("/autocompact 42000")
    shown = session.handle_command("/autocompact")

    assert _mentions(shown.message, 42_000)


@pytest.mark.anyio
async def test_show_after_off_says_off(tmp_path: Path) -> None:
    session = await _session(tmp_path, threshold=50_000)

    session.handle_command("/autocompact off")
    shown = session.handle_command("/autocompact")

    assert shown.handled is True
    assert shown.message is not None
    assert "off" in shown.message.lower()


@pytest.mark.anyio
async def test_setting_a_value_after_off_re_enables(tmp_path: Path) -> None:
    session = await _session(tmp_path, threshold=50_000)

    session.handle_command("/autocompact off")
    session.handle_command("/autocompact 30000")

    assert session.auto_compaction_enabled is True
    assert session.auto_compact_token_threshold == 30_000


@pytest.mark.anyio
async def test_show_does_not_change_the_setting(tmp_path: Path) -> None:
    session = await _session(tmp_path, threshold=50_000)

    result = session.handle_command("/autocompact")

    assert result.handled is True
    assert session.auto_compact_token_threshold == 50_000
    assert session.auto_compaction_enabled is True


@pytest.mark.anyio
async def test_next_compaction_decision_uses_lowered_threshold(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            _answer("First answer"),
            _answer("Second answer"),
            _answer("Generated automatic summary"),
            _answer("Third answer"),
        ]
    )
    session = await _session(tmp_path, provider)
    await _drain(session.prompt(LARGE_PROMPT))
    await _drain(session.prompt("Continue."))
    assert session.context_token_estimate < session.auto_compact_token_threshold  # type: ignore[operator]
    assert await _compactions(tmp_path) == []

    session.handle_command("/autocompact 1000")
    await _drain(session.prompt("Next."))

    assert len(await _compactions(tmp_path)) == 1


@pytest.mark.anyio
async def test_next_compaction_decision_respects_off(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            _answer("First answer"),
            _answer("Second answer"),
            _answer("Third answer"),
        ]
    )
    session = await _session(tmp_path, provider, threshold=1)
    await _drain(session.prompt(LARGE_PROMPT))
    await _drain(session.prompt("Continue."))
    assert await _compactions(tmp_path) == []

    session.handle_command("/autocompact off")
    await _drain(session.prompt("Next."))

    assert await _compactions(tmp_path) == []
    assert len(provider.calls) == 3


@pytest.mark.anyio
async def test_next_compaction_decision_uses_raised_threshold(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            _answer("First answer"),
            _answer("Second answer"),
            _answer("Third answer"),
        ]
    )
    session = await _session(tmp_path, provider, threshold=1)
    await _drain(session.prompt(LARGE_PROMPT))
    await _drain(session.prompt("Continue."))
    assert await _compactions(tmp_path) == []
    assert session.context_token_estimate < session.context_window_tokens

    session.handle_command("/autocompact 100%")
    await _drain(session.prompt("Next."))

    assert await _compactions(tmp_path) == []
    assert len(provider.calls) == 3


# --- rejected values: message, setting unchanged ----------------------------------


@pytest.mark.anyio
@pytest.mark.parametrize(
    "argument",
    [
        "abc",
        "1.5",
        "%",
        "50%%",
        "fifty",
        "10 20",
        "0",
        "-5",
        "0%",
        "-10%",
        "101%",
        "999999999999",
    ],
)
async def test_rejected_value_leaves_setting_unchanged(tmp_path: Path, argument: str) -> None:
    session = await _session(tmp_path, threshold=50_000)

    result = session.handle_command(f"/autocompact {argument}")

    assert result.handled is True
    assert result.message
    assert session.auto_compact_token_threshold == 50_000
    assert session.auto_compaction_enabled is True


@pytest.mark.anyio
async def test_token_count_above_context_window_is_rejected(tmp_path: Path) -> None:
    session = await _session(tmp_path, threshold=50_000)
    too_big = session.context_window_tokens + 1

    result = session.handle_command(f"/autocompact {too_big}")

    assert result.handled is True
    assert result.message
    assert session.auto_compact_token_threshold == 50_000


@pytest.mark.anyio
async def test_rejected_value_while_off_stays_off(tmp_path: Path) -> None:
    session = await _session(tmp_path, threshold=50_000)
    session.handle_command("/autocompact off")

    result = session.handle_command("/autocompact abc")

    assert result.message
    assert session.auto_compaction_enabled is False
    assert session.auto_compact_token_threshold is None


@pytest.mark.anyio
@pytest.mark.parametrize("threshold", ["abc", "0", 0, -5, "101%", 1.5, True, None])
async def test_rpc_rejects_bad_threshold_and_leaves_setting_unchanged(
    tmp_path: Path, threshold: object
) -> None:
    session = await _session(tmp_path, threshold=50_000)

    (changed,) = await _rpc(
        session, {"id": "bad", "type": "set_auto_compaction_threshold", "threshold": threshold}
    )

    assert changed["success"] is False
    assert changed.get("error")
    assert session.auto_compact_token_threshold == 50_000
    assert session.auto_compaction_enabled is True

    # The command itself exists: a valid value through it is accepted.
    (accepted,) = await _rpc(
        session, {"id": "ok", "type": "set_auto_compaction_threshold", "threshold": 40_000}
    )
    assert accepted["success"] is True
    assert session.auto_compact_token_threshold == 40_000
