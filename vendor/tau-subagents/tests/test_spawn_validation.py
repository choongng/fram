"""The agent tool checks provider, model, and credentials before it reports a spawn."""

import json
from pathlib import Path

import pytest
from test_extension import (
    RecordingSession,
    _agent_tool,
    _extension_module,
    _load_runtime,
    _patch_provider_settings,
    _wait_for,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def only_deepseek_has_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every builtin provider's key env var, then give deepseek one."""
    from tau_coding import load_provider_settings

    for provider in load_provider_settings().providers:
        if provider.api_key_env:
            monkeypatch.delenv(provider.api_key_env, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")


def _get_result_tool(runtime):  # noqa: ANN001, ANN202
    return next(
        tool for tool in runtime.extension_tools if tool.name == "get_subagent_result"
    )


async def _assert_no_run(runtime) -> None:  # noqa: ANN001
    fetched = await _get_result_tool(runtime).execute("c", {"agent_id": "agent-1"})
    assert "Unknown agent_id: agent-1. Known agents: none" in fetched.text


async def test_unknown_provider_fails_the_call_and_names_authenticated_ones(
    tmp_path: Path, only_deepseek_has_credentials: None
) -> None:
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))

    result = await _agent_tool(runtime).execute(
        "call-1",
        {
            "prompt": "p",
            "description": "d",
            "run_in_background": True,
            "provider": "no-such-provider",
        },
    )

    assert "Unknown provider: no-such-provider" in result.text
    assert "Providers with credentials: deepseek." in result.text
    assert "started" not in result.text
    await _assert_no_run(runtime)


async def test_unknown_model_fails_the_call_and_lists_configured_models(
    tmp_path: Path, only_deepseek_has_credentials: None
) -> None:
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))

    result = await _agent_tool(runtime).execute(
        "call-1",
        {
            "prompt": "p",
            "description": "d",
            "provider": "deepseek",
            "model": "made-up-model",
        },
    )

    assert "Model is not configured for provider deepseek" in result.text
    assert "Available models:" in result.text
    await _assert_no_run(runtime)


async def test_provider_without_credentials_fails_the_call(
    tmp_path: Path, only_deepseek_has_credentials: None
) -> None:
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))

    result = await _agent_tool(runtime).execute(
        "call-1",
        {
            "prompt": "p",
            "description": "d",
            "run_in_background": True,
            "provider": "anthropic",
        },
    )

    assert "provider anthropic has no credentials" in result.text
    assert "Providers with credentials: deepseek." in result.text
    await _assert_no_run(runtime)


async def test_schedule_checks_the_provider_too(
    tmp_path: Path, only_deepseek_has_credentials: None
) -> None:
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))
    await runtime.emit_session_start("startup")

    result = await _agent_tool(runtime).execute(
        "call-1",
        {"prompt": "p", "description": "d", "schedule": "5m", "provider": "xai"},
    )

    assert "provider xai has no credentials" in result.text
    await runtime.emit_session_shutdown("quit")


async def test_parent_provider_skips_the_credential_check(tmp_path: Path) -> None:
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))
    module = _extension_module()
    _patch_provider_settings(module)  # resolves to provider "fake", the parent's
    checked: list[object] = []

    def never_usable(provider: object) -> bool:
        checked.append(provider)
        return False

    module._provider_is_usable = never_usable  # type: ignore[attr-defined]

    result = await _agent_tool(runtime).execute(
        "call-1", {"prompt": "p", "description": "d", "run_in_background": True}
    )
    assert "Agent started in background." in result.text
    assert checked == []


async def test_late_setup_failure_writes_a_stub_transcript(tmp_path: Path) -> None:
    runtime = _load_runtime(tmp_path)
    session = RecordingSession(tmp_path)
    runtime.bind(session)
    module = _extension_module()
    _patch_provider_settings(module)

    def failing_create(provider, model, thinking_level):  # noqa: ANN001, ANN202
        raise RuntimeError("Missing OpenAI Codex OAuth credentials")

    module.create_model_provider = failing_create  # type: ignore[attr-defined]

    result = await _agent_tool(runtime).execute(
        "call-1", {"prompt": "Do it", "description": "d", "run_in_background": True}
    )
    assert "Agent started in background." in result.text

    await _wait_for(lambda: session.followed_up)
    note = session.followed_up[0]
    assert "<status>error</status>" in note
    output_path = Path(
        note.split("<output-file>", 1)[1].split("</output-file>", 1)[0]
    )
    entries = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [entry["type"] for entry in entries] == ["user", "error"]
    assert entries[0]["message"]["content"] == "Do it"
    assert "Missing OpenAI Codex OAuth credentials" in entries[1]["message"]["content"]


def test_provider_and_model_schema_discourage_guessing(
    tmp_path: Path, only_deepseek_has_credentials: None
) -> None:
    runtime = _load_runtime(tmp_path)
    tool = next(tool for tool in runtime.extension_tools if tool.name == "agent")
    properties = tool.parameters["properties"]
    provider_text = properties["provider"]["description"]
    model_text = properties["model"]["description"]
    assert "never guess" in provider_text
    assert "never guess" in model_text
    assert "(at session start: deepseek)" in provider_text


async def test_other_provider_gets_its_own_default_model(
    tmp_path: Path, only_deepseek_has_credentials: None
) -> None:
    from tau_ai import FakeProvider
    from tau_coding import load_provider_settings
    from test_extension import _text_stream

    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))
    module = _extension_module()
    created: list[tuple[str, str]] = []

    def recording_create(provider, model, thinking_level):  # noqa: ANN001, ANN202
        created.append((provider.name, model))
        return FakeProvider([_text_stream("done")])

    module.create_model_provider = recording_create  # type: ignore[attr-defined]

    # The parent runs model "fake" on provider "fake". That model is not
    # configured for deepseek, so deepseek runs its own default model.
    result = await _agent_tool(runtime).execute(
        "call-1", {"prompt": "p", "description": "d", "provider": "deepseek"}
    )
    assert "done" in result.text
    default = load_provider_settings().get_provider("deepseek").default_model
    assert created == [("deepseek", default)]
