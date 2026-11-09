"""The auto-compact threshold bound follows the currently selected model.

The TUI model picker only *previews* a cross-provider choice until the next
prompt. Looking the context window up under the committed runtime's provider
misses the previewed model and falls back to the default window, which would
refuse a threshold the previewed model has room for.

Criteria under test:

- After a model switch by any path (TUI preview, committed provider change,
  same-provider ``/model``), ``context_window_tokens`` is the selected model's
  window, so ``/autocompact 250000`` is accepted on a 1M model and ``/status``
  reports that window.
- A dynamic (extension) provider model's declared window is used too.
- A user-set threshold above a newly selected model's own limit (its window
  less the compaction reserve) is capped at that limit while the model is
  active, and comes back when a model with room for it is selected again.
- A model no catalog declares has no limit to cap at: a user-set threshold is
  used as given, a percentage (of a guessed window) is refused, and
  ``context_window_source`` says the window is the default.
- A window declared only in the model's catalog metadata counts as declared.
- Without a user-set threshold, compaction fires at the model's limit, at most
  ``DEFAULT_AUTO_COMPACT_TOKEN_CAP``, so a 1M-token model compacts at all.
- The catalog lookup for a dashed Anthropic id resolves to its declared window.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tau_agent.session import JsonlSessionStorage
from tau_ai import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig
from tau_coding.context_window import (
    DEFAULT_AUTO_COMPACT_TOKEN_CAP,
    DEFAULT_COMPACTION_RESERVE_TOKENS,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
)
from tau_coding.extensions import DynamicProvider, OpenAICompatibleTransport, ProviderModel
from tau_coding.paths import TauPaths
from tau_coding.provider_config import (
    OpenAICompatibleProviderConfig,
    ProviderModelMetadata,
    ProviderSettings,
    load_provider_settings,
)
from tau_coding.session import ModelChoice

SMALL_WINDOW = 64_000
BIG_WINDOW = 1_000_000


def _provider(name: str, windows: dict[str, int]) -> OpenAICompatibleProviderConfig:
    models = tuple(windows)
    return OpenAICompatibleProviderConfig(
        name=name,
        api_key_env="LOCAL_API_KEY",
        credential_name=None,
        models=models,
        default_model=models[0],
        context_windows=windows,
    )


@pytest.fixture
async def session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> CodingSession:
    monkeypatch.setenv("LOCAL_API_KEY", "test-key")
    monkeypatch.setenv("TAU_HOME", str(tmp_path / ".tau"))
    local = _provider("local", {"small": SMALL_WINDOW, "small-big": BIG_WINDOW})
    cloud = _provider("cloud", {"big": BIG_WINDOW})
    loaded = await CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider([]),
            model="small",
            provider_name="local",
            provider_settings=ProviderSettings(providers=(local, cloud)),
            runtime_provider_config=local,
            system="You are Tau.",
            cwd=tmp_path,
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            extensions_enabled=False,
        )
    )
    yield loaded
    await loaded.aclose()


def _mentions(message: str | None, number: int) -> bool:
    assert message is not None
    return str(number) in message.replace(",", "").replace("_", "")


def _assert_250k_accepted(session: CodingSession) -> None:
    assert session.context_window_tokens == BIG_WINDOW
    result = session.handle_command("/autocompact 250000")
    assert result.message is not None
    assert "out of range" not in result.message
    assert session.auto_compact_token_threshold == 250_000
    assert _mentions(session.handle_command("/autocompact").message, BIG_WINDOW)
    assert _mentions(session.handle_command("/session").message, BIG_WINDOW)


@pytest.mark.anyio
async def test_starting_window_is_the_small_models(session: CodingSession) -> None:
    assert session.context_window_tokens == SMALL_WINDOW
    message = session.handle_command("/autocompact 250000").message
    assert message is not None and "out of range" in message


@pytest.mark.anyio
async def test_tui_preview_of_other_provider_uses_its_window(session: CodingSession) -> None:
    session.preview_model_choice(ModelChoice("cloud", "big"))

    assert session.has_pending_selection
    _assert_250k_accepted(session)


@pytest.mark.anyio
async def test_committed_provider_switch_uses_new_window(session: CodingSession) -> None:
    await session.select_provider_model(ModelChoice("cloud", "big"), persist_default=False)

    _assert_250k_accepted(session)


@pytest.mark.anyio
async def test_set_model_choice_across_providers_uses_new_window(session: CodingSession) -> None:
    session.set_model_choice(ModelChoice("cloud", "big"))

    _assert_250k_accepted(session)


@pytest.mark.anyio
async def test_same_provider_model_switch_uses_new_window(session: CodingSession) -> None:
    session.set_model("small-big")

    _assert_250k_accepted(session)


@pytest.mark.anyio
async def test_dynamic_provider_model_window_is_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TAU_HOME", str(tmp_path / ".tau"))
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider([]),
            model="wide",
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "dynamic.jsonl"),
            cwd=tmp_path,
            provider_name="local",
            provider_settings=ProviderSettings(),
        )
    )
    assert session.context_window_tokens == DEFAULT_CONTEXT_WINDOW_TOKENS
    session.extension_runtime.provider_registry.register(
        "test",
        DynamicProvider(
            id="local",
            display_name="Local",
            models=(ProviderModel("wide", context_window=BIG_WINDOW),),
            default_model="wide",
            transport=OpenAICompatibleTransport("http://localhost:8080/v1"),
        ),
    )

    _assert_250k_accepted(session)
    await session.aclose()


@pytest.mark.anyio
async def test_threshold_too_big_for_new_model_is_clamped_then_restored(
    session: CodingSession,
) -> None:
    session.set_model("small-big")
    session.handle_command("/autocompact 250000")
    assert session.auto_compact_token_threshold == 250_000

    session.set_model("small")
    assert session.context_window_tokens == SMALL_WINDOW
    assert session.auto_compact_token_threshold == SMALL_WINDOW - DEFAULT_COMPACTION_RESERVE_TOKENS

    session.set_model("small-big")
    assert session.auto_compact_token_threshold == 250_000


@pytest.mark.anyio
async def test_threshold_inside_the_reserve_is_capped_at_the_models_limit(
    session: CodingSession,
) -> None:
    session.handle_command(f"/autocompact {SMALL_WINDOW - 1}")

    assert session.auto_compact_token_threshold == SMALL_WINDOW - DEFAULT_COMPACTION_RESERVE_TOKENS


@pytest.mark.anyio
async def test_default_threshold_is_the_models_limit_up_to_the_cap(
    session: CodingSession,
) -> None:
    assert session.auto_compact_token_threshold == SMALL_WINDOW - DEFAULT_COMPACTION_RESERVE_TOKENS

    session.set_model("small-big")

    assert session.auto_compact_token_threshold == DEFAULT_AUTO_COMPACT_TOKEN_CAP


@pytest.mark.anyio
async def test_undeclared_model_keeps_a_user_set_threshold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("TAU_HOME", str(tmp_path / ".tau"))
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider([]),
            model="unlisted",
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "unlisted.jsonl"),
            cwd=tmp_path,
            provider_name="local",
            provider_settings=ProviderSettings(),
        )
    )
    assert session.declared_context_window_tokens is None
    assert "default" in session.context_window_source

    result = session.handle_command("/autocompact 400000")

    assert result.message is not None and "out of range" not in result.message
    assert session.auto_compact_token_threshold == 400_000
    refused = session.handle_command("/autocompact 50%").message
    assert refused is not None and "percentage" in refused
    assert session.auto_compact_token_threshold == 400_000
    await session.aclose()


@pytest.mark.anyio
async def test_window_declared_in_model_metadata_is_used(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LOCAL_API_KEY", "test-key")
    local = OpenAICompatibleProviderConfig(
        name="local",
        api_key_env="LOCAL_API_KEY",
        credential_name=None,
        models=("meta",),
        default_model="meta",
        model_metadata={"meta": ProviderModelMetadata(context_window=SMALL_WINDOW)},
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider([]),
            model="meta",
            provider_name="local",
            provider_settings=ProviderSettings(providers=(local,)),
            system="You are Tau.",
            cwd=tmp_path,
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            extensions_enabled=False,
        )
    )

    assert session.context_window_tokens == SMALL_WINDOW
    assert session.context_window_source == "configured catalog"
    await session.aclose()


@pytest.mark.anyio
async def test_threshold_within_new_window_is_kept(session: CodingSession) -> None:
    session.handle_command("/autocompact 50000")
    session.set_model("small-big")

    assert session.auto_compact_token_threshold == 50_000


def test_catalog_lookup_resolves_dashed_anthropic_id(tmp_path: Path) -> None:
    tau_home = tmp_path / ".tau"
    tau_home.mkdir()
    (tau_home / "catalog.toml").write_text(
        """
schema_version = 1

[[providers]]
name = "anthropic"
display_name = "Anthropic"
kind = "anthropic"
base_url = "https://api.anthropic.com"
api_key_env = "ANTHROPIC_API_KEY"
credential_name = "anthropic"
models = ["claude-sonnet-4-6", "claude-opus-5-5"]
default_model = "claude-sonnet-4-6"
docs_url = "https://docs.anthropic.com"

[providers.context_windows]
claude-sonnet-4-6 = 200000
claude-opus-5-5 = 1000000
""".strip(),
        encoding="utf-8",
    )

    provider = load_provider_settings(TauPaths(home=tau_home)).get_provider("anthropic")

    assert "claude-opus-5-5" in provider.models
    assert provider.context_windows["claude-opus-5-5"] == BIG_WINDOW
