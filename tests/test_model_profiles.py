"""Per-model profiles: the measurements, their store, the probe and the CLI."""

import ast
import json
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest
from typer.testing import CliRunner

from pi_event_helpers import assistant_done, assistant_error, assistant_start
from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    AssistantMessageDiagnostic,
    ResponseTiming,
    TextContent,
    ThinkingContent,
    ToolCall,
    Usage,
)
from tau_agent.provider_events import TextDeltaEvent
from tau_agent.session import JsonlSessionStorage
from tau_agent.tools import AgentTool
from tau_ai import FakeProvider
from tau_ai.events import AssistantMessageEvent
from tau_ai.model_limits import RuntimeModelLimits
from tau_ai.model_profile import (
    MIN_MEASURED_TOKENS,
    AveragedValue,
    ModelProfile,
    ProfileKey,
    averaged,
    samples_from_response,
)
from tau_coding import CodingSession, CodingSessionConfig
from tau_coding.cli import app
from tau_coding.context_window import CHARS_PER_TOKEN
from tau_coding.model_probe import CAP_MAX_TOKENS, ProbeError, cap_honoured, probe_model
from tau_coding.model_profiles import (
    MIN_TRUSTED_RATIO_WEIGHT,
    ModelProfileStore,
    context_chars_per_token,
    read_model_profiles,
)
from tau_coding.paths import TauPaths

KEY = ProfileKey(provider="local", endpoint="http://localhost:8080/v1", model="test-model")


def _response(
    *,
    text: str = "",
    thinking: str = "",
    call: ToolCall | None = None,
    usage: Usage,
    first_ms: int | None = 1000,
    total_ms: int = 11_000,
) -> AssistantMessage:
    content: list[TextContent | ThinkingContent | ToolCall] = []
    if thinking:
        content.append(ThinkingContent(thinking=thinking))
    if text:
        content.append(TextContent(text=text))
    if call is not None:
        content.append(call)
    return AssistantMessage(
        content=content,
        usage=usage,
        timing=ResponseTiming(time_to_first_output_ms=first_ms, total_duration_ms=total_ms),
        stop_reason="toolUse" if call is not None else "stop",
    )


def test_a_response_yields_decode_prefill_and_chars_per_token_samples() -> None:
    message = _response(text="x" * 2000, usage=Usage(input=2000, output=500))

    samples = samples_from_response(message)

    assert samples is not None
    assert samples.context_tokens == 2000
    # 500 output tokens over the 10 s after the first output.
    assert samples.decode_tokens_per_second is not None
    assert samples.decode_tokens_per_second.value == pytest.approx(50.0)
    assert samples.decode_tokens_per_second.tokens == 500
    # 2000 uncached prompt tokens read before the first output at 1 s.
    assert samples.prefill_tokens_per_second is not None
    assert samples.prefill_tokens_per_second.value == pytest.approx(2000.0)
    assert samples.output_chars_per_token is not None
    assert samples.output_chars_per_token.value == pytest.approx(4.0)
    assert samples.combined_chars_per_token is not None
    assert samples.combined_chars_per_token.value == pytest.approx(4.0)

    profile = ModelProfile(key=KEY).with_message(message, now_ms=123)
    assert profile.decode_tokens_per_second["lt_8k"].value == pytest.approx(50.0)
    assert profile.prefill_tokens_per_second == AveragedValue(2000.0, 1, 123, 2000)
    assert profile.largest_output_tokens == 500
    assert profile.updated_at_ms == 123


def test_thinking_and_output_keep_separate_chars_per_token() -> None:
    call = ToolCall(id="c1", name="write", arguments={"path": "a.py", "content": "y" * 360})
    call_chars = len("write") + len(
        json.dumps(call.arguments, ensure_ascii=False, separators=(",", ":"))
    )
    reported = _response(
        thinking="t" * 900,
        text="o" * 40,
        call=call,
        usage=Usage(input=1000, output=400, reasoning=300),
    )

    samples = samples_from_response(reported)

    assert samples is not None
    assert samples.thinking_chars_per_token is not None
    assert samples.thinking_chars_per_token.value == pytest.approx(3.0)
    assert samples.output_chars_per_token is not None
    assert samples.output_chars_per_token.value == pytest.approx((40 + call_chars) / 100)
    profile = ModelProfile(key=KEY).with_message(reported, now_ms=1)
    assert profile.thinking_chars_per_token is not None
    assert profile.output_chars_per_token is not None
    assert profile.thinking_chars_per_token.value != profile.output_chars_per_token.value

    # Unreported reasoning with both kinds present cannot be split: only the
    # combined ratio is sampled.
    mixed = _response(thinking="t" * 900, text="o" * 400, usage=Usage(input=1000, output=400))
    mixed_samples = samples_from_response(mixed)
    assert mixed_samples is not None
    assert mixed_samples.thinking_chars_per_token is None
    assert mixed_samples.output_chars_per_token is None
    assert mixed_samples.combined_chars_per_token is not None
    assert mixed_samples.combined_chars_per_token.value == pytest.approx(1300 / 400)

    # A response of thinking alone gives all its output tokens to thinking.
    only_thinking = _response(thinking="t" * 600, usage=Usage(input=1000, output=200))
    only_samples = samples_from_response(only_thinking)
    assert only_samples is not None
    assert only_samples.thinking_chars_per_token is not None
    assert only_samples.thinking_chars_per_token.value == pytest.approx(3.0)
    assert only_samples.output_chars_per_token is None


def test_cached_prompt_tokens_are_not_counted_as_prefill() -> None:
    mostly_cached = _response(text="x" * 400, usage=Usage(input=300, cache_read=20_000, output=100))
    samples = samples_from_response(mostly_cached)
    assert samples is not None
    assert samples.prefill_tokens_per_second is None
    assert samples.context_tokens == 20_300

    # Partly cached: only the fresh input and the cache writes were read.
    partly = _response(
        text="x" * 400, usage=Usage(input=1000, cache_read=500, cache_write=200, output=100)
    )
    partly_samples = samples_from_response(partly)
    assert partly_samples is not None
    assert partly_samples.prefill_tokens_per_second is not None
    assert partly_samples.prefill_tokens_per_second.tokens == 1200
    assert partly_samples.prefill_tokens_per_second.value == pytest.approx(1200.0)


def test_a_response_without_reported_usage_is_skipped(tmp_path: Path) -> None:
    no_usage = _response(text="hello", usage=Usage())
    estimated = _response(text="x" * 400, usage=Usage(input=1000, output=100))
    estimated.diagnostics = [
        AssistantMessageDiagnostic(type="reasoning_budget_cut", details={"usage_estimated": True})
    ]
    errored = _response(text="x", usage=Usage(input=1000, output=10))
    errored.stop_reason = "error"

    for message in (no_usage, estimated, errored):
        assert samples_from_response(message) is None
        profile = ModelProfile(key=KEY)
        assert profile.with_message(message, now_ms=5) is profile

    store = ModelProfileStore(tmp_path / "model-profiles.json")
    assert store.record_response(KEY, no_usage) is None
    assert not store.path.exists()


def test_the_average_weights_by_tokens_and_clips_outliers() -> None:
    first = averaged(None, 10.0, tokens=50, now_ms=1)
    assert first == AveragedValue(value=10.0, samples=1, updated_at_ms=1, weight=50)

    # Past the warm-up, alpha = 1 - 0.9 ** (tokens / 1000).
    settled = AveragedValue(value=10.0, samples=30, updated_at_ms=1, weight=50_000)
    after_1k = settled.updated(20.0, tokens=1000, now_ms=2)
    assert after_1k.value == pytest.approx(11.0)
    assert after_1k.samples == 31
    assert after_1k.weight == 51_000
    assert after_1k.updated_at_ms == 2
    after_2k = settled.updated(20.0, tokens=2000, now_ms=2)
    assert after_2k.value == pytest.approx(10.0 + 0.19 * 10.0)
    after_100 = settled.updated(20.0, tokens=100, now_ms=2)
    assert after_100.value == pytest.approx(10.0 + (1 - 0.9**0.1) * 10.0)

    # More than 3x off is clipped to 3x, not dropped.
    high = settled.updated(1000.0, tokens=1000, now_ms=3)
    assert high.value == pytest.approx(10.0 + 0.1 * (30.0 - 10.0))
    assert high.samples == 31
    low = settled.updated(0.01, tokens=1000, now_ms=3)
    assert low.value == pytest.approx(10.0 + 0.1 * (10.0 / 3 - 10.0))


def test_a_tiny_first_sample_does_not_fix_the_value() -> None:
    # A probe's 2-token "OK" (about 1 char a token) came first, then
    # a 300-token reply at 2.5 chars a token.
    first = averaged(None, 1.016, tokens=2, now_ms=1)
    after = first.updated(2.5, tokens=300, now_ms=2)
    assert after.value == pytest.approx(1.016 + (300 / 302) * (2.5 - 1.016))
    assert after.value == pytest.approx(2.5, abs=0.02)
    assert after.weight == 302

    # While the weight is small, a far-off sample is not clipped.
    unclipped = averaged(None, 1.0, tokens=40, now_ms=1).updated(10.0, tokens=400, now_ms=2)
    assert unclipped.value == pytest.approx(1.0 + (400 / 440) * 9.0)

    # Until about 10k tokens the value is the token-weighted mean.
    mean = averaged(None, 2.0, tokens=3000, now_ms=1).updated(4.0, tokens=1000, now_ms=2)
    assert mean.value == pytest.approx((2.0 * 3000 + 4.0 * 1000) / 4000)


def test_short_responses_give_no_decode_or_chars_per_token_sample() -> None:
    short = _response(text="x" * 60, usage=Usage(input=2000, output=MIN_MEASURED_TOKENS - 1))
    samples = samples_from_response(short)
    assert samples is not None
    assert samples.decode_tokens_per_second is None
    assert samples.output_chars_per_token is None
    assert samples.combined_chars_per_token is None
    assert samples.prefill_tokens_per_second is not None

    enough = _response(text="x" * 100, usage=Usage(input=2000, output=MIN_MEASURED_TOKENS))
    enough_samples = samples_from_response(enough)
    assert enough_samples is not None
    assert enough_samples.decode_tokens_per_second is not None
    assert enough_samples.combined_chars_per_token is not None

    # A long reply with only a few visible tokens samples thinking but not output.
    mixed = _response(
        thinking="t" * 900, text="ok", usage=Usage(input=2000, output=310, reasoning=300)
    )
    mixed_samples = samples_from_response(mixed)
    assert mixed_samples is not None
    assert mixed_samples.thinking_chars_per_token is not None
    assert mixed_samples.output_chars_per_token is None


def test_a_stored_value_without_a_weight_loads_as_untrusted(tmp_path: Path) -> None:
    old = {
        "version": 1,
        "profiles": [
            {
                "provider": "local",
                "endpoint": KEY.endpoint,
                "model": KEY.model,
                "decode_tokens_per_second": {
                    "lt_8k": {"value": 26.5, "samples": 3, "updated_at_ms": 9},
                    "8k_32k": None,
                    "ge_32k": None,
                },
                "prefill_tokens_per_second": None,
                "chars_per_token": {
                    "thinking": None,
                    "output": None,
                    "combined": {"value": 2.5, "samples": 2, "updated_at_ms": 9},
                },
                "limits": {},
                "request_parameters": {},
                "updated_at_ms": 9,
            }
        ],
    }
    store = ModelProfileStore(tmp_path / "model-profiles.json")
    store.path.write_text(json.dumps(old), encoding="utf-8")

    profile = store.load()[KEY]

    assert profile.decode_tokens_per_second["lt_8k"] == AveragedValue(26.5, 3, 9, 0.0)
    assert profile.combined_chars_per_token == AveragedValue(2.5, 2, 9, 0.0)
    assert profile.to_json()["chars_per_token"]["combined"]["weight"] == 0.0


def test_decode_speed_is_bucketed_by_context_size() -> None:
    profile = ModelProfile(key=KEY)
    for prompt, speed_ms in ((1000, 10_000), (10_000, 20_000), (50_000, 40_000)):
        message = _response(
            text="x" * 2000,
            usage=Usage(input=prompt, output=500),
            first_ms=1000,
            total_ms=1000 + speed_ms,
        )
        profile = profile.with_message(message, now_ms=7)

    decode = profile.decode_tokens_per_second
    assert decode["lt_8k"].value == pytest.approx(50.0)
    assert decode["8k_32k"].value == pytest.approx(25.0)
    assert decode["ge_32k"].value == pytest.approx(12.5)
    assert all(value.samples == 1 for value in decode.values())


def test_profiles_are_keyed_by_provider_endpoint_and_model(tmp_path: Path) -> None:
    store = ModelProfileStore.at(TauPaths(home=tmp_path))
    message = _response(text="x" * 2000, usage=Usage(input=2000, output=500))
    other_server = ProfileKey(
        provider="local", endpoint="http://other-host:8080/v1", model="test-model"
    )
    other_model = ProfileKey(provider="local", endpoint=KEY.endpoint, model="other-model")
    other_provider = ProfileKey(provider="lan", endpoint=KEY.endpoint, model="test-model")

    for key in (KEY, KEY, other_server, other_model, other_provider):
        store.record_response(key, message)

    profiles = store.load()
    assert set(profiles) == {KEY, other_server, other_model, other_provider}
    assert profiles[KEY].combined_chars_per_token is not None
    assert profiles[KEY].combined_chars_per_token.samples == 2
    assert profiles[other_server].combined_chars_per_token is not None
    assert profiles[other_server].combined_chars_per_token.samples == 1

    paths = TauPaths(home=tmp_path)
    same_model = read_model_profiles(model="test-model", paths=paths)
    assert {profile.key for profile in same_model} == {KEY, other_server, other_provider}
    only_local = read_model_profiles(provider="local", model="test-model", paths=paths)
    assert {profile.key for profile in only_local} == {KEY, other_server}


def _answer(text: str, output: int) -> list[object]:
    message = AssistantMessage(content=text, usage=Usage(input=300, output=output))
    return [assistant_start(), assistant_done(message=message)]


def test_a_session_updates_the_stored_profile_after_each_response(
    tmp_path: Path,
) -> None:
    provider = FakeProvider([_answer("a" * 160, 40), _answer("b" * 480, 120)])

    async def run() -> list[ModelProfile]:
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                system="You are Tau.",
                storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
                cwd=tmp_path,
                tools=[],
                provider_name="local",
            )
        )
        seen: list[ModelProfile] = []
        for prompt in ("One.", "Two."):
            async for _event in session.prompt(prompt):
                pass
            seen.extend(read_model_profiles())
        return seen

    after_first, after_second = anyio.run(run)

    assert after_first.key.provider == "local"
    assert after_first.key.model == "fake"
    assert after_first.combined_chars_per_token is not None
    assert after_first.combined_chars_per_token.samples == 1
    assert after_first.combined_chars_per_token.value == pytest.approx(4.0)
    assert after_second.key == after_first.key
    assert after_second.combined_chars_per_token is not None
    assert after_second.combined_chars_per_token.samples == 2
    assert after_second.largest_output_tokens == 120
    assert TauPaths().model_profiles_path.exists()


def test_a_failing_profile_write_never_breaks_a_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(ModelProfileStore, "save", broken)
    provider = FakeProvider([_answer("done", 3)])

    async def run() -> CodingSession:
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model="fake",
                system="You are Tau.",
                storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
                cwd=tmp_path,
                tools=[],
            )
        )
        async for _event in session.prompt("Go."):
            pass
        return session

    session = anyio.run(run)
    assert isinstance(session.messages[-1], AssistantMessage)
    assert session.messages[-1].text == "done"


class _PacedProvider:
    """Answers like a slow server: a pause before the first output and while decoding."""

    def __init__(
        self,
        max_tokens: int,
        *,
        honours_max_tokens: bool,
        calls: list[int],
    ) -> None:
        self.max_tokens = max_tokens
        self.honours_max_tokens = honours_max_tokens
        self.calls = calls
        self.closed = False

    async def discover_model_limits(self, model: str) -> RuntimeModelLimits | None:
        return RuntimeModelLimits(context_window=32_768, max_output_tokens=8_192)

    async def aclose(self) -> None:
        self.closed = True

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: object = None,
        session_id: str | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        self.calls.append(self.max_tokens)
        prompt = "".join(getattr(message, "text", "") for message in messages)
        wanted = 300 if "ISO-8601" in prompt else 1
        output = min(wanted, self.max_tokens) if self.honours_max_tokens else wanted
        capped = self.honours_max_tokens and wanted > self.max_tokens
        text = "w" * (output * 3)

        async def events() -> AsyncIterator[AssistantMessageEvent]:
            yield assistant_start()
            await anyio.sleep(0.03)
            yield TextDeltaEvent(
                content_index=0, delta=text, partial=AssistantMessage(content=text)
            )
            await anyio.sleep(0.03)
            final = AssistantMessage(
                content=text, usage=Usage(input=len(prompt) // 4, output=output)
            )
            yield assistant_done(message=final, finish_reason="length" if capped else None)

        return events()


def _probe(tmp_path: Path, *, honours: bool) -> tuple[ModelProfile, list[int]]:
    calls: list[int] = []
    store = ModelProfileStore(tmp_path / "model-profiles.json")

    def provider_for(max_tokens: int) -> _PacedProvider:
        return _PacedProvider(max_tokens, honours_max_tokens=honours, calls=calls)

    profile = anyio.run(lambda: probe_model(KEY, provider_for, store=store))
    assert store.load()[KEY] == profile
    return profile, calls


def test_the_probe_warm_starts_a_profile(tmp_path: Path) -> None:
    profile, calls = _probe(tmp_path, honours=True)

    assert len(calls) == 3
    assert profile.key == KEY
    assert profile.prefill_tokens_per_second is not None
    assert profile.prefill_tokens_per_second.value > 0
    assert profile.decode_tokens_per_second["lt_8k"].value > 0
    assert profile.combined_chars_per_token is not None
    assert profile.combined_chars_per_token.value == pytest.approx(3.0)
    assert profile.output_chars_per_token is not None
    assert profile.context_window == 32_768
    assert profile.max_output_tokens == 8_192
    assert profile.largest_output_tokens == 300
    assert profile.updated_at_ms > 0


def test_the_probe_records_whether_max_tokens_is_honoured(tmp_path: Path) -> None:
    honoured, calls = _probe(tmp_path / "a", honours=True)
    assert CAP_MAX_TOKENS in calls
    assert honoured.max_tokens_honoured is True

    ignored, _ = _probe(tmp_path / "b", honours=False)
    assert ignored.max_tokens_honoured is False

    stopped_early = AssistantMessage(content="one", usage=Usage(output=5))
    assert cap_honoured(stopped_early, CAP_MAX_TOKENS) is None
    assert cap_honoured(None, CAP_MAX_TOKENS) is None


def test_profile_show_prints_json_another_program_can_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "tau-home"
    monkeypatch.setenv("TAU_HOME", str(home))
    store = ModelProfileStore.at(TauPaths())
    message = _response(text="x" * 2000, usage=Usage(input=2000, output=500))
    store.record_response(KEY, message)
    store.record_response(ProfileKey("other", "http://x/v1", "m2"), message)

    result = CliRunner().invoke(app, ["profile", "show", "--model", "test-model", "--json"])

    assert result.exit_code == 0, result.output
    document = json.loads(result.output)
    assert document["version"] == 1
    assert len(document["profiles"]) == 1
    profile = document["profiles"][0]
    assert set(profile) == {
        "provider",
        "endpoint",
        "model",
        "decode_tokens_per_second",
        "prefill_tokens_per_second",
        "chars_per_token",
        "limits",
        "request_parameters",
        "updated_at_ms",
    }
    assert (profile["provider"], profile["endpoint"], profile["model"]) == (
        "local",
        "http://localhost:8080/v1",
        "test-model",
    )
    assert set(profile["decode_tokens_per_second"]) == {"lt_8k", "8k_32k", "ge_32k"}
    assert profile["decode_tokens_per_second"]["lt_8k"]["value"] == pytest.approx(50.0)
    assert profile["decode_tokens_per_second"]["lt_8k"]["samples"] == 1
    assert profile["decode_tokens_per_second"]["ge_32k"] is None
    assert set(profile["chars_per_token"]) == {"thinking", "output", "combined"}
    assert set(profile["limits"]) == {
        "context_window",
        "max_output_tokens",
        "largest_output_tokens",
    }
    assert profile["request_parameters"] == {"max_tokens_honoured": None}
    assert ModelProfile.from_json(profile) == store.load()[KEY]

    text = CliRunner().invoke(app, ["profile", "show"])
    assert text.exit_code == 0
    assert "local/test-model @ http://localhost:8080/v1" in text.output


def test_the_measuring_code_imports_nothing_from_tau_coding() -> None:
    import tau_ai.model_profile as measuring

    source = Path(measuring.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert imported, "the AST walk found no imports at all"
    assert not [name for name in imported if name.split(".")[0] == "tau_coding"]
    assert not [name for name in imported if name in {"os", "pathlib", "io", "shutil"}]

    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, tau_ai.model_profile; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] == 'tau_coding'))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert loaded.stdout.strip() == "[]"


def test_context_estimates_wait_for_enough_evidence() -> None:
    ratio = AveragedValue(2.5, samples=3, updated_at_ms=1, weight=MIN_TRUSTED_RATIO_WEIGHT - 1)
    assert context_chars_per_token(None) == CHARS_PER_TOKEN
    assert context_chars_per_token(ModelProfile(key=KEY)) == CHARS_PER_TOKEN
    assert (
        context_chars_per_token(ModelProfile(key=KEY, combined_chars_per_token=ratio))
        == CHARS_PER_TOKEN
    )
    settled = AveragedValue(2.5, samples=9, updated_at_ms=1, weight=MIN_TRUSTED_RATIO_WEIGHT)
    assert context_chars_per_token(ModelProfile(key=KEY, combined_chars_per_token=settled)) == 2.5


def test_a_session_estimates_context_at_the_measured_ratio(tmp_path: Path) -> None:
    async def estimate() -> int:
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="fake",
                system="s" * 1000,
                storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
                cwd=tmp_path,
                tools=[],
                provider_name="local",
            )
        )
        return session.context_usage.system_tokens

    assert anyio.run(estimate) == 250

    ModelProfileStore.at().put(
        ModelProfile(
            key=ProfileKey(provider="local", endpoint="", model="fake"),
            combined_chars_per_token=AveragedValue(
                2.0, samples=9, updated_at_ms=1, weight=MIN_TRUSTED_RATIO_WEIGHT
            ),
        )
    )
    assert anyio.run(estimate) == 500


def test_a_failed_probe_stores_nothing_and_reports_the_error(tmp_path: Path) -> None:
    failed = [assistant_start(), assistant_error("connection refused")]
    store = ModelProfileStore(tmp_path / "model-profiles.json")

    with pytest.raises(ProbeError, match="connection refused"):
        anyio.run(lambda: probe_model(KEY, lambda _cap: FakeProvider([failed]), store=store))

    assert store.load() == {}


def test_the_probe_command_exits_non_zero_when_the_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fails(*_args: object, **_kwargs: object) -> ModelProfile:
        raise ProbeError("connection refused")

    monkeypatch.setattr("tau_coding.cli.run_probe", fails)
    result = CliRunner().invoke(app, ["probe", "--model=m"])
    assert result.exit_code == 1
    assert "connection refused" in result.output


@pytest.mark.parametrize(
    "words",
    [["probe"], ["probe", "--model", "m"], ["probe", "--provider=local", "-m", "m"]],
)
def test_probe_with_only_options_is_the_command(
    words: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str | None] = []

    async def fake(provider: str | None, model: str | None, **_: object) -> ModelProfile:
        seen.append(model)
        return ModelProfile(key=KEY)

    monkeypatch.setattr("tau_coding.cli.run_probe", fake)
    assert CliRunner().invoke(app, words).exit_code == 0
    assert seen == ["m" if len(words) > 1 else None]


@pytest.mark.parametrize("words", [["probe", "the", "auth", "bug"], ["profile", "this", "code"]])
def test_a_prompt_starting_with_those_words_is_not_the_command(
    words: list[str],
) -> None:
    from tau_coding import cli

    assert not cli._probe_requested(words)
    assert words[:2] != ["profile", "show"]


def test_profile_show_rejects_an_unknown_option() -> None:
    result = CliRunner().invoke(app, ["profile", "show", "--nope"])
    assert result.exit_code != 0


def test_the_cached_profile_follows_a_model_change(tmp_path: Path) -> None:
    store = ModelProfileStore.at()
    ratio = AveragedValue(2.0, samples=9, updated_at_ms=1, weight=MIN_TRUSTED_RATIO_WEIGHT)
    store.put(
        ModelProfile(
            key=ProfileKey(provider="local", endpoint="", model="measured"),
            combined_chars_per_token=ratio,
        )
    )

    async def ratios() -> tuple[float, float]:
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=FakeProvider([]),
                model="unmeasured",
                system="You are Tau.",
                storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
                cwd=tmp_path,
                tools=[],
                provider_name="local",
            )
        )
        before = session._context_chars_per_token()
        session._harness.config.model = "measured"
        return before, session._context_chars_per_token()

    assert anyio.run(ratios) == (CHARS_PER_TOKEN, 2.0)
