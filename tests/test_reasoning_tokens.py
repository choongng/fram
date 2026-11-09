"""Thinking is budgeted by amount; the stall timer is the only clock.

A budget in seconds would give a slow model less thinking than a fast one, so
time never cuts reasoning. Output time only backstops real faults such as a stuck
server.
"""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import httpx
import pytest

from conftest import isolate_home
from tau_agent import AssistantMessage
from tau_agent.messages import ThinkingContent
from tau_ai import FakeProvider
from tau_ai.env import OpenAICompatibleConfig
from tau_ai.openai_compatible import OpenAICompatibleProvider
from tau_ai.stall import STREAM_STALL_SECONDS_ENV
from tau_coding import CodingSession, OpenAICompatibleProviderConfig, ProviderSettings
from tau_coding import session as coding_session_module
from tau_coding.guards import (
    BUDGET_CUT_NUDGE,
    BUDGET_CUT_WRITE_NUDGE,
    OUTPUT_LIMIT_NUDGE,
)
from tau_coding.reasoning_guard import (
    TURN_REASONING_TOKENS_ENV,
)
from test_session_output_limit import _answers, _session, _streamed_reasoning

#: Distinct sentences, so the reasoning-loop guard never fires on them.
REASONING = " ".join(f"Byte {n} holds bits {n * 7} to {n * 7 + 6}." for n in range(4000))


def _reasons_long() -> list[object]:
    never_reached = AssistantMessage(content="The reasoning ran to its end.")
    return _streamed_reasoning(REASONING, then=never_reached, chunk=40)


async def _thinking_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    main: FakeProvider,
    *,
    thinking_format: str | None = None,
    level: str = "medium",
) -> CodingSession:
    """A session on `main` at thinking `level`, its model of `thinking_format`."""
    isolate_home(monkeypatch, tmp_path)
    provider_config = OpenAICompatibleProviderConfig(
        name="local",
        models=("fake",),
        default_model="fake",
        compat={"thinkingFormat": thinking_format} if thinking_format else {},
        thinking_levels=("off", "low", "medium", "high"),
        thinking_models=("fake",),
        thinking_default=level,  # type: ignore[arg-type]
        thinking_parameter="reasoning_effort",
    )
    monkeypatch.setattr(coding_session_module, "_create_runtime_provider", lambda *a, **k: main)
    return await _session(
        tmp_path,
        main,
        provider_name="local",
        provider_settings=ProviderSettings(providers=(provider_config,)),
        runtime_provider_config=provider_config,
        thinking_level=level,
    )


def _fast_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clock that runs 1000 s per reading, for any code that still reads one."""
    ticks = iter(range(0, 10**9, 1000))
    monkeypatch.setattr(
        coding_session_module, "_reasoning_clock", lambda: next(ticks), raising=False
    )


def _cut(session: CodingSession) -> AssistantMessage:
    [cut] = [
        message
        for message in session.messages
        if isinstance(message, AssistantMessage) and message.diagnostics
    ]
    return cut


def _thought(message: AssistantMessage) -> int:
    return sum(
        len(block.thinking) for block in message.content if isinstance(block, ThinkingContent)
    )


@pytest.fixture
def anyio_backend() -> str:
    """One backend, unparametrized, so each test's node id is its plain name."""
    return "asyncio"


@pytest.mark.anyio
async def test_reasoning_is_cut_by_tokens_not_seconds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # With the clock racing, no budget means no cut.
    monkeypatch.delenv(TURN_REASONING_TOKENS_ENV, raising=False)
    _fast_clock(monkeypatch)
    main = FakeProvider(
        [_streamed_reasoning(REASONING[:4000], then=AssistantMessage(content="Done."))]
    )
    session = await _thinking_session(monkeypatch, tmp_path, main)
    async for _event in session.prompt("Encode it."):
        pass
    assert session.messages[-1].text == "Done."
    assert len(main.calls) == 1

    # A token budget cuts at that many tokens, however fast: at the model's
    # thinking ratio, 2.5 characters a token while it is unmeasured.
    monkeypatch.setenv(TURN_REASONING_TOKENS_ENV, "300")
    main = FakeProvider([_reasons_long(), _answers("Acted.")])
    session = await _thinking_session(monkeypatch, tmp_path / "b", main)
    async for _event in session.prompt("Encode it."):
        pass
    cut = _cut(session)
    assert cut.stop_reason == "length"
    [diagnostic] = cut.diagnostics or []
    assert diagnostic.type == "reasoning_budget_cut"
    assert diagnostic.details is not None and diagnostic.details["tokens"] == 300
    assert diagnostic.details["chars"] == 750
    assert 750 < _thought(cut) <= 750 + 40
    assert session.messages[-1].text == "Acted."


def _sse(*deltas: dict[str, object], finish: str | None = "stop") -> list[bytes]:
    chunks = [
        ("data: " + json.dumps({"choices": [{"delta": delta}]}) + "\n\n").encode()
        for delta in deltas
    ]
    if finish is not None:
        done = {"choices": [{"delta": {}, "finish_reason": finish}]}
        chunks += [("data: " + json.dumps(done) + "\n\n").encode(), b"data: [DONE]\n\n"]
    return chunks


class _Paced(httpx.AsyncByteStream):
    """A response body that sends `chunks` with `gap` seconds before each, then
    waits `then` seconds before closing (a server gone quiet)."""

    def __init__(self, chunks: list[bytes], *, gap: float = 0.0, then: float = 0.0) -> None:
        self.chunks, self.gap, self.then = chunks, gap, then

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            await anyio.sleep(self.gap)
            yield chunk
        await anyio.sleep(self.then)


async def _stream(bodies: list[_Paced]) -> tuple[list[object], int]:
    """Stream one response from a fake server answering each request with the
    next of `bodies`; returns the events and the requests made."""
    requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        body = bodies[min(requests, len(bodies) - 1)]
        requests += 1
        return httpx.Response(200, stream=body, headers={"content-type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="test",
                base_url="https://example.test/v1",
                max_retries=1,
                max_retry_delay_seconds=0,
            ),
            client=client,
        )
        events = [
            event
            async for event in provider.stream_response(
                model="fake", system="", messages=[], tools=[]
            )
        ]
    return events, requests


@pytest.mark.anyio
async def test_a_slow_steady_stream_is_never_cut_by_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The stall timer is 0.2 s; the stream takes over 1 s, a token every 0.05 s.
    monkeypatch.setenv(STREAM_STALL_SECONDS_ENV, "0.2")
    deltas = [{"reasoning_content": f"step {n}. "} for n in range(20)]
    body = _Paced(_sse(*deltas, {"content": "Done."}), gap=0.05)
    with anyio.fail_after(10):
        events, requests = await _stream([body])
    done = events[-1]
    assert getattr(done, "type", None) == "done", done
    assert done.message.text == "Done."  # type: ignore[attr-defined]
    assert "step 19." in done.message.thinking_text  # type: ignore[attr-defined]
    assert requests == 1

    # In the session too: no budget, no cut, however long it takes.
    monkeypatch.delenv(TURN_REASONING_TOKENS_ENV, raising=False)
    _fast_clock(monkeypatch)
    main = FakeProvider(
        [_streamed_reasoning(REASONING, then=AssistantMessage(content="Done."), chunk=40)]
    )
    session = await _thinking_session(monkeypatch, tmp_path, main)
    async for _event in session.prompt("Encode it."):
        pass
    assert session.messages[-1].text == "Done."
    assert len(main.calls) == 1


@pytest.mark.anyio
async def test_a_silent_stream_is_ended_by_the_stall_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(STREAM_STALL_SECONDS_ENV, "0.2")
    # Content arrived, then the server went quiet (an SSE keep-alive is not an
    # event): the request ends as an error naming the silence, not as a cut.
    quiet = _Paced([*_sse({"reasoning_content": "Thinking."}, finish=None), b": ping\n\n"], then=30)
    with anyio.fail_after(10):
        events, requests = await _stream([quiet])
    error = events[-1]
    assert getattr(error, "type", None) == "error", error
    message = error.error  # type: ignore[attr-defined]
    assert message.stop_reason == "error"
    assert message.error_message == "the server sent nothing for 0.2 s"
    assert requests == 1

    # Nothing arrived at all: the transport retry sends it again.
    silent = _Paced([], then=30)
    answers = _Paced(_sse({"content": "Done."}))
    with anyio.fail_after(10):
        events, requests = await _stream([silent, answers])
    assert getattr(events[-1], "type", None) == "done"
    assert events[-1].message.text == "Done."  # type: ignore[attr-defined]
    assert requests == 2


@pytest.mark.anyio
async def test_the_cut_nudge_names_thinking_not_the_output_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(TURN_REASONING_TOKENS_ENV, "300")
    main = FakeProvider([_reasons_long(), _reasons_long(), _answers("Wrote it.")])
    session = await _thinking_session(monkeypatch, tmp_path, main)
    async for _event in session.prompt("Encode it."):
        pass

    sent = [call[2][-1].text for call in main.calls]
    assert sent == ["Encode it.", BUDGET_CUT_NUDGE, BUDGET_CUT_WRITE_NUDGE]
    for nudge in (BUDGET_CUT_NUDGE, BUDGET_CUT_WRITE_NUDGE):
        words = nudge.lower()
        assert "thinking budget" in words, nudge
        assert "output limit" not in words and "minutes" not in words, nudge
        assert "tool call" in words, nudge
    # The budget is in characters, so the nudge names no unit.
    assert "token" not in BUDGET_CUT_NUDGE.lower()
    # A real output-limit cut is still named as one.
    assert "output limit" in OUTPUT_LIMIT_NUDGE


@pytest.mark.anyio
async def test_an_unbounded_thinking_format_gets_a_token_budget_from_its_level(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(TURN_REASONING_TOKENS_ENV, raising=False)

    async def cut_chars(where: Path, thinking_format: str | None, level: str) -> int | None:
        main = FakeProvider([_reasons_long(), _answers("Acted.")])
        session = await _thinking_session(
            monkeypatch, where, main, thinking_format=thinking_format, level=level
        )
        async for _event in session.prompt("Encode it."):
            pass
        cuts = [m for m in session.messages if isinstance(m, AssistantMessage) and m.diagnostics]
        if not cuts:
            return None
        [diagnostic] = cuts[0].diagnostics or []
        assert diagnostic.type == "reasoning_budget_cut"
        chars = (diagnostic.details or {})["chars"]
        assert isinstance(chars, int)
        return chars

    # qwen-chat-template sends only enable_thinking: the level becomes a budget,
    # in characters.
    assert await cut_chars(tmp_path / "medium", "qwen-chat-template", "medium") == 32768
    assert await cut_chars(tmp_path / "low", "qwen-chat-template", "low") == 8192
    assert await cut_chars(tmp_path / "qwen", "qwen", "low") == 8192
    # A format that sends the level to the server is left to the server.
    assert await cut_chars(tmp_path / "openai", None, "low") is None
    # The environment's budget wins over the level's: 500 tokens at 2.5 a token.
    monkeypatch.setenv(TURN_REASONING_TOKENS_ENV, "500")
    assert await cut_chars(tmp_path / "env", "qwen-chat-template", "medium") == 1250
