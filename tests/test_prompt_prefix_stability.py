"""Each request a session sends starts with the whole of the one before it.

A server's prompt cache (llama.cpp's slot, OpenAI's and Anthropic's prefix
caches) only helps while the next request starts with the bytes of the last
one. A change of request parameters for a single request, such as thinking off
for one follow-up, can make a chat template rewrite the system turn, and the
server then reads the whole prompt again. This test runs a session through tool
calls with thinking, a cut at the thinking budget and its follow-up, and a
second prompt, builds the chat payload of every request, and checks that every
one extends the one before it: the same parameters, system turn and tools, and
the earlier messages unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from conftest import isolate_home
from pi_event_helpers import assistant_done, assistant_start
from tau_agent import AssistantMessage
from tau_agent.messages import TextContent, ThinkingContent, ToolCall
from tau_ai import FakeProvider
from tau_ai.openai_compatible import _build_chat_payload
from tau_coding import OpenAICompatibleProviderConfig, ProviderSettings
from tau_coding import session as coding_session_module
from tau_coding.guards import BUDGET_CUT_NUDGE
from tau_coding.reasoning_guard import TURN_REASONING_TOKENS_ENV
from test_session_output_limit import _session, _streamed_reasoning

COMPAT = {"thinkingFormat": "qwen-chat-template"}


def _calls(n: int, thought: str) -> list[object]:
    call = ToolCall(id=f"call-{n}", name="fetch", arguments={"n": n})
    message = AssistantMessage(content=[ThinkingContent(thinking=thought), call])
    return [assistant_start(), assistant_done(message=message)]


def _answers(text: str) -> list[object]:
    message = AssistantMessage(
        content=[ThinkingContent(thinking="Done, then."), TextContent(text=text)]
    )
    return [assistant_start(), assistant_done(message=message)]


class _Routed:
    """The runtime provider a session builds: `main`, noting the thinking level
    it was built for on each request, as a real provider sends it."""

    def __init__(self, main: FakeProvider, level: str | None, levels: list[str | None]) -> None:
        self._main = main
        self._level = level
        self._levels = levels

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        self._levels.append(self._level)
        return self._main.stream_response(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._main, name)


def _payload(
    level: str | None, system: str, messages: list[Any], tools: list[Any]
) -> dict[str, Any]:
    return _build_chat_payload(
        model="local-model",
        system=system,
        messages=messages,
        tools=tools,
        reasoning_effort=level,
        thinking_format="qwen-chat-template",
        compat=COMPAT,
        max_tokens=16384,
        cache_prompt=True,
        id_slot=0,
    )


@pytest.mark.anyio
async def test_every_request_extends_the_one_before_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolate_home(monkeypatch, tmp_path)
    monkeypatch.setenv(TURN_REASONING_TOKENS_ENV, "5")
    long = " ".join(f"Byte {n} holds bits {n * 7} to {n * 7 + 6}." for n in range(40))
    main = FakeProvider(
        [
            _calls(1, "Read the reference first."),
            _streamed_reasoning(long, then=AssistantMessage(content="never reached")),
            _calls(2, ""),
            _answers("Encoded."),
            _calls(3, "Now the decoder."),
            _answers("Decoded."),
        ]
    )
    provider_config = OpenAICompatibleProviderConfig(
        name="local",
        models=("local-model",),
        default_model="local-model",
        thinking_levels=("off", "low", "medium"),
        thinking_models=("local-model",),
        thinking_default="medium",
        thinking_parameter="reasoning_effort",
    )
    levels: list[str | None] = []
    monkeypatch.setattr(
        coding_session_module,
        "_create_runtime_provider",
        lambda *a, thinking_level=None, **k: _Routed(main, thinking_level, levels),
    )
    session = await _session(
        tmp_path,
        main,
        provider_name="local",
        provider_settings=ProviderSettings(providers=(provider_config,)),
        runtime_provider_config=provider_config,
    )

    for prompt in ("Encode it.", "Now decode it."):
        async for _event in session.prompt(prompt):
            pass

    # Every request went through a provider the session built for it.
    assert len(levels) == len(main.calls) == 6
    payloads = [
        _payload(level, system, messages, tools)
        for level, (_model, system, messages, tools) in zip(levels, main.calls, strict=True)
    ]
    # The third request is the follow-up to the cut at the thinking budget.
    assert payloads[2]["messages"][-1] == {"role": "user", "content": BUDGET_CUT_NUDGE}

    for before, after in zip(payloads, payloads[1:], strict=False):
        assert {k: v for k, v in after.items() if k != "messages"} == {
            k: v for k, v in before.items() if k != "messages"
        }
        sent = before["messages"]
        assert after["messages"][: len(sent)] == sent
