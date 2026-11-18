"""A budget cut's follow-up is sent with its reply begun past the thinking.

A model asked in words to stop thinking after a budget cut may think each
follow-up to its budget again and never act. Sending the follow-up with thinking
off would stop it, but chat templates that mark thinking in the system turn
would then change the whole prompt, so the server would read it all again. The
guard instead ends the request with an assistant turn of
`CLOSED_THINKING_PREFILL`, which the template renders with an empty, closed
thinking block, where the provider says it continues such a turn.
"""

from __future__ import annotations

from typing import Any

from tau_agent import AssistantMessage, UserMessage
from tau_ai import openai_compatible
from tau_ai.openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    _build_chat_payload,
)
from tau_coding import reasoning_guard
from tau_coding.guards import BUDGET_CUT_NUDGE, BUDGET_CUT_WRITE_NUDGE
from tau_coding.reasoning_guard import (
    CLOSED_THINKING_PREFILL,
    ReasoningGuardSetting,
    RequestGuard,
    is_budget_cut_follow_up,
)
from tau_coding.request_gate import ConcurrencyLimitedProvider


class Recording:
    """A provider that keeps the messages of each request it is sent."""

    def __init__(self, *, prefill: bool | None) -> None:
        if prefill is not None:
            self.takes_assistant_prefill = lambda model: prefill
        self.sent: list[list[Any]] = []

    def stream_response(self, **kwargs: Any) -> object:
        self.sent.append(kwargs["messages"])
        return iter(())


def _send(provider: object, text: str, *, enabled: bool = True) -> list[Any]:
    history = [UserMessage(content="Write codec.py."), AssistantMessage(content="")]
    guard = RequestGuard(
        provider,  # type: ignore[arg-type]
        setting=lambda: ReasoningGuardSetting(enabled=enabled),
    )
    guard.stream_response(
        model="m", system="s", messages=[*history, UserMessage(content=text)], tools=[]
    )
    assert isinstance(provider, Recording)
    sent = provider.sent[-1]
    assert sent[: len(history) + 1] == [*history, UserMessage(content=text)]
    return sent[len(history) + 1 :]


def test_a_follow_up_is_sent_with_its_reply_begun() -> None:
    for nudge in (BUDGET_CUT_NUDGE, BUDGET_CUT_WRITE_NUDGE):
        added = _send(Recording(prefill=True), nudge)
        assert len(added) == 1
        assert isinstance(added[0], AssistantMessage)
        assert added[0].text == CLOSED_THINKING_PREFILL


def test_any_other_request_is_sent_as_it_is() -> None:
    assert _send(Recording(prefill=True), "Write the decoder.") == []


def test_a_provider_that_takes_no_prefill_gets_the_follow_up_as_it_is() -> None:
    assert _send(Recording(prefill=False), BUDGET_CUT_NUDGE) == []
    assert _send(Recording(prefill=None), BUDGET_CUT_NUDGE) == []


def test_a_follow_up_is_sent_as_it_is_while_the_guard_is_off() -> None:
    assert _send(Recording(prefill=True), BUDGET_CUT_NUDGE, enabled=False) == []


def test_the_follow_ups_budget_is_still_the_reply_share(monkeypatch: Any) -> None:
    seen: list[list[Any]] = []
    real = reasoning_guard.request_reasoning_chars

    def spy(messages: list[Any], *args: Any, **kwargs: Any) -> int | None:
        seen.append(messages)
        return real(messages, *args, **kwargs)

    monkeypatch.setattr(reasoning_guard, "request_reasoning_chars", spy)
    assert len(_send(Recording(prefill=True), BUDGET_CUT_NUDGE)) == 1
    assert seen and is_budget_cut_follow_up(seen[-1])


def test_the_capability_reaches_the_guard_through_the_request_gate() -> None:
    gated = ConcurrencyLimitedProvider(
        Recording(prefill=True),  # type: ignore[arg-type]
        provider_name="local",
        limit=1,
    )
    assert gated.takes_assistant_prefill("m") is True  # forwarded to the provider it wraps


def test_the_compat_value_turns_the_capability_on() -> None:
    def provider(**compat: Any) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            OpenAICompatibleConfig(api_key="k", base_url="http://localhost:8080/v1", compat=compat)
        )

    assert provider(supportsAssistantPrefill=True).takes_assistant_prefill("m")
    assert not provider().takes_assistant_prefill("m")
    assert not provider(supportsAssistantPrefill="yes").takes_assistant_prefill("m")


def test_a_request_routed_to_responses_takes_no_prefill() -> None:
    def provider(**config: Any) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key="k",
                base_url="http://localhost:8080/v1",
                compat={"supportsAssistantPrefill": True},
                **config,
            )
        )

    assert not provider(api="openai-responses").takes_assistant_prefill("m")
    responses_model = "gpt-5.5"
    assert openai_compatible._use_responses_api(responses_model)
    assert not provider().takes_assistant_prefill(responses_model)
    assert provider(infer_api_from_model=False).takes_assistant_prefill(responses_model)


def test_the_prefill_is_the_payloads_last_message_and_leaves_the_rest_alone() -> None:
    history = [UserMessage(content="Write codec.py."), UserMessage(content=BUDGET_CUT_NUDGE)]
    kwargs: dict[str, Any] = {
        "model": "m",
        "system": "s",
        "tools": [],
        "reasoning_effort": "medium",
        "thinking_format": "qwen-chat-template",
    }
    plain = _build_chat_payload(messages=history, **kwargs)
    begun = _build_chat_payload(
        messages=[*history, AssistantMessage(content=CLOSED_THINKING_PREFILL)], **kwargs
    )
    assert begun["messages"][:-1] == plain["messages"]
    assert begun["messages"][-1] == {"role": "assistant", "content": CLOSED_THINKING_PREFILL}
    assert {k: v for k, v in begun.items() if k != "messages"} == {
        k: v for k, v in plain.items() if k != "messages"
    }
