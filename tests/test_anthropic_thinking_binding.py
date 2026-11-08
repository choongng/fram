"""Replayed Anthropic thinking blocks stay scoped to the live continuation.

A thinking-block signature is bound to the request prefix that produced it.
Compaction, injected context, resume or branch change that prefix, and every
later request that replays the block answers with a 400 naming the signature
or its block binding, and the conversation cannot continue. Tau sends thinking
only for the tool loop the trailing tool results continue (every step since the
last real user message, so each request's prefix extends the previous one), and
recovers once, without thinking, if the provider still rejects a replay.
"""

import json
from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from anthropic_transport_helpers import OK_SSE as _OK_SSE
from anthropic_transport_helpers import SIGNATURE_400 as _SIGNATURE_400
from anthropic_transport_helpers import SignatureBindingTransport as _SignatureBindingTransport
from anthropic_transport_helpers import blocks as _blocks
from anthropic_transport_helpers import canonical as _canonical
from tau_agent import UserMessage
from tau_agent.messages import (
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ToolResultMessage,
)
from tau_agent.tools import ToolCall
from tau_ai import (
    AnthropicConfig,
    AnthropicProvider,
    AssistantDoneEvent,
    AssistantErrorEvent,
)
from tau_ai.anthropic import _build_messages_payload, _prune_replayed_thinking
from tau_ai.env import CACHE_RETENTION_NONE

_MODEL = "claude-test"


def _assistant_thinking_turn(
    *,
    tool_call_id: str | None = None,
    text: str | None = "done",
    signature: str = "sig-1",
) -> AssistantMessage:
    content: list[Any] = [
        ThinkingContent(thinking="reasoning", thinking_signature=signature),
    ]
    if text is not None:
        content.append(TextContent(text=text))
    if tool_call_id is not None:
        content.append(
            ToolCall(id=tool_call_id, name="read", arguments={"path": "a.py"}),
        )
    return AssistantMessage(
        api="anthropic-messages",
        provider="anthropic",
        model=_MODEL,
        content=content,
    )


def _has_thinking(payload_message: Mapping[str, Any]) -> bool:
    return any(
        block.get("type") in {"thinking", "redacted_thinking"} for block in _blocks(payload_message)
    )


def _build(messages: list[Any]) -> dict[str, Any]:
    return _build_messages_payload(
        model=_MODEL,
        system="You are Tau.",
        messages=messages,
        tools=[],
        cache_retention=CACHE_RETENTION_NONE,
    )


def test_completed_earlier_thinking_turn_is_not_replayed() -> None:
    messages = [
        UserMessage(content="start"),
        _assistant_thinking_turn(),
        UserMessage(content="next step"),
    ]

    payload = _build(messages)

    payload_messages = payload["messages"]
    assert [m["role"] for m in payload_messages] == ["user", "assistant", "user"]
    assert [block["type"] for block in _blocks(payload_messages[1])] == ["text"]


def test_continuation_thinking_block_is_replayed_with_signature() -> None:
    messages = [
        UserMessage(content="read a.py"),
        _assistant_thinking_turn(tool_call_id="toolu_1", text=None),
        ToolResultMessage(tool_call_id="toolu_1", tool_name="read", content="body"),
    ]

    payload = _build(messages)

    assert _blocks(payload["messages"][1]) == [
        {"type": "thinking", "thinking": "reasoning", "signature": "sig-1"},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "read",
            "input": {"path": "a.py"},
        },
    ]


def test_only_the_trailing_continuation_keeps_its_thinking() -> None:
    messages = [
        UserMessage(content="start"),
        _assistant_thinking_turn(),
        UserMessage(content="keep going"),
        _assistant_thinking_turn(tool_call_id="toolu_2", text=None),
        ToolResultMessage(tool_call_id="toolu_2", tool_name="read", content="body"),
    ]

    payload = _build(messages)

    payload_messages = payload["messages"]
    assert [m["role"] for m in payload_messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert not _has_thinking(payload_messages[1])
    assert _has_thinking(payload_messages[3])


def test_every_step_of_the_live_tool_loop_keeps_its_thinking() -> None:
    """Pruning earlier loop steps would change the prefix on every tool call.

    Each step's request is a strict extension of the previous one, so the
    prompt cache keeps hitting and each replayed block goes out under the
    prefix it was produced with.
    """
    first_step = [
        UserMessage(content="start"),
        _assistant_thinking_turn(tool_call_id="toolu_31", text=None, signature="sig-a"),
        ToolResultMessage(tool_call_id="toolu_31", tool_name="read", content="body one"),
    ]
    second_step = [
        *first_step,
        _assistant_thinking_turn(tool_call_id="toolu_32", text=None, signature="sig-b"),
        ToolResultMessage(tool_call_id="toolu_32", tool_name="read", content="body two"),
    ]

    first = _build(first_step)["messages"]
    second = _build(second_step)["messages"]

    assert second[: len(first)] == first
    assert [block["signature"] for m in second for block in _thinking_blocks(m)] == [
        "sig-a",
        "sig-b",
    ]


def _thinking_history() -> list[Any]:
    return [
        UserMessage(content="start"),
        _assistant_thinking_turn(tool_call_id="toolu_9", text=None),
        ToolResultMessage(tool_call_id="toolu_9", tool_name="read", content="body"),
    ]


def _provider(
    responses: list[httpx.Response],
    captured: list[dict[str, Any]],
) -> AnthropicProvider:
    """Provider whose transport replays `responses`, then streams ok text.

    Every request payload is appended to `captured`.
    """
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode()))
        if queue:
            return queue.pop(0)
        return httpx.Response(
            200,
            text=_OK_SSE,
            headers={"content-type": "text/event-stream"},
        )

    return AnthropicProvider(
        AnthropicConfig(
            api_key="test-key",
            base_url="https://example.test/v1",
            thinking_budget_tokens=1024,
        ),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.anyio
async def test_thinking_signature_400_retries_once_without_thinking() -> None:
    captured: list[dict[str, Any]] = []
    provider = _provider(
        [
            httpx.Response(
                400,
                text=_SIGNATURE_400,
                headers={"content-type": "application/json"},
            ),
        ],
        captured,
    )

    events = [
        event
        async for event in provider.stream_response(
            model=_MODEL,
            system="You are Tau.",
            messages=_thinking_history(),
            tools=[],
        )
    ]

    assert isinstance(events[-1], AssistantDoneEvent)
    assert events[-1].message.text == "ok"
    assert len(captured) == 2
    # The first request replayed thinking; the retry after the 400 sent none,
    # in history or in config.
    assert _has_thinking(captured[0]["messages"][1])
    assert captured[0]["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert not any(_has_thinking(m) for m in captured[1]["messages"])
    assert "thinking" not in captured[1]

    # The next user message ends the recovery: thinking is on again, and the
    # older signatures are pruned rather than replayed.
    next_messages = _thinking_history() + [
        AssistantMessage(
            api="anthropic-messages",
            provider="anthropic",
            model=_MODEL,
            content=[
                ThinkingContent(thinking="more", thinking_signature="sig-2"),
                TextContent(text="ok"),
            ],
        ),
        UserMessage(content="and now?"),
    ]
    events2 = [
        event
        async for event in provider.stream_response(
            model=_MODEL,
            system="You are Tau.",
            messages=next_messages,
            tools=[],
        )
    ]

    assert isinstance(events2[-1], AssistantDoneEvent)
    assert len(captured) == 3
    assert "thinking" in captured[2]
    assert not any(_has_thinking(m) for m in captured[2]["messages"])


@pytest.mark.anyio
async def test_unrelated_400_stays_terminal_without_retry() -> None:
    captured: list[dict[str, Any]] = []
    provider = _provider(
        [
            httpx.Response(
                400,
                text=json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "messages: too long",
                        },
                    }
                ),
                headers={"content-type": "application/json"},
            ),
        ],
        captured,
    )

    events = [
        event
        async for event in provider.stream_response(
            model=_MODEL,
            system="You are Tau.",
            messages=_thinking_history(),
            tools=[],
        )
    ]

    assert len(captured) == 1
    assert isinstance(events[-1], AssistantErrorEvent)


# ---------------------------------------------------------------------------
# Per-rule coverage of the criterion: a tau conversation on an Anthropic
# thinking model cannot be stopped by a thinking-block signature the provider
# no longer recognises. Past thinking is not replayed at all: only what the
# current tool-use continuation needs is sent. And if the provider still
# answers that 400, tau recovers the conversation once rather than ending the
# run.
# ---------------------------------------------------------------------------


def _thinking_blocks(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        block
        for block in _blocks(message)
        if block.get("type") in {"thinking", "redacted_thinking"}
    ]


def _has_tool_result(message: Mapping[str, Any]) -> bool:
    return any(block.get("type") == "tool_result" for block in _blocks(message))


def test_completed_turn_loses_only_its_thinking_not_the_turn() -> None:
    """A finished assistant turn stays in history; its thinking does not."""
    messages = [
        UserMessage(content="start"),
        _assistant_thinking_turn(text="first answer"),
        UserMessage(content="and now"),
    ]

    payload = _build(messages)
    payload_messages = payload["messages"]

    # The turn itself is still sent — stripping must not drop the message.
    assert [m["role"] for m in payload_messages] == ["user", "assistant", "user"]
    assert len(_blocks(payload_messages[1])) > 0
    # None of it is thinking: a completed turn's thinking is never replayed.
    assert _thinking_blocks(payload_messages[1]) == []
    assert not any(_has_thinking(m) for m in payload_messages)


def test_active_continuation_replays_thinking_with_its_signature() -> None:
    """The tool-use continuation the results answer keeps signature-bearing thinking."""
    messages = [
        UserMessage(content="read a.py"),
        _assistant_thinking_turn(tool_call_id="toolu_11", text=None),
        ToolResultMessage(tool_call_id="toolu_11", tool_name="read", content="file body"),
    ]

    payload = _build(messages)
    assistant = payload["messages"][1]
    thinking = _thinking_blocks(assistant)

    assert len(thinking) == 1
    for block in thinking:
        assert block.get("signature") == "sig-1"
    # The thinking still precedes the tool_use it produced.
    assert [block["type"] for block in _blocks(assistant)] == [
        "thinking",
        "tool_use",
    ]


def test_only_the_trailing_continuation_keeps_thinking_earlier_ones_are_stripped() -> None:
    """Every assistant turn before the live continuation is sent thinking-free."""
    messages = [
        UserMessage(content="start"),
        _assistant_thinking_turn(text="first answer"),
        UserMessage(content="keep going"),
        _assistant_thinking_turn(tool_call_id="toolu_21", text=None),
        ToolResultMessage(tool_call_id="toolu_21", tool_name="read", content="body one"),
        UserMessage(content="continue"),
        _assistant_thinking_turn(tool_call_id="toolu_22", text=None),
        ToolResultMessage(tool_call_id="toolu_22", tool_name="read", content="body two"),
    ]

    payload = _build(messages)
    payload_messages = payload["messages"]
    assistant_indexes = [
        index for index, message in enumerate(payload_messages) if message["role"] == "assistant"
    ]
    last_assistant_idx = assistant_indexes[-1]
    assert last_assistant_idx < len(payload_messages) - 1

    assert not _has_thinking(payload_messages[0])
    for index in assistant_indexes[:-1]:
        assert index < last_assistant_idx
        assert not _has_thinking(payload_messages[index])
    assert _has_thinking(payload_messages[last_assistant_idx])


def test_thinking_only_completed_turn_is_dropped_not_sent_empty() -> None:
    """Stripping can empty a thinking-only turn; the inert turn is omitted.

    Anthropic rejects empty assistant content, so a turn that held nothing but
    replayed thinking must disappear from the payload rather than go out as
    `{"role": "assistant", "content": []}`.
    """
    messages = [
        UserMessage(content="start"),
        AssistantMessage(
            api="anthropic-messages",
            provider="anthropic",
            model=_MODEL,
            content=[
                ThinkingContent(thinking="pure reasoning", thinking_signature="sig-z"),
            ],
        ),
        UserMessage(content="next"),
    ]

    payload = _build(messages)

    assert [m["role"] for m in payload["messages"]] == ["user", "user"]
    assert not any(_has_thinking(m) for m in payload["messages"])


def test_redacted_thinking_is_stripped_with_the_rest_of_the_past() -> None:
    """`redacted_thinking` counts as thinking when history is stripped.

    Raw payload messages go through the pruner directly here: tau's own
    message conversion never emits ``redacted_thinking`` on its own, so the
    only way a ``redacted_thinking`` block reaches the pruning step is as an
    already-converted payload block — which is exactly what the pruner must
    recognise. A completed turn loses its redacted block (and keeps the rest
    of the turn); the live continuation keeps it, signature and all.
    """
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "start"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "redacted_thinking",
                    "data": "bG9yZW0=",
                    "signature": "sig-redacted",
                },
                {"type": "text", "text": "first answer"},
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "keep going"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "redacted_thinking",
                    "data": "aXBzdW0=",
                    "signature": "sig-redacted-live",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_61",
                    "name": "read",
                    "input": {"path": "a.py"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_61",
                    "content": "body",
                },
            ],
        },
    ]

    _prune_replayed_thinking(messages)

    completed = messages[1]
    assert not _has_thinking(completed)
    assert [block["type"] for block in _blocks(completed)] == ["text"]
    live = messages[3]
    assert _has_thinking(live)
    assert [(block["type"], block.get("signature")) for block in _blocks(live)] == [
        ("redacted_thinking", "sig-redacted-live"),
        ("tool_use", None),
    ]


@pytest.mark.anyio
async def test_signature_400_is_retried_once_without_thinking_and_the_run_continues() -> None:
    """One signature 400 costs a retry without thinking, not the run."""
    captured: list[dict[str, Any]] = []
    provider = _provider(
        [
            httpx.Response(
                400,
                text=_SIGNATURE_400,
                headers={"content-type": "application/json"},
            ),
        ],
        captured,
    )

    events = [
        event
        async for event in provider.stream_response(
            model=_MODEL,
            system="You are Tau.",
            messages=_thinking_history(),
            tools=[],
        )
    ]

    assert isinstance(events[-1], AssistantDoneEvent), events[-1]
    assert events[-1].message.text == "ok"
    assert len(captured) == 2
    assert "thinking" not in captured[1]
    assert not any(_has_thinking(m) for m in captured[1]["messages"])


@pytest.mark.anyio
async def test_recovery_does_not_loop_when_the_provider_rejects_again() -> None:
    """Recovery is one rebuilt request, not a retry loop on every 400."""
    captured: list[dict[str, Any]] = []
    provider = _provider(
        [
            httpx.Response(
                400,
                text=_SIGNATURE_400,
                headers={"content-type": "application/json"},
            ),
            httpx.Response(
                400,
                text=_SIGNATURE_400,
                headers={"content-type": "application/json"},
            ),
        ],
        captured,
    )

    events = [
        event
        async for event in provider.stream_response(
            model=_MODEL,
            system="You are Tau.",
            messages=_thinking_history(),
            tools=[],
        )
    ]

    assert len(captured) == 2
    assert "thinking" not in captured[1]
    assert isinstance(events[-1], AssistantErrorEvent)


@pytest.mark.anyio
async def test_thinking_returns_at_the_next_user_message() -> None:
    """A new user message starts a loop with no stale signature to replay."""
    captured: list[dict[str, Any]] = []
    sse = {"content-type": "text/event-stream"}
    provider = _provider(
        [
            httpx.Response(400, text=_SIGNATURE_400, headers={"content-type": "application/json"}),
            httpx.Response(200, text=_OK_SSE, headers=sse),
            httpx.Response(200, text=_OK_SSE, headers=sse),
        ],
        captured,
    )

    for messages in (_thinking_history(), [UserMessage(content="A new conversation")]):
        _ = [
            event
            async for event in provider.stream_response(
                model=_MODEL, system="You are Tau.", messages=messages, tools=[]
            )
        ]

    assert "thinking" not in captured[1]
    assert "thinking" in captured[2]


@pytest.mark.anyio
async def test_recovery_sticks_across_requests_even_with_live_thinking_history() -> None:
    """After one rejection, later requests carry no thinking anywhere."""
    captured: list[dict[str, Any]] = []
    provider = _provider(
        [
            httpx.Response(
                400,
                text=_SIGNATURE_400,
                headers={"content-type": "application/json"},
            ),
        ],
        captured,
    )

    first = [
        event
        async for event in provider.stream_response(
            model=_MODEL,
            system="You are Tau.",
            messages=_thinking_history(),
            tools=[],
        )
    ]
    assert isinstance(first[-1], AssistantDoneEvent)
    assert _has_thinking(captured[0]["messages"][1])

    # A later request whose history still has a live continuation: thinking is
    # back in the conversation, but this provider no longer replays or enables it.
    second = [
        event
        async for event in provider.stream_response(
            model=_MODEL,
            system="You are Tau.",
            messages=_thinking_history()
            + [
                AssistantMessage(
                    api="anthropic-messages",
                    provider="anthropic",
                    model=_MODEL,
                    content=[
                        ThinkingContent(thinking="more", thinking_signature="sig-3"),
                        TextContent(text="answered"),
                    ],
                ),
                UserMessage(content="next task"),
                _assistant_thinking_turn(tool_call_id="toolu_31", text=None),
                ToolResultMessage(
                    tool_call_id="toolu_31",
                    tool_name="read",
                    content="still here",
                ),
            ],
            tools=[],
        )
    ]

    assert isinstance(second[-1], AssistantDoneEvent)
    # captured[0] replayed thinking, captured[1] was the recovery retry, and
    # captured[2] is this later request: still no thinking, anywhere.
    assert len(captured) == 3
    assert "thinking" not in captured[2]
    assert not any(_has_thinking(m) for m in captured[2]["messages"])


@pytest.mark.anyio
async def test_thinking_replay_latch_is_per_provider_instance() -> None:
    """The recovery latch lives on the provider, not on the module.

    Two conversations in one process get one provider each; the one that took
    the signature 400 stops replaying thinking, and the other must still send
    it — a module-level latch would silently disable thinking for every
    session after the first rejection.
    """
    latched_captured: list[dict[str, Any]] = []
    latched = _provider(
        [
            httpx.Response(
                400,
                text=_SIGNATURE_400,
                headers={"content-type": "application/json"},
            ),
        ],
        latched_captured,
    )
    events = [
        event
        async for event in latched.stream_response(
            model=_MODEL,
            system="You are Tau.",
            messages=_thinking_history(),
            tools=[],
        )
    ]
    assert isinstance(events[-1], AssistantDoneEvent)
    assert len(latched_captured) == 2
    assert "thinking" not in latched_captured[1]

    fresh_captured: list[dict[str, Any]] = []
    fresh = _provider([], fresh_captured)
    events2 = [
        event
        async for event in fresh.stream_response(
            model=_MODEL,
            system="You are Tau.",
            messages=_thinking_history(),
            tools=[],
        )
    ]
    assert isinstance(events2[-1], AssistantDoneEvent)
    assert len(fresh_captured) == 1
    assert fresh_captured[0]["thinking"] == {
        "type": "enabled",
        "budget_tokens": 1024,
    }
    assert _has_thinking(fresh_captured[0]["messages"][1])


def _binding_provider(transport: _SignatureBindingTransport) -> AnthropicProvider:
    return AnthropicProvider(
        AnthropicConfig(
            api_key="test-key",
            base_url="https://example.test/v1",
            thinking_budget_tokens=1024,
        ),
        client=httpx.AsyncClient(transport=transport),
    )


@pytest.mark.anyio
async def test_regression_compacted_prefix_does_not_brick_the_session() -> None:
    """A compacted prefix does not end the run on a replayed signature.

    ``sig-1`` was issued for a conversation whose request prefix was the single
    user turn below. Compaction replaced that turn with a summary, so a replay
    of the block would be rejected by the provider. A completed turn's thinking
    is never replayed, so the request goes out clean on the first try and the
    run finishes.
    """
    transport = _SignatureBindingTransport(
        bound_prefixes={
            "sig-1": _canonical([{"role": "user", "content": "read a.py"}]),
        }
    )
    provider = _binding_provider(transport)

    compacted = [
        UserMessage(content="[compact] earlier: asked to read a.py"),
        _assistant_thinking_turn(text="read a.py, done"),
        UserMessage(content="now write b.py"),
    ]

    events = [
        event
        async for event in provider.stream_response(
            model=_MODEL,
            system="You are Tau.",
            messages=compacted,
            tools=[],
        )
    ]

    assert isinstance(events[-1], AssistantDoneEvent), events[-1]
    assert events[-1].message.text == "ok"
    # No 400 to recover from: nothing stale was replayed in the first place.
    assert len(transport.captured) == 1
    assert not any(_has_thinking(m) for m in transport.captured[0]["messages"])


@pytest.mark.anyio
async def test_regression_injected_context_over_past_thinking_does_not_brick() -> None:
    """Injected context between turns does not end the run on replayed thinking.

    An injected turn changes the prefix the first continuation's thinking was
    bound to, so replaying it would be rejected with the signature 400. Only
    the live continuation is replayed, and the injection did not disturb its
    binding.
    """
    # ``sig-1`` was issued when the request prefix was just the first user
    # turn; the injected turn that follows it makes any replay of ``sig-1``
    # invalid. ``sig-2`` belongs to the live continuation and is learned by the
    # transport on the request that carries it.
    transport = _SignatureBindingTransport(
        bound_prefixes={
            "sig-1": _canonical([{"role": "user", "content": "read a.py"}]),
        }
    )
    provider = _binding_provider(transport)

    messages = [
        UserMessage(content="read a.py"),
        _assistant_thinking_turn(tool_call_id="toolu_41", text=None),
        ToolResultMessage(tool_call_id="toolu_41", tool_name="read", content="body"),
        UserMessage(content="[injected] note: prefer relative paths"),
        _assistant_thinking_turn(tool_call_id="toolu_42", text=None, signature="sig-2"),
        ToolResultMessage(tool_call_id="toolu_42", tool_name="read", content="body two"),
    ]

    events = [
        event
        async for event in provider.stream_response(
            model=_MODEL,
            system="You are Tau.",
            messages=messages,
            tools=[],
        )
    ]

    assert isinstance(events[-1], AssistantDoneEvent), events[-1]
    # One request, answered ok: the stale signature never reached the provider.
    assert len(transport.captured) == 1
    captured_messages = transport.captured[0]["messages"]
    final_result_idx = len(captured_messages) - 1
    assert _has_tool_result(captured_messages[final_result_idx])
    for index, message in enumerate(captured_messages[:-1]):
        if index == final_result_idx - 1:
            continue  # the live continuation the final result answers
        assert not _has_thinking(message), f"stale thinking at index {index}"
    assert _has_thinking(captured_messages[final_result_idx - 1])


def test_past_thinking_is_absent_except_immediately_before_the_final_result() -> None:
    """A compacted, injected history sends thinking only for the live tool call."""
    messages = [
        UserMessage(content="[compact] earlier conversation summary"),
        _assistant_thinking_turn(text="earlier answer"),
        UserMessage(content="go on"),
        _assistant_thinking_turn(tool_call_id="toolu_51", text=None),
        ToolResultMessage(tool_call_id="toolu_51", tool_name="read", content="body"),
        UserMessage(content="[injected] prefer relative paths"),
        _assistant_thinking_turn(tool_call_id="toolu_52", text=None),
        ToolResultMessage(tool_call_id="toolu_52", tool_name="read", content="body two"),
    ]

    payload = _build(messages)
    payload_messages = payload["messages"]
    final_result_idx = len(payload_messages) - 1
    assert _has_tool_result(payload_messages[final_result_idx])
    live_idx = final_result_idx - 1
    assert payload_messages[live_idx]["role"] == "assistant"

    # Pre-fix code sent all three thinking blocks (six blocks across the three
    # assistant turns); exactly one survives now. This is the regression pin:
    # if pruning ever stops firing, this count goes back to three and fails.
    surviving = [block for message in payload_messages for block in _thinking_blocks(message)]
    assert len(surviving) == 1

    for index, message in enumerate(payload_messages):
        if index == live_idx:
            assert _has_thinking(message)
            for block in _thinking_blocks(message):
                assert block.get("signature") == "sig-1"
            continue
        assert not _has_thinking(message)
