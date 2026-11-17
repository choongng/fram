"""A stored turn that holds only reasoning is replayed as a well-formed turn.

A turn cut at its thinking budget, or one that ended before an answer or tool
call, holds reasoning and nothing else. Providers carry reasoning beside a
turn's content or not at all, so replayed as stored it reaches the server with
empty content, or is left out of the request and its two neighbouring user
turns end up adjacent. These tests build every provider's payload for such a
session and check that the turn is there, has text, and renders the same in
each later request.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytest

from pi_event_helpers import assistant_done, assistant_start
from tau_agent import (
    AgentHarness,
    AgentHarnessConfig,
    AgentMessage,
    AssistantMessage,
    ToolResultMessage,
    UserMessage,
)
from tau_agent.loop import (
    BUDGET_CUT_TURN_TEXT,
    EMPTY_TURN_TEXT,
    LOOPED_TURN_TEXT,
    REASONING_ONLY_TURN_TEXT,
    provider_context,
)
from tau_agent.messages import (
    REASONING_BUDGET_CUT,
    REASONING_LOOP_CUT,
    AssistantMessageDiagnostic,
    StopReason,
    TextContent,
    ThinkingContent,
    ToolCall,
)
from tau_ai import FakeProvider
from tau_ai.anthropic import _build_messages_payload
from tau_ai.google import _build_google_payload
from tau_ai.mistral import _build_mistral_payload
from tau_ai.openai_codex import _build_codex_payload
from tau_ai.openai_compatible import _build_chat_payload, _build_responses_payload

THOUGHT = "Work out the layout first. " * 20
Builder = Callable[[list[AgentMessage]], list[Mapping[str, Any]]]


def _chat(thinking_format: str) -> Builder:
    def build(messages: list[AgentMessage]) -> list[Mapping[str, Any]]:
        payload = _build_chat_payload(
            model="m",
            system="s",
            messages=messages,
            tools=[],
            reasoning_effort="medium",
            thinking_format=thinking_format,
        )
        return payload["messages"][1:]  # type: ignore[index,return-value]

    return build


def _responses(messages: list[AgentMessage]) -> list[Mapping[str, Any]]:
    return _build_responses_payload(model="m", system="s", messages=messages, tools=[])["input"]  # type: ignore[return-value]


def _codex(messages: list[AgentMessage]) -> list[Mapping[str, Any]]:
    return _build_codex_payload(model="m", system="s", messages=messages, tools=[])["input"]  # type: ignore[return-value]


def _anthropic(messages: list[AgentMessage]) -> list[Mapping[str, Any]]:
    return _build_messages_payload(model="m", system="s", messages=messages, tools=[])["messages"]  # type: ignore[return-value]


def _google(messages: list[AgentMessage]) -> list[Mapping[str, Any]]:
    payload = _build_google_payload(
        model="m", system="s", messages=messages, tools=[], reasoning_effort=None, max_tokens=None
    )
    return payload["contents"]  # type: ignore[return-value]


def _mistral(messages: list[AgentMessage]) -> list[Mapping[str, Any]]:
    payload = _build_mistral_payload(
        model="m", system="s", messages=messages, tools=[], reasoning_effort=None, max_tokens=None
    )
    return payload["messages"][1:]  # type: ignore[index,return-value]


# Each builder, with the `api` of the turns the provider itself stores.
BUILDERS: dict[str, tuple[Builder, str]] = {
    "openai-chat": (_chat("openai"), "openai-completions"),
    "qwen-chat-template": (_chat("qwen-chat-template"), "openai-completions"),
    "openai-responses": (_responses, "openai-responses"),
    "codex": (_codex, "openai-codex-responses"),
    "anthropic": (_anthropic, "anthropic-messages"),
    "anthropic-foreign-turn": (_anthropic, "openai-completions"),
    "google": (_google, "google-generative-ai"),
    "mistral": (_mistral, "mistral-conversations"),
}


def _reasoning_only(
    api: str, *, cut: str | None, stop_reason: StopReason = "length"
) -> AssistantMessage:
    diagnostics = [AssistantMessageDiagnostic(type=cut, timestamp=1)] if cut else None
    return AssistantMessage(
        content=[ThinkingContent(thinking=THOUGHT)],
        stop_reason=stop_reason,
        api=api,
        diagnostics=diagnostics,
    )


def _session(turn: AssistantMessage) -> list[AgentMessage]:
    return [
        UserMessage(content="Write the encoder."),
        turn,
        UserMessage(content="Act on it now."),
        AssistantMessage(content="Written.", api=turn.api),
        UserMessage(content="And the decoder."),
    ]


def _turns(items: list[Mapping[str, Any]]) -> list[tuple[str, str]]:
    """The role and visible text of each turn in a payload's message list.
    Reasoning carried as an item or field of its own is not visible text."""
    turns: list[tuple[str, str]] = []
    for item in items:
        role = item.get("role")
        if role is None:
            continue
        body = item.get("parts", item.get("content"))
        if isinstance(body, str):
            text = body
        else:
            text = "".join(
                block.get("text", "")
                for block in body or []
                if not block.get("thought") and block.get("type") != "thinking"
            )
        turns.append(("assistant" if role == "model" else role, text))
    return turns


@pytest.mark.parametrize("name", BUILDERS)
@pytest.mark.parametrize(
    ("cut", "stop_reason", "text"),
    [
        (REASONING_BUDGET_CUT, "length", BUDGET_CUT_TURN_TEXT),
        (REASONING_LOOP_CUT, "length", LOOPED_TURN_TEXT),
        (None, "length", REASONING_ONLY_TURN_TEXT),
        (None, "stop", REASONING_ONLY_TURN_TEXT),
    ],
)
def test_a_reasoning_only_turn_is_replayed_with_text(
    name: str, cut: str | None, stop_reason: StopReason, text: str
) -> None:
    build, api = BUILDERS[name]
    stored = _session(_reasoning_only(api, cut=cut, stop_reason=stop_reason))

    turns = _turns(build(provider_context(stored)))

    assert [role for role, _ in turns] == ["user", "assistant", "user", "assistant", "user"]
    assert turns[1] == ("assistant", text)
    assert all(visible.strip() for _, visible in turns)


@pytest.mark.parametrize("name", BUILDERS)
def test_a_budget_cut_turn_renders_the_same_in_every_later_request(name: str) -> None:
    """The server's prompt cache holds only while a replayed turn keeps its
    bytes as the conversation grows."""
    build, api = BUILDERS[name]
    stored = _session(_reasoning_only(api, cut=REASONING_BUDGET_CUT))

    def rendered(upto: int) -> Mapping[str, Any]:
        item = build(provider_context(stored[:upto]))[1]
        # Anthropic's cache breakpoint moves with the end of the request.
        return {key: value for key, value in item.items() if key != "cache_control"}

    assert rendered(3) == rendered(4) == rendered(5)


@pytest.mark.parametrize("name", ["openai-chat", "qwen-chat-template", "google", "mistral"])
def test_a_budget_cut_turn_keeps_its_reasoning_where_the_format_carries_it(name: str) -> None:
    build, api = BUILDERS[name]
    stored = _session(_reasoning_only(api, cut=REASONING_BUDGET_CUT))

    item = build(provider_context(stored))[1]

    carried = item.get("reasoning_content") or item["parts"][0]["text"]
    assert carried == THOUGHT


def test_the_stored_turn_is_left_as_it_was() -> None:
    turn = _reasoning_only("openai-completions", cut=REASONING_BUDGET_CUT)
    stored = _session(turn)
    before = turn.model_dump()

    replayed = provider_context(stored)[1]

    assert turn.model_dump() == before
    assert isinstance(replayed, AssistantMessage)
    assert replayed.content == [
        ThinkingContent(thinking=THOUGHT),
        TextContent(text=BUDGET_CUT_TURN_TEXT),
    ]


def test_a_turn_with_text_or_a_tool_call_is_replayed_unchanged() -> None:
    spoken = AssistantMessage(
        content=[ThinkingContent(thinking=THOUGHT), TextContent(text="Done.")]
    )
    called = AssistantMessage(
        content=[
            ThinkingContent(thinking=THOUGHT),
            TextContent(text="\n"),
            ToolCall(id="call-1", name="read", arguments={"path": "a.py"}),
        ],
        stop_reason="toolUse",
    )
    stored: list[AgentMessage] = [
        UserMessage(content="Go."),
        spoken,
        UserMessage(content="Read it."),
        called,
        ToolResultMessage(tool_call_id="call-1", tool_name="read", content="x = 1"),
    ]

    replayed = provider_context(stored)

    assert replayed[1] is spoken
    assert replayed[3] is called


@pytest.mark.parametrize("name", BUILDERS)
@pytest.mark.parametrize("stop_reason", ["error", "aborted"])
def test_a_failed_reasoning_only_turn_is_replayed_with_text(
    name: str, stop_reason: StopReason
) -> None:
    """A turn that failed or was interrupted after some reasoning is kept in
    the session; once a message follows it, it is sent like any other."""
    build, api = BUILDERS[name]
    failed = _reasoning_only(api, cut=None, stop_reason=stop_reason)
    stored = _session(failed.model_copy(update={"error_message": "connection lost"}))

    turns = _turns(build(provider_context(stored)))

    assert [role for role, _ in turns] == ["user", "assistant", "user", "assistant", "user"]
    assert turns[1] == ("assistant", REASONING_ONLY_TURN_TEXT)


@pytest.mark.parametrize("name", BUILDERS)
@pytest.mark.parametrize("stop_reason", ["error", "aborted", "length", "stop"])
def test_a_reasoning_only_turn_at_the_end_of_a_request_is_left_out(
    name: str, stop_reason: StopReason
) -> None:
    """A request that ended with the turn and its line of text would hand the
    model the line as the start of its reply, or be refused by a server that
    wants the user to speak last."""
    build, api = BUILDERS[name]
    stored = _session(_reasoning_only(api, cut=None, stop_reason=stop_reason))[:2]

    turns = _turns(build(provider_context(stored)))

    assert turns == [("user", "Write the encoder.")]


@pytest.mark.parametrize("stop_reason", ["error", "aborted", "length", "stop"])
def test_a_turn_reads_the_same_before_and_after_a_message_follows_it(
    stop_reason: StopReason,
) -> None:
    """Leaving the last turn out does not move the cached prefix: what came
    before it is sent the same, and from the first request that has the turn
    it never changes."""
    stored = _session(_reasoning_only("openai-completions", cut=None, stop_reason=stop_reason))

    trailing, followed, later = (provider_context(stored[:upto]) for upto in (2, 3, 5))

    assert followed[: len(trailing)] == trailing
    assert later[: len(followed)] == followed


@pytest.mark.anyio
async def test_continuing_after_a_failed_reasoning_only_turn_ends_on_the_user() -> None:
    failed = AssistantMessage(
        content=[ThinkingContent(thinking=THOUGHT)],
        stop_reason="error",
        error_message="connection lost",
    )
    provider = FakeProvider(
        [[assistant_start(), assistant_done(AssistantMessage(content="Written."))]]
    )
    harness = AgentHarness(
        AgentHarnessConfig(provider=provider, model="fake", system=""),
        messages=[UserMessage(content="Write the encoder."), failed],
    )

    _ = [event async for event in harness.continue_()]

    [(_model, _system, sent, _tools)] = provider.calls
    assert sent == [UserMessage(content="Write the encoder.")]
    # The failed turn stays in the session, and is sent once the reply follows it.
    assert harness.messages[1] is failed
    after = provider_context(list(harness.messages))
    assert [message.text for message in after[1:]] == [REASONING_ONLY_TURN_TEXT, "Written."]
    assert after[:1] == sent


def _blank(*content: object, **fields: object) -> AssistantMessage:
    return AssistantMessage(content=list(content), **fields)  # type: ignore[arg-type]


LOOP_CUT = [AssistantMessageDiagnostic(type=REASONING_LOOP_CUT, timestamp=1)]
BUDGET_CUT = [AssistantMessageDiagnostic(type=REASONING_BUDGET_CUT, timestamp=1)]
THINKING = ThinkingContent(thinking=THOUGHT)


@pytest.mark.parametrize(
    ("turn", "content"),
    [
        # Blank text is no reply: the line takes its place.
        (
            _blank(THINKING, TextContent(text="\n\n")),
            [THINKING, TextContent(text=REASONING_ONLY_TURN_TEXT)],
        ),
        (
            _blank(TextContent(text=" "), THINKING, stop_reason="length", diagnostics=BUDGET_CUT),
            [THINKING, TextContent(text=BUDGET_CUT_TURN_TEXT)],
        ),
        (
            _blank(THINKING, TextContent(text="\n"), stop_reason="length", diagnostics=LOOP_CUT),
            [TextContent(text=LOOPED_TURN_TEXT)],
        ),
        # A turn with no reasoning did not end "after its reasoning".
        (_blank(TextContent(text="\n\n")), [TextContent(text=EMPTY_TURN_TEXT)]),
        (_blank(), [TextContent(text=EMPTY_TURN_TEXT)]),
        (
            _blank(ThinkingContent(thinking="")),
            [
                ThinkingContent(thinking=""),
                TextContent(text=REASONING_ONLY_TURN_TEXT),
            ],
        ),
    ],
)
def test_blank_text_is_replaced_by_the_line(turn: AssistantMessage, content: list[object]) -> None:
    replayed = provider_context(_session(turn))[1]

    assert isinstance(replayed, AssistantMessage)
    assert replayed.content == content


@pytest.mark.parametrize("name", BUILDERS)
def test_a_blank_reply_reaches_no_provider_as_blank_text(name: str) -> None:
    build, api = BUILDERS[name]
    blank = AssistantMessage(content=[TextContent(text="\n\n")], api=api)

    turns = _turns(build(provider_context(_session(blank))))

    assert turns[1] == ("assistant", EMPTY_TURN_TEXT)


@pytest.mark.parametrize("stop_reason", ["error", "aborted", "length"])
def test_a_failed_turn_with_only_blank_text_is_not_replayed(stop_reason: StopReason) -> None:
    blank = AssistantMessage(content=[TextContent(text="\n")], stop_reason=stop_reason)

    replayed = provider_context(_session(blank))

    assert [type(message) for message in replayed] == [
        UserMessage,
        UserMessage,
        AssistantMessage,
        UserMessage,
    ]
