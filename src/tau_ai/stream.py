"""Translate Tau's transitional provider parser output into Pi stream events."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import aclosing
from typing import Any, cast

from tau_agent.messages import (
    AssistantMessage,
    AssistantMessageDiagnostic,
    TextContent,
    ThinkingContent,
    Usage,
)
from tau_ai._provider_events import (
    ProviderErrorEvent,
    ProviderEvent,
    ProviderResponseEndEvent,
    ProviderResponseStartEvent,
    ProviderRetryEvent,
    ProviderTextDeltaEvent,
    ProviderThinkingDeltaEvent,
    ProviderToolCallEvent,
)
from tau_ai.events import (
    AssistantDoneEvent,
    AssistantErrorEvent,
    AssistantMessageEvent,
    AssistantRetryEvent,
    AssistantStartEvent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)


def _snapshot(message: AssistantMessage) -> AssistantMessage:
    return message.model_copy(deep=True)


async def _end_active_block(
    partial: AssistantMessage,
    index: int | None,
) -> AsyncIterator[AssistantMessageEvent]:
    """End the active text/thinking block before the provider changes channels."""
    if index is None:
        return
    block = partial.content[index]
    if isinstance(block, TextContent):
        yield TextEndEvent(
            content_index=index,
            content=block.text,
            partial=_snapshot(partial),
        )
    elif isinstance(block, ThinkingContent):
        yield ThinkingEndEvent(
            content_index=index,
            content=block.thinking,
            partial=_snapshot(partial),
        )


def _copy_replay_metadata(target: AssistantMessage, source: AssistantMessage) -> None:
    """Copy provider metadata onto streamed blocks without changing their order."""
    source_thinking = [block for block in source.content if isinstance(block, ThinkingContent)]
    target_thinking = [block for block in target.content if isinstance(block, ThinkingContent)]
    for target_block, source_block in zip(target_thinking, source_thinking, strict=False):
        target_block.thinking_signature = source_block.thinking_signature
        target_block.redacted = source_block.redacted

    source_text = [block for block in source.content if isinstance(block, TextContent)]
    target_text = [block for block in target.content if isinstance(block, TextContent)]
    for target_text_block, source_text_block in zip(target_text, source_text, strict=False):
        target_text_block.text_signature = source_text_block.text_signature


#: A Responses-API reply stopped by a content filter. Unlike the output limit,
#: it says nothing about whether the last tool call was cut.
FILTERED_INCOMPLETE = "incomplete:content_filter"


def responses_status(response: Mapping[str, Any]) -> str | None:
    """A Responses-API response's status, telling a content filter from the limit."""
    status = response.get("status")
    if not isinstance(status, str):
        return None
    details = response.get("incomplete_details")
    if (
        status == "incomplete"
        and isinstance(details, Mapping)
        and details.get("reason") == "content_filter"
    ):
        return FILTERED_INCOMPLETE
    return status


def _finish_reason(value: str | None, *, has_tools: bool) -> str:
    # The output limit wins over a tool call: the call it ends is cut off, and
    # the agent loop answers it without running it.
    if value in {"length", "max_tokens", "MAX_TOKENS", "incomplete"}:
        return "length"
    if has_tools or value in {"tool_calls", "tool_use", "toolUse"}:
        return "toolUse"
    if value == FILTERED_INCOMPLETE:
        return "length"
    return "stop"


async def canonicalize_provider_stream(
    source: AsyncIterator[ProviderEvent],
    *,
    api: str,
    provider: str,
    model: str,
    independent_channels: bool = False,
) -> AsyncIterator[AssistantMessageEvent]:
    """Canonicalize one old internal parser stream.

    Provider parsers remain isolated behind this private bridge while they are
    migrated incrementally. The public provider protocol exposes only Pi events.
    Chat Completions uses independent channels: a channel switch does not end
    either block. Other transports preserve their sequential block ordering.
    """
    partial = AssistantMessage(api=api, provider=provider, model=model)
    active_index: int | None = None
    active_kind: str | None = None
    channel_indices: dict[str, int] = {}
    started = False
    terminal = False

    # A consumer that stops early must release the HTTP response and timers now.
    async with aclosing(cast("AsyncGenerator[ProviderEvent, None]", source)):
        async for event in source:
            if isinstance(event, ProviderRetryEvent):
                # Retries never become assistant content; they are surfaced as a
                # transient status (AssistantRetryEvent -> ProviderRetryEvent) so a
                # frontend can say why the agent is quiet between attempts.
                yield AssistantRetryEvent(
                    attempt=event.attempt,
                    max_attempts=event.max_attempts,
                    delay_seconds=event.delay_seconds,
                    message=event.message,
                )
                continue
            if isinstance(event, ProviderResponseStartEvent):
                if event.response_provider is not None:
                    partial.response_provider = event.response_provider
                if not started:
                    started = True
                    yield AssistantStartEvent(partial=_snapshot(partial))
                continue
            if not started:
                started = True
                yield AssistantStartEvent(partial=_snapshot(partial))

            if isinstance(event, ProviderTextDeltaEvent):
                if independent_channels:
                    active_index = channel_indices.get("text")
                    active_kind = "text" if active_index is not None else None
                if active_kind != "text":
                    if not independent_channels:
                        async for end_event in _end_active_block(partial, active_index):
                            yield end_event
                    active_index = len(partial.content)
                    active_kind = "text"
                    partial.content.append(TextContent(text=""))
                    channel_indices["text"] = active_index
                    yield TextStartEvent(content_index=active_index, partial=_snapshot(partial))
                assert active_index is not None
                block = partial.content[active_index]
                assert isinstance(block, TextContent)
                block.text += event.delta
                yield TextDeltaEvent(
                    content_index=active_index,
                    delta=event.delta,
                    partial=_snapshot(partial),
                )
            elif isinstance(event, ProviderThinkingDeltaEvent):
                if independent_channels:
                    active_index = channel_indices.get("thinking")
                    active_kind = "thinking" if active_index is not None else None
                if active_kind != "thinking":
                    if not independent_channels:
                        async for end_event in _end_active_block(partial, active_index):
                            yield end_event
                    active_index = len(partial.content)
                    active_kind = "thinking"
                    partial.content.append(ThinkingContent(thinking=""))
                    channel_indices["thinking"] = active_index
                    yield ThinkingStartEvent(
                        content_index=active_index,
                        partial=_snapshot(partial),
                    )
                assert active_index is not None
                block = partial.content[active_index]
                assert isinstance(block, ThinkingContent)
                block.thinking += event.delta
                yield ThinkingDeltaEvent(
                    content_index=active_index,
                    delta=event.delta,
                    partial=_snapshot(partial),
                )
            elif isinstance(event, ProviderToolCallEvent):
                indices = list(channel_indices.values()) if independent_channels else [active_index]
                for index in indices:
                    async for end_event in _end_active_block(partial, index):
                        yield end_event
                channel_indices.clear()
                active_index = None
                active_kind = None
                index = len(partial.content)
                partial.content.append(event.tool_call.model_copy(deep=True))
                yield ToolCallStartEvent(content_index=index, partial=_snapshot(partial))
                yield ToolCallEndEvent(
                    content_index=index,
                    tool_call=event.tool_call,
                    partial=_snapshot(partial),
                )
            elif isinstance(event, ProviderResponseEndEvent):
                indices = list(channel_indices.values()) if independent_channels else [active_index]
                for index in indices:
                    async for end_event in _end_active_block(partial, index):
                        yield end_event
                channel_indices.clear()
                active_index = None
                active_kind = None

                # Preserve the exact streamed content order. The parser's final
                # message remains authoritative only for response metadata/usage.
                final = event.message.model_copy(deep=True)
                final.api = api
                final.provider = provider
                final.model = model
                final.response_provider = (
                    event.message.response_provider or partial.response_provider
                )
                final.content = [block.model_copy(deep=True) for block in partial.content]
                if not final.content and event.message.content:
                    final.content = [block.model_copy(deep=True) for block in event.message.content]
                _copy_replay_metadata(final, event.message)
                final.stop_reason = _finish_reason(
                    event.finish_reason,
                    has_tools=bool(final.tool_calls),
                )  # type: ignore[assignment]
                yield AssistantDoneEvent(reason=final.stop_reason, message=final)  # type: ignore[arg-type]
                terminal = True
            elif isinstance(event, ProviderErrorEvent):
                error = partial.model_copy(deep=True)
                error.response_provider = event.response_provider or partial.response_provider
                error.stop_reason = "error"
                error.error_message = event.message
                error.diagnostics = [
                    AssistantMessageDiagnostic(type="provider_error", details=event.data)
                ]
                yield AssistantErrorEvent(reason="error", error=error)
                terminal = True

    if not started:
        yield AssistantStartEvent(partial=_snapshot(partial))
    if not terminal:
        error = partial.model_copy(deep=True)
        error.stop_reason = "error"
        error.error_message = "Provider stream ended without a terminal event"
        error.usage = Usage()
        yield AssistantErrorEvent(reason="error", error=error)
