"""Pi-compatible coding-session events consumed by frontends and SDK users."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from tau_agent.events import AgentEvent
from tau_agent.messages import AgentMessage, WireModel
from tau_agent.session.entries import SessionEntry


class SessionAgentEndEvent(WireModel):
    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage] = Field(default_factory=list)
    will_retry: bool = Field(False)


class AgentSettledEvent(WireModel):
    type: Literal["agent_settled"] = "agent_settled"


class QueueUpdateEvent(WireModel):
    type: Literal["queue_update"] = "queue_update"
    steering: tuple[str, ...] = ()
    follow_up: tuple[str, ...] = Field(())


CompactionReason = Literal["manual", "threshold", "overflow"]


class CompactionStartEvent(WireModel):
    type: Literal["compaction_start"] = "compaction_start"
    reason: CompactionReason


class CompactionEndEvent(WireModel):
    type: Literal["compaction_end"] = "compaction_end"
    reason: CompactionReason
    result: object | None = None
    aborted: bool = False
    will_retry: bool = Field(False)
    error_message: str | None = Field(None)


class EntryAppendedEvent(WireModel):
    type: Literal["entry_appended"] = "entry_appended"
    entry: SessionEntry


class SessionInfoChangedEvent(WireModel):
    type: Literal["session_info_changed"] = "session_info_changed"
    name: str | None = None


class ThinkingLevelChangedEvent(WireModel):
    type: Literal["thinking_level_changed"] = "thinking_level_changed"
    level: str


class AutoRetryStartEvent(WireModel):
    type: Literal["auto_retry_start"] = "auto_retry_start"
    attempt: int
    max_attempts: int
    delay_ms: int
    error_message: str


class AutoRetryEndEvent(WireModel):
    type: Literal["auto_retry_end"] = "auto_retry_end"
    success: bool
    attempt: int
    final_error: str | None = Field(None)


class PromptCacheMissEvent(WireModel):
    """A large prompt was fully reprocessed by a provider that usually caches it.

    Emitted after the assistant message whose usage reported no cache read,
    once that provider and model have reported a cache hit earlier in the
    session. See `tau_coding.prompt_cache`.
    """

    type: Literal["prompt_cache_miss"] = "prompt_cache_miss"
    provider: str
    model: str
    prompt_tokens: int

    @property
    def status_text(self) -> str:
        """One-line status message frontends show for this miss."""
        return (
            f"Prompt cache miss: {self.prompt_tokens:,} prompt tokens reprocessed "
            f"(earlier {self.provider} turns were served from cache)"
        )


type SessionOwnEvent = Annotated[
    SessionAgentEndEvent
    | AgentSettledEvent
    | QueueUpdateEvent
    | CompactionStartEvent
    | CompactionEndEvent
    | EntryAppendedEvent
    | SessionInfoChangedEvent
    | ThinkingLevelChangedEvent
    | AutoRetryStartEvent
    | AutoRetryEndEvent
    | PromptCacheMissEvent,
    Field(discriminator="type"),
]
type CodingSessionEvent = AgentEvent | SessionOwnEvent
type AgentSessionEvent = CodingSessionEvent
