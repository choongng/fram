"""Pi-compatible events emitted by Tau's portable agent layer."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from tau_agent.messages import AgentMessage, ToolResultMessage, WireModel
from tau_agent.provider_events import AssistantMessageEvent
from tau_agent.tools import AgentToolResult
from tau_agent.types import JSONValue


class AgentStartEvent(WireModel):
    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(WireModel):
    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage] = Field(default_factory=list)


class TurnStartEvent(WireModel):
    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(WireModel):
    type: Literal["turn_end"] = "turn_end"
    message: AgentMessage
    tool_results: list[ToolResultMessage] = Field(default_factory=list)


class MessageStartEvent(WireModel):
    type: Literal["message_start"] = "message_start"
    message: AgentMessage


class MessageUpdateEvent(WireModel):
    type: Literal["message_update"] = "message_update"
    message: AgentMessage
    assistant_message_event: AssistantMessageEvent = Field(
        serialization_alias="assistantMessageEvent"
    )


class MessageEndEvent(WireModel):
    type: Literal["message_end"] = "message_end"
    message: AgentMessage


class ToolExecutionStartEvent(WireModel):
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: dict[str, JSONValue] = Field(default_factory=dict)


class ToolExecutionUpdateEvent(WireModel):
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    args: dict[str, JSONValue] = Field(default_factory=dict)
    partial_result: AgentToolResult


class ToolExecutionEndEvent(WireModel):
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: AgentToolResult
    is_error: bool


class ProviderRetryEvent(WireModel):
    """The provider is re-sending a request that transiently failed.

    Emitted while the agent waits between transport attempts (a dropped
    connection, a transient HTTP error, a stream that sent nothing), so a
    frontend can say why the agent is quiet.
    Transient: nothing is persisted and no assistant content exists for it.
    """

    type: Literal["provider_retry"] = "provider_retry"
    attempt: int
    max_attempts: int
    delay_seconds: float
    message: str


type AgentEvent = Annotated[
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
    | ProviderRetryEvent,
    Field(discriminator="type"),
]
