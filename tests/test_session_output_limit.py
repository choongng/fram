"""A turn whose reasoning loops is cut early, and the output-limit nudge."""

from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import pytest

from pi_event_helpers import assistant_done, assistant_start
from tau_agent import (
    AgentMessage,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    UserMessage,
)
from tau_agent.messages import assistant_content
from tau_agent.provider_events import ThinkingDeltaEvent
from tau_agent.session import JsonlSessionStorage
from tau_ai import CancellationToken, FakeProvider
from tau_ai.events import AssistantMessageEvent
from tau_coding import (
    CodingSession,
    CodingSessionConfig,
)
from tau_coding.guards import LOOP_CUT_NUDGE, OUTPUT_LIMIT_NUDGE

# Reasoning that opens like this, then goes round the same block until the
# output limit with no tool call.
LOOP_OPENING = """\
The check says "no change" because the `name` and `notes` are not different enough.
I'll change the `name` to be very different.
I'll use a `python3 -c` command that is very different.
I'll also include the `labels`.
I'll also include the `notes`.

"""
LOOP_ROUND = """\
Actually, I'll just run it.
I'll also add `docs/` to the scope.
I'll also check if `tests/test_case_data.py` exists.
I'll just run the step.

Actually, I'll use a different `name` that is very different.
I'll use a `python3 -c` command that is very different.
I'll also include the `labels`.
I'll also include the `notes`.

"""
CALL = ToolCall(id="call-1", name="fetch", arguments={})


class ClosingStreams:
    """A fake provider that records how far each stream was read."""

    def __init__(self, streams: Iterable[Iterable[AssistantMessageEvent]]) -> None:
        self._streams = [list(stream) for stream in streams]
        self.calls: list[list[AgentMessage]] = []
        self.read: list[int] = []
        self.closed: list[bool] = []

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        self.calls.append(list(messages))
        stream = self._streams.pop(0)
        index = len(self.read)
        self.read.append(0)
        self.closed.append(False)

        async def iterator() -> AsyncIterator[AssistantMessageEvent]:
            try:
                for event in stream:
                    self.read[index] += 1
                    yield event
            finally:
                self.closed[index] = True

        return iterator()


def _fetch_tool() -> AgentTool:
    async def fetch(
        tool_call_id: object,
        arguments: object,
        signal: object = None,
        on_update: object = None,
    ) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="fetched")])

    return AgentTool(
        name="fetch",
        label="Fetch",
        description="Fetch a document.",
        parameters={"type": "object"},
        execute_fn=fetch,
    )


def _streamed_reasoning(text: str, *, then: AssistantMessage, chunk: int = 9) -> list[object]:
    """A response that streams `text` as reasoning, then ends with `then`."""
    deltas = [
        ThinkingDeltaEvent(
            content_index=0,
            delta=text[i : i + chunk],
            partial=AssistantMessage(content=[ThinkingContent(thinking=text[: i + chunk])]),
        )
        for i in range(0, len(text), chunk)
    ]
    return [assistant_start(), *deltas, assistant_done(message=then)]


def _acts() -> list[object]:
    return [assistant_start(), assistant_done(message=AssistantMessage(content=[CALL]))]


def _answers(text: str = "Done.") -> list[object]:
    return [assistant_start(), assistant_done(message=AssistantMessage(content=text))]


def _truncated() -> list[object]:
    return [
        assistant_start(),
        assistant_done(
            message=AssistantMessage(content=[ThinkingContent(thinking="Hmm. " * 50)]),
            finish_reason="length",
        ),
    ]


async def _session(tmp_path: Path, provider: object, **config: object) -> CodingSession:
    return await CodingSession.load(
        CodingSessionConfig(
            provider=provider,  # type: ignore[arg-type]
            model="fake",
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            tools=[_fetch_tool()],
            **config,  # type: ignore[arg-type]
        )
    )


async def _run(session: CodingSession, prompt: str = "Design it.") -> None:
    async for _event in session.prompt(prompt):
        pass


def _nudges(session: CodingSession) -> list[UserMessage]:
    return [
        message
        for message in session.messages
        if isinstance(message, UserMessage) and message.text in {OUTPUT_LIMIT_NUDGE, LOOP_CUT_NUDGE}
    ]


@pytest.mark.anyio
async def test_reasoning_repetition_stops_turn_as_output_limit(tmp_path: Path) -> None:
    looping = LOOP_OPENING + LOOP_ROUND * 40
    never_reached = AssistantMessage(content="The loop ran to its end.")
    provider = ClosingStreams(
        [_streamed_reasoning(looping, then=never_reached), _acts(), _answers()]
    )
    session = await _session(tmp_path, provider)

    await _run(session)

    cut = session.messages[1]
    assert isinstance(cut, AssistantMessage)
    assert cut.stop_reason == "length"
    assert [diagnostic.type for diagnostic in cut.diagnostics or []] == ["reasoning_loop_cut"]
    # Stopped well inside the loop, and the request was closed there.
    assert 0 < len(cut.thinking_text) < len(looping) // 4
    assert looping.startswith(cut.thinking_text)
    assert provider.read[0] < len(looping) // 9 // 4
    assert provider.closed[0]
    # Handled as a cut-off at the cap: nudged, and the reply acted. A loop's
    # reasoning is not replayed, and the nudge says so.
    assert provider.calls[1][-1].text == LOOP_CUT_NUDGE
    assert not any(
        isinstance(message, AssistantMessage) and message.thinking_text
        for message in provider.calls[1]
    )
    assert len(provider.calls) == 3
    assert session.messages[-1].text == "Done."


@pytest.mark.anyio
@pytest.mark.parametrize(
    "looping",
    [
        "<|channel>thought\n" * 200,  # one token, a line each
        "thought " * 1_000,  # the same run with no line breaks
    ],
)
async def test_degenerate_token_run_is_cut(tmp_path: Path, looping: str) -> None:
    provider = ClosingStreams(
        [_streamed_reasoning(looping, then=AssistantMessage(content="x")), _answers()]
    )
    session = await _session(tmp_path, provider)

    await _run(session)

    cut = session.messages[1]
    assert isinstance(cut, AssistantMessage)
    assert cut.stop_reason == "length"
    assert len(cut.thinking_text) < len(looping)
    assert len(provider.calls) == 2


# One long line of reasoning whose sentences are mostly these two, alternating.
SENTENCE_OPENING = (
    "The first entry has the value 17, which would put the marker at slot 18. "
    "Looking at the example table again, I need to reconsider what the value means. "
)
SENTENCE_ROUND = (
    "I'm realizing the value at each slot says where the next marker goes, so I track my "
    "place and add markers as I go. "
    "The value tells me how far ahead the next marker is, and I copy entries until I "
    "reach it, then add the marker and continue. "
)


@pytest.mark.anyio
async def test_sentences_repeated_within_one_line_are_cut(tmp_path: Path) -> None:
    looping = SENTENCE_OPENING + SENTENCE_ROUND * 193
    assert "\n" not in looping
    provider = ClosingStreams(
        [_streamed_reasoning(looping, then=AssistantMessage(content="x")), _answers()]
    )
    session = await _session(tmp_path, provider)

    await _run(session)

    cut = session.messages[1]
    assert isinstance(cut, AssistantMessage)
    assert cut.stop_reason == "length"
    assert [diagnostic.type for diagnostic in cut.diagnostics or []] == ["reasoning_loop_cut"]
    assert 0 < len(cut.thinking_text) < len(looping) // 4
    assert provider.calls[1][-1].text == LOOP_CUT_NUDGE
    assert len(provider.calls) == 2


def _varied_reasoning(lines: int) -> str:
    """Long reasoning that comes back to a short line now and then."""
    out = []
    for i in range(lines):
        out.append(f"Case {i}: a node of depth {i % 7} splits into {i * 3 + 1} and {i * 5 + 2}.")
        if i % 60 == 0:
            out.append("Let me check that again.")
        if i % 9 == 0:
            out.append("")
    return "\n".join(out) + "\n"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "reasoning",
    [
        _varied_reasoning(400),
        # A long paragraph with no line breaks, varied throughout.
        " ".join(f"step {i} gives {i * i % 97}." for i in range(1_500)),
        # One long line of varied sentences that comes back to a short one often.
        " ".join(
            f"Byte {i} maps to {i * 7 % 256}." + (" Wait." if i % 5 == 0 else "")
            for i in range(600)
        ),
        # A short line said a few times among varied ones, in a short reply.
        "Hmm.\nThe left half is 1.\nHmm.\nThe right half is 0.\nHmm.\nSo the id is (1, 0).\n" * 3,
    ],
)
async def test_reasoning_without_repetition_is_not_cut(tmp_path: Path, reasoning: str) -> None:
    final = AssistantMessage(content=assistant_content("", [CALL]))
    provider = ClosingStreams([_streamed_reasoning(reasoning, then=final), _answers()])
    session = await _session(tmp_path, provider)

    await _run(session)

    message = session.messages[1]
    assert isinstance(message, AssistantMessage)
    assert message.stop_reason == "toolUse"
    assert not message.diagnostics
    assert not _nudges(session)
    assert len(provider.calls) == 2


@pytest.mark.anyio
async def test_per_run_nudge_bound_skips_nudges_whose_reply_acted(tmp_path: Path) -> None:
    # Each cut-off's nudge is answered with a tool call: the model is acting,
    # not thinking itself out of room, so the per-run bound of six does not
    # end the run at the seventh cut-off.
    provider = FakeProvider([*[_truncated(), _acts()] * 8, _answers()])
    session = await _session(tmp_path, provider)

    await _run(session)

    assert len(_nudges(session)) == 8
    assert session.messages[-1].text == "Done."


@pytest.mark.anyio
async def test_in_a_row_bound_still_stops_a_model_that_never_acts(tmp_path: Path) -> None:
    provider = FakeProvider([_truncated() for _ in range(5)])
    session = await _session(tmp_path, provider)

    await _run(session)

    assert len(_nudges(session)) == 2
    assert len(provider.calls) == 3
