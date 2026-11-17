"""Ctrl+T shows and hides every thinking block in the transcript at once.

Each test streams several turns that each think, toggles, and reads every
block's visibility off the mounted rows: after the turns have settled, and
while the last one is still streaming.
"""

import asyncio

import pytest
from textual.widgets import Static
from textual.widgets._markdown import MarkdownBlock

from tau_agent import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    AgentToolResult,
    AssistantMessage,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    UserMessage,
)
from tau_agent.provider_events import TextDeltaEvent, ThinkingDeltaEvent, ThinkingEndEvent
from tau_coding.tui.app import HIDDEN_THINKING_NOTICE, PromptInput, TauTuiApp
from tau_coding.tui.widgets import (
    StreamingTranscriptMessageWidget,
    ThinkingPlaceholder,
    TranscriptMessageWidget,
    TranscriptView,
)
from test_tui_app import FakeSession

TURNS = 3
# Positions in `_turn_events(..., with_tool=False)`.
AFTER_FIRST_THINKING_DELTA = 4
AFTER_THINKING = 5
AFTER_TEXT_STARTED = 6


def _thinking_delta(partial: AssistantMessage, delta: str) -> MessageUpdateEvent:
    return MessageUpdateEvent(
        message=partial,
        assistant_message_event=ThinkingDeltaEvent(content_index=0, delta=delta, partial=partial),
    )


def _text_delta(partial: AssistantMessage, delta: str) -> MessageUpdateEvent:
    return MessageUpdateEvent(
        message=partial,
        assistant_message_event=TextDeltaEvent(content_index=1, delta=delta, partial=partial),
    )


def _turn_events(index: int, *, with_tool: bool) -> list[AgentEvent]:
    """One prompt and its reply: thinking streamed in two deltas, then an answer."""
    thinking = f"plan {index}"
    answer = f"answer {index}"
    partial = AssistantMessage()
    events: list[AgentEvent] = [
        MessageEndEvent(message=UserMessage(content=f"prompt {index}")),
        AgentStartEvent(),
        MessageStartEvent(message=partial),
        _thinking_delta(partial, "plan "),
        _thinking_delta(partial, str(index)),
    ]
    final: list[ThinkingContent | TextContent]
    if with_tool:
        call = ToolCall(
            id=f"call-{index}",
            name="bash",
            arguments={"command": "ls", "description": f"Step {index}"},
        )
        events += [
            MessageEndEvent(
                message=AssistantMessage(content=[ThinkingContent(thinking=thinking), call])
            ),
            ToolExecutionStartEvent(tool_call_id=call.id, tool_name=call.name, args=call.arguments),
            ToolExecutionEndEvent(
                tool_call_id=call.id,
                tool_name=call.name,
                result=AgentToolResult(content="ok"),
                is_error=False,
            ),
        ]
        partial = AssistantMessage()
        events.append(MessageStartEvent(message=partial))
        final = [TextContent(text=answer)]
    else:
        final = [ThinkingContent(thinking=thinking), TextContent(text=answer)]
    events += [
        _text_delta(partial, answer),
        MessageEndEvent(message=AssistantMessage(content=final)),
        AgentEndEvent(),
    ]
    return events


def _agentic_run(turn: int, steps: int) -> list[AgentEvent]:
    """One agent run of several messages, each thinking, then calling a tool."""
    events: list[AgentEvent] = [AgentStartEvent()]
    for step in range(steps):
        thinking = f"plan {turn}.{step}"
        note = f"note {turn}.{step}"
        partial = AssistantMessage()
        events += [
            MessageStartEvent(message=partial),
            _thinking_delta(partial, thinking[:5]),
            _thinking_delta(partial, thinking[5:]),
            MessageUpdateEvent(
                message=partial,
                assistant_message_event=ThinkingEndEvent(
                    content_index=0, content=thinking, partial=partial
                ),
            ),
            _text_delta(partial, note),
        ]
        content: list[ThinkingContent | TextContent | ToolCall] = [
            ThinkingContent(thinking=thinking),
            TextContent(text=note),
        ]
        if step == steps - 1:
            events.append(MessageEndEvent(message=AssistantMessage(content=content)))
            continue
        call = ToolCall(
            id=f"call-{turn}-{step}",
            name="bash",
            arguments={"command": "ls", "description": f"Step {turn}.{step}"},
        )
        events += [
            MessageEndEvent(message=AssistantMessage(content=[*content, call])),
            ToolExecutionStartEvent(tool_call_id=call.id, tool_name=call.name, args=call.arguments),
            ToolExecutionEndEvent(
                tool_call_id=call.id,
                tool_name=call.name,
                result=AgentToolResult(content="ok"),
                is_error=False,
            ),
        ]
    events.append(AgentEndEvent())
    return events


def _rows(app: TauTuiApp) -> list[TranscriptMessageWidget | StreamingTranscriptMessageWidget]:
    transcript = app.query_one("#transcript", TranscriptView)
    return [
        child
        for child in transcript.children
        if isinstance(child, TranscriptMessageWidget | StreamingTranscriptMessageWidget)
    ]


def _visible_thinking(app: TauTuiApp) -> list[str]:
    """The text of every thinking block on screen, in transcript order."""
    return [row.selection_text.strip() for row in _rows(app) if row.item.role == "thinking"]


def _rendered_thinking(app: TauTuiApp) -> list[list[str]]:
    """The Markdown blocks each thinking row draws, which can differ from its text."""
    return [
        [str(block.content) for block in row.query_children(MarkdownBlock)]
        for row in _rows(app)
        if row.item.role == "thinking"
    ]


def _notice_shown(app: TauTuiApp) -> bool:
    """Whether the notice says hidden thinking is happening.

    A shown notice sits over the transcript's blank placeholder row, the row
    the answer or tool call takes next; otherwise it draws nothing."""
    notice = app.query_one("#thinking-notice", Static)
    shown = str(notice.content) == HIDDEN_THINKING_NOTICE
    assert notice.visible == shown
    assert app.query_one("#prompt", PromptInput).border_title is None
    # The notice is never transcript text.
    assert all(row.selection_text.strip() != HIDDEN_THINKING_NOTICE for row in _rows(app))
    if shown:
        placeholder = app.query_one(ThinkingPlaceholder)
        assert placeholder.display
        assert notice.region.y == placeholder.region.y
    return shown


def _streamer(app: TauTuiApp):
    async def stream(event: AgentEvent) -> None:
        # The app projects a confirmed user message itself, as `_run_prompt`
        # does; handing it to the adapter too would add it twice.
        if not (isinstance(event, MessageEndEvent) and isinstance(event.message, UserMessage)):
            app.adapter.apply(event)
        await app._apply_streaming_transcript_event(event)

    return stream


async def _toggle(pilot) -> None:
    await pilot.press("ctrl+t")
    await pilot.pause()


@pytest.mark.anyio
@pytest.mark.parametrize("with_tool", [False, True])
@pytest.mark.parametrize("start_shown", [False, True])
async def test_thinking_toggle_all_after_streaming(with_tool: bool, start_shown: bool) -> None:
    app = TauTuiApp(FakeSession())
    stream = _streamer(app)
    every_plan = [f"plan {index}" for index in range(TURNS)]

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        if start_shown:
            await _toggle(pilot)
        for index in range(TURNS):
            for event in _turn_events(index, with_tool=with_tool):
                await stream(event)
                await pilot.pause()

        assert _visible_thinking(app) == (every_plan if start_shown else [])
        for _ in range(3):
            await _toggle(pilot)
            assert _visible_thinking(app) == (every_plan if app.state.show_thinking else [])
            assert not _notice_shown(app)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "toggle_at", [AFTER_FIRST_THINKING_DELTA, AFTER_THINKING, AFTER_TEXT_STARTED]
)
@pytest.mark.parametrize("start_shown", [False, True])
async def test_thinking_toggle_all_during_streaming(toggle_at: int, start_shown: bool) -> None:
    """A toggle while the last turn streams reaches the earlier turns, and the
    block still streaming stays one row that its MessageEnd replaces."""
    app = TauTuiApp(FakeSession())
    stream = _streamer(app)
    earlier = [f"plan {index}" for index in range(TURNS - 1)]
    every_plan = [*earlier, f"plan {TURNS - 1}"]

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        if start_shown:
            await _toggle(pilot)
        for index in range(TURNS):
            for position, event in enumerate(_turn_events(index, with_tool=False)):
                last_turn = index == TURNS - 1
                if last_turn and position == toggle_at:
                    await _toggle(pilot)
                    in_flight = app.state.items[-1].text.strip()
                    if app.state.show_thinking:
                        assert _visible_thinking(app) == [*earlier, in_flight]
                    else:
                        assert _visible_thinking(app) == []
                await stream(event)
                await pilot.pause()
                if last_turn and position >= toggle_at and app.state.show_thinking:
                    # Deltas after the toggle land in the same row: the open
                    # block is never split in two, nor left behind twice.
                    shown = _visible_thinking(app)
                    assert shown[: len(earlier)] == earlier
                    assert len(shown) == TURNS

        assert app.state.show_thinking is not start_shown
        assert _visible_thinking(app) == (every_plan if app.state.show_thinking else [])
        for _ in range(2):
            await _toggle(pilot)
            assert _visible_thinking(app) == (every_plan if app.state.show_thinking else [])


@pytest.mark.anyio
async def test_hiding_thinking_while_the_answer_streams_shows_no_status_line() -> None:
    """The answer is only a buffer until MessageEnd, so the thinking item is
    still last in the state while text is visibly streaming below it."""
    app = TauTuiApp(FakeSession())
    stream = _streamer(app)

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        for event in _turn_events(0, with_tool=False):
            await stream(event)
            if isinstance(event, MessageUpdateEvent) and isinstance(
                event.assistant_message_event, TextDeltaEvent
            ):
                break
        await pilot.pause()
        assert app.state.assistant_buffer

        await _toggle(pilot)
        await _toggle(pilot)
        assert not _notice_shown(app)

        # The full redraw a window shift runs takes the same decision.
        app.query_one("#transcript", TranscriptView)._redraw(scroll_end=False, preserve_window=True)
        await pilot.pause()
        assert not _notice_shown(app)


@pytest.mark.anyio
async def test_thinking_toggle_all_status_line_only_for_the_open_block() -> None:
    """Hidden, only the open block brings up the prompt-border notice;
    earlier hidden blocks show nothing."""
    app = TauTuiApp(FakeSession())
    stream = _streamer(app)

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        await _toggle(pilot)
        events = _turn_events(TURNS - 1, with_tool=False)
        for index in range(TURNS - 1):
            for event in _turn_events(index, with_tool=False):
                await stream(event)
        for event in events[:AFTER_FIRST_THINKING_DELTA]:
            await stream(event)
        await pilot.pause()
        assert _visible_thinking(app) == ["plan 0", "plan 1", "plan"]

        await _toggle(pilot)
        assert _visible_thinking(app) == []
        assert _notice_shown(app)

        await stream(events[AFTER_FIRST_THINKING_DELTA])
        await pilot.pause()
        assert _notice_shown(app)

        await _toggle(pilot)
        assert not _notice_shown(app)
        assert _visible_thinking(app) == ["plan 0", "plan 1", "plan 2"]

        await _toggle(pilot)
        for event in events[AFTER_THINKING:]:
            await stream(event)
            await pilot.pause()
        # Answered and settled: the status line is taken back.
        assert not _notice_shown(app)
        assert _visible_thinking(app) == []
        await _toggle(pilot)
        assert _visible_thinking(app) == ["plan 0", "plan 1", "plan 2"]


@pytest.mark.anyio
@pytest.mark.parametrize("start_shown", [False, True])
async def test_thinking_toggle_all_through_run_prompt(start_shown: bool) -> None:
    """Several agent runs, each several thinking messages with tool calls."""
    session = FakeSession()
    app = TauTuiApp(session)
    every_plan = [f"plan {turn}.{step}" for turn in range(TURNS) for step in range(3)]

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        if start_shown:
            await _toggle(pilot)
        for turn in range(TURNS):
            session.events = tuple(_agentic_run(turn, 3))
            await app._run_prompt(f"prompt {turn}")
            await pilot.pause()

        assert _visible_thinking(app) == (every_plan if start_shown else [])
        for _ in range(3):
            await _toggle(pilot)
            assert _visible_thinking(app) == (every_plan if app.state.show_thinking else [])


@pytest.mark.anyio
@pytest.mark.parametrize("start_shown", [False, True])
async def test_thinking_toggle_all_in_a_paged_window(start_shown: bool) -> None:
    """Past the mounted window, every block in the window toggles."""
    app = TauTuiApp(FakeSession())
    stream = _streamer(app)

    def window_plans() -> list[str]:
        transcript = app.query_one("#transcript", TranscriptView)
        window = app.state.items[transcript._window_start : transcript._window_end]
        return [item.text for item in window if item.role == "thinking"]

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        if start_shown:
            await _toggle(pilot)
        for index in range(30):
            for event in _turn_events(index, with_tool=True):
                await stream(event)
            await pilot.pause()
        assert app.query_one("#transcript", TranscriptView)._window_start > 0
        assert len(window_plans()) > TURNS

        assert _visible_thinking(app) == (window_plans() if start_shown else [])
        for _ in range(2):
            await _toggle(pilot)
            assert _visible_thinking(app) == (window_plans() if app.state.show_thinking else [])


@pytest.mark.anyio
async def test_thinking_toggle_off_and_on_mid_stream_draws_each_paragraph_once() -> None:
    """The row a toggle mounts mid-stream starts with the text so far. The next
    delta must not reach Markdown before that text is drawn, or it is drawn
    twice: Markdown.append reads its parse position before taking its lock."""
    app = TauTuiApp(FakeSession())
    stream = _streamer(app)

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        await _toggle(pilot)
        partial = AssistantMessage()
        for event in [
            MessageEndEvent(message=UserMessage(content="prompt")),
            AgentStartEvent(),
            MessageStartEvent(message=partial),
            _thinking_delta(partial, "one.\n\n"),
            _thinking_delta(partial, "two.\n\n"),
            _thinking_delta(partial, "three"),
        ]:
            await stream(event)
            await pilot.pause()

        # Straight to the actions: a key press would wait for the mount, and
        # a real delta lands sooner.
        app.action_toggle_thinking()
        app.action_toggle_thinking()
        await stream(_thinking_delta(partial, " more\n\nfour"))
        for _ in range(3):
            await pilot.pause()

        assert _rendered_thinking(app) == [["one.", "two.", "three more", "four"]]


@pytest.mark.anyio
@pytest.mark.parametrize("toggles", [1, 2])
@pytest.mark.parametrize("start_shown", [False, True])
async def test_thinking_toggle_while_the_first_row_mounts(toggles: int, start_shown: bool) -> None:
    """A toggle can land while the first delta awaits its row's mount. The
    toggle rebuilds thinking from state, which already holds that delta, so
    the delta must neither crash the stream nor reach a second row."""
    app = TauTuiApp(FakeSession())
    stream = _streamer(app)

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        if start_shown:
            await _toggle(pilot)
        partial = AssistantMessage()
        for event in [
            MessageEndEvent(message=UserMessage(content="prompt")),
            AgentStartEvent(),
            MessageStartEvent(message=partial),
        ]:
            await stream(event)

        first = _thinking_delta(partial, "plan ")
        app.adapter.apply(first)
        mounting = asyncio.create_task(app._apply_streaming_transcript_event(first))
        await asyncio.sleep(0)
        # Shown thinking awaits its row's mount; hidden thinking mounts nothing.
        assert mounting.done() is not start_shown
        for _ in range(toggles):
            app.action_toggle_thinking()
        await mounting
        await pilot.pause()
        assert _visible_thinking(app) == (["plan"] if app.state.show_thinking else [])
        await stream(_thinking_delta(partial, "0"))
        await pilot.pause()
        assert _visible_thinking(app) == (["plan 0"] if app.state.show_thinking else [])
        for event in [
            MessageEndEvent(
                message=AssistantMessage(
                    content=[ThinkingContent(thinking="plan 0"), TextContent(text="answer 0")]
                )
            ),
            AgentEndEvent(),
        ]:
            await stream(event)
            await pilot.pause()

        assert [item.role for item in app.state.items] == ["user", "thinking", "assistant"]
        assert _visible_thinking(app) == (["plan 0"] if app.state.show_thinking else [])
        await _toggle(pilot)
        assert _visible_thinking(app) == (["plan 0"] if app.state.show_thinking else [])


@pytest.mark.anyio
async def test_hidden_thinking_moves_nothing_on_screen() -> None:
    """Hidden thinking holds the answer's row from its first delta.

    The notice sits on a blank placeholder row with a blank row above it.
    While thinking streams nothing moves, and the answer then takes the
    notice's row: the rows above it and the prompt stay where they are."""
    app = TauTuiApp(FakeSession())
    stream = _streamer(app)

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        transcript = app.query_one("#transcript", TranscriptView)
        prompt = app.query_one("#prompt", PromptInput)
        notice = app.query_one("#thinking-notice", Static)
        events = _turn_events(0, with_tool=False)
        for event in events[:AFTER_FIRST_THINKING_DELTA]:
            await stream(event)
        await pilot.pause()
        assert _notice_shown(app)

        def layout() -> tuple[tuple[tuple[int, int], ...], int, int]:
            # Every shown transcript row's place on screen, then the prompt's.
            rows = tuple(
                (child.region.y, child.region.height)
                for child in transcript.children
                if child.display
            )
            return (rows, prompt.region.y, prompt.region.height)

        before = layout()
        notice_y = notice.region.y
        # The user's message, a blank row, then the notice.
        user_row = _rows(app)[-1]
        assert notice_y == user_row.region.bottom + 1
        for event in events[AFTER_FIRST_THINKING_DELTA : AFTER_TEXT_STARTED - 1]:
            await stream(event)
            await pilot.pause()
            assert _notice_shown(app)
            assert layout() == before
        await stream(events[AFTER_TEXT_STARTED - 1])
        await pilot.pause()
        assert not _notice_shown(app)
        assert not [p for p in app.query(ThinkingPlaceholder) if p.display]
        # The answer's first line lands where the notice was, and the rows
        # above it stay put.
        after = layout()
        assert after[0][:-1] == before[0][:-1]
        assert after[0][-1] == before[0][-1] == (notice_y, 1)
        assert after[1:] == before[1:]


def _silent_tool_burst(steps: int, *, answer: bool) -> list[AgentEvent]:
    """One run whose messages each think and call a tool with no text, then
    optionally a last message that thinks and answers."""
    events: list[AgentEvent] = [AgentStartEvent()]
    for step in range(steps + (1 if answer else 0)):
        thinking = f"plan {step}"
        partial = AssistantMessage()
        events += [
            MessageStartEvent(message=partial),
            _thinking_delta(partial, thinking),
            MessageUpdateEvent(
                message=partial,
                assistant_message_event=ThinkingEndEvent(
                    content_index=0, content=thinking, partial=partial
                ),
            ),
        ]
        if step == steps:
            events += [
                _text_delta(partial, "done"),
                MessageEndEvent(
                    message=AssistantMessage(
                        content=[ThinkingContent(thinking=thinking), TextContent(text="done")]
                    )
                ),
            ]
            continue
        call = ToolCall(
            id=f"call-{step}",
            name="bash",
            arguments={"command": "ls", "description": f"Step {step}"},
        )
        events += [
            MessageEndEvent(
                message=AssistantMessage(content=[ThinkingContent(thinking=thinking), call])
            ),
            ToolExecutionStartEvent(tool_call_id=call.id, tool_name=call.name, args=call.arguments),
            ToolExecutionEndEvent(
                tool_call_id=call.id,
                tool_name=call.name,
                result=AgentToolResult(content="ok"),
                is_error=False,
            ),
        ]
    events.append(AgentEndEvent())
    return events


@pytest.mark.anyio
@pytest.mark.parametrize("answer", [False, True])
async def test_hidden_thinking_between_tool_calls_never_moves_the_history_down(
    answer: bool,
) -> None:
    """A tool row that follows another joins it with no blank row above, so
    the placeholder before it holds no blank row either. Were it taller than
    the row that replaces it, each call would push the history up and then
    let it fall back by a row. An answer after the burst brings its own blank
    row, which only pushes the history further up."""
    app = TauTuiApp(FakeSession())
    stream = _streamer(app)
    async with app.run_test(size=(100, 16)) as pilot:
        await pilot.pause()
        # Enough history that the transcript scrolls, so growth moves rows up.
        for turn in range(3):
            for event in _turn_events(turn, with_tool=False):
                await stream(event)
        await stream(MessageEndEvent(message=UserMessage(content="go")))
        await pilot.pause()
        anchor = _rows(app)[-1]
        last_y = anchor.region.y
        for event in _silent_tool_burst(3, answer=answer):
            await stream(event)
            await pilot.pause()
            assert anchor.region.y <= last_y, type(event).__name__
            last_y = anchor.region.y
        roles = [row.item.role for row in _rows(app)]
        assert roles.count("tool") == 3


@pytest.mark.anyio
async def test_a_run_that_ends_while_thinking_drops_the_placeholder() -> None:
    """With no answer or tool row to take its place, the blank row goes when
    the run ends rather than stay as a gap above the prompt."""
    app = TauTuiApp(FakeSession())
    stream = _streamer(app)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        for event in _turn_events(0, with_tool=False)[:AFTER_FIRST_THINKING_DELTA]:
            await stream(event)
        await pilot.pause()
        assert _notice_shown(app)
        await stream(AgentEndEvent())
        app._refresh_chrome()
        await pilot.pause()
        assert not _notice_shown(app)
        assert not [p for p in app.query(ThinkingPlaceholder) if p.display]


@pytest.mark.anyio
async def test_a_row_mounted_after_the_placeholder_retires_it_in_the_same_call() -> None:
    """The row that follows thinking takes the placeholder's place as it is
    inserted. Were the placeholder still shown at the next layout, that frame
    would hold both rows and push the history up."""
    app = TauTuiApp(FakeSession())
    stream = _streamer(app)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        for event in _turn_events(0, with_tool=False)[:AFTER_FIRST_THINKING_DELTA]:
            await stream(event)
        await pilot.pause()
        transcript = app.query_one("#transcript", TranscriptView)
        assert [p for p in app.query(ThinkingPlaceholder) if p.display]
        mounting = transcript.mount(Static("next row"), before=transcript._bottom_boundary)
        assert not [p for p in app.query(ThinkingPlaceholder) if p.display]
        await mounting


@pytest.mark.anyio
@pytest.mark.parametrize("between", ["queued messages", "extension widget"])
async def test_the_notice_covers_nothing_between_the_transcript_and_it(between: str) -> None:
    """Lifted one row, the notice lands on the transcript's placeholder only
    when the transcript ends directly above it. With queued messages or an
    extension widget in between, it keeps its own row above the prompt."""
    session = FakeSession()
    app = TauTuiApp(session)
    stream = _streamer(app)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        for event in _turn_events(0, with_tool=False)[:AFTER_FIRST_THINKING_DELTA]:
            await stream(event)
        await pilot.pause()
        if between == "queued messages":
            session.queued_steering_messages = ("one more thing",)
            occupant = app.query_one("#queued-messages", Static)
        else:
            occupant = Static("extension status")
            await app.query_one("#above-prompt-slot").mount(occupant)
        app._refresh_chrome()
        await pilot.pause()
        notice = app.query_one("#thinking-notice", Static)
        prompt_row = app.query_one("#prompt-row")
        assert str(notice.content) == HIDDEN_THINKING_NOTICE
        assert notice.region.bottom == prompt_row.region.y
        assert not notice.region.overlaps(occupant.region)


@pytest.mark.anyio
async def test_scrolled_back_the_notice_keeps_its_own_row() -> None:
    """Reading earlier output, the placeholder is off screen; the notice stays
    in its row above the prompt instead of covering a line of history."""
    app = TauTuiApp(FakeSession())
    stream = _streamer(app)
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        earlier_turns = 8
        for turn in range(earlier_turns):
            for event in _turn_events(turn, with_tool=False):
                await stream(event)
        for event in _turn_events(earlier_turns, with_tool=False)[:AFTER_FIRST_THINKING_DELTA]:
            await stream(event)
        await pilot.pause()
        notice = app.query_one("#thinking-notice", Static)
        prompt_row = app.query_one("#prompt-row")
        assert _notice_shown(app)
        assert notice.region.bottom < prompt_row.region.y

        transcript = app.query_one("#transcript", TranscriptView)
        assert transcript.max_scroll_y > 0
        transcript.scroll_home(animate=False, immediate=True)
        await pilot.pause()
        assert not transcript.following_output
        app._apply_activity_indicator()
        await pilot.pause()
        assert str(notice.content) == HIDDEN_THINKING_NOTICE
        assert notice.region.bottom == prompt_row.region.y


@pytest.mark.anyio
@pytest.mark.parametrize("show_thinking", [False, True], ids=["hidden", "shown"])
async def test_message_end_keeps_the_streamed_blocks_mounted(show_thinking: bool) -> None:
    """A finished reply keeps the rows it streamed into. Replacing them with new
    rows blanked the reply for the frames the new Markdown took to draw: a
    flicker on every long answer, though the layout came out the same."""
    app = TauTuiApp(FakeSession())
    stream = _streamer(app)
    events = _turn_events(0, with_tool=False)
    end = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, MessageEndEvent) and isinstance(event.message, AssistantMessage)
    )

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        if show_thinking:
            await _toggle(pilot)
        transcript = app.query_one("#transcript", TranscriptView)
        for event in events[:end]:
            await stream(event)
        await pilot.pause()
        streamed = [
            child
            for child in transcript.children
            if isinstance(child, StreamingTranscriptMessageWidget)
        ]
        assert [widget.item.role for widget in streamed] == (
            ["thinking", "assistant"] if show_thinking else ["assistant"]
        )

        for event in events[end:]:
            await stream(event)
        await pilot.pause()

        shown = [item for item in app.state.items if item.role in {"thinking", "assistant"}]
        if not show_thinking:
            shown = [item for item in shown if item.role != "thinking"]
        assert [transcript._item_widgets[id(item)] for item in shown] == streamed
        assert all(widget.parent is transcript for widget in streamed)
        assert all(widget.item is item for widget, item in zip(streamed, shown, strict=True))
        assert transcript._active_message_widgets == []
