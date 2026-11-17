# TUI: keep the layout still while the agent works

Rows that appear and vanish while a turn runs make the screen jump: the
transcript moves under the reader, or the prompt moves under the cursor. The
rule this note sets is that transient UI either has space reserved for it all
the time, or floats over other content. It never inserts rows into the layout.

## What was changed

While the model thinks with thinking tokens hidden, the TUI used to mount a
status row, "Thinking… Press Ctrl+T to show thinking tokens.", at the end of the
transcript, and remove it as soon as an answer or tool call followed. Each turn
therefore added a row and took it back.

The notice now has a row of its own, `#thinking-notice`, directly above the
prompt. The row is always one line tall and blank unless hidden thinking is in
progress, so the notice coming and going changes no layout. Its text starts in
the prompt's text column.

The notice first sat in the prompt's top border, as its `border_title`. Text on
the border read as part of the prompt rather than as a status, so it moved up a
row. The row costs the transcript one line.

In that row the notice sat directly under the transcript's last line, with no
gap. Then, when the answer arrived, its blank row and first line pushed the
history up two rows. Now, while thinking is hidden, the transcript ends with a
`ThinkingPlaceholder`: a blank row with the usual blank row above it, the same
shape as the first line of an answer or tool row (after a tool row it drops that
blank row; see below). The notice is lifted one row, over the placeholder, so it
has a blank row above it. The row that follows takes the placeholder's place,
and the history does not move back down. The scroll happens once, when thinking
starts, rather than when the answer lands.

- `TuiState.hidden_thinking_in_progress` decides from state: a run is going,
  thinking is hidden, no answer text is streaming, and the newest item is
  thinking.
- `_apply_activity_indicator` sets the notice text, along with the spinner and
  border colour it already drew. It runs on chrome refreshes, on each streamed
  message update and after Ctrl+T. It shows the notice only while there is
  something to say, so its lifted row never hides a transcript line. It lifts
  the notice only when the transcript's newest row is directly above it: the
  transcript follows the newest output, and no extension view, extension widget
  or queued-message list sits in between. Otherwise the notice keeps its own row
  rather than cover history or that other content.
- `TranscriptView.sync_thinking_placeholder` mounts the placeholder at the end
  of the transcript when hidden thinking starts, and again after any rebuild
  that left it elsewhere. It drops the placeholder when the run ends or thinking
  is shown.
- `TranscriptView.mount` retires the placeholder when the rows it mounts land
  after it. It hides the placeholder in the same call that inserts the row, so
  the next layout has the new row and not the placeholder, and no frame shows
  both or neither. This cannot be left to `pre_layout`: Textual drops hidden
  widgets from the arrangement before it calls `pre_layout`, so a change of
  `display` made there only takes effect on the following pass.
- When a tool row directly follows another, the two join with no blank row
  between them, and `TranscriptView.pre_layout` gives the placeholder the same
  treatment: after a tool row it drops its blank row too. The placeholder is
  then never taller than the row that replaces it. A tool row takes its place
  exactly, and an answer, which keeps its blank row, only pushes the history
  further up. Before this, each thinking step in a burst of tool calls pushed
  the history up two rows and let it fall back one, so the transcript bounced.

### Autocomplete floats above the prompt

Autocomplete suggestions used to sit in the layout under the prompt. Opening
them shrank the transcript and pushed the prompt up by as many as 18 rows, and
closing them moved it back.

`#autocomplete` is now `overlay: screen` with `position: absolute`, so it takes
no room in the layout. `_place_completions` gives it the prompt row's width,
sizes it to its rendered lines plus the border, and offsets it so its bottom
edge meets the prompt's top edge. It covers the bottom of the transcript and
leaves at least `COMPLETION_MIN_TRANSCRIPT_LINES` rows of it in view.

- `_completion_line_budget()` counts the rows above the prompt. It no longer
  depends on the list's own height, so the old "keep the largest measured
  height" logic and the initial terminal-fraction estimate are gone.
- The prompt moves when the terminal resizes or the prompt grows. The app
  subscribes to the screen's `screen_layout_refresh_signal` and places the list
  again when the prompt row's or main pane's region has changed. Comparing
  regions matters: placing the list updates its content, which triggers another
  layout pass.

## Next steps

- Audit the other rows that change height while a turn runs: `#queued-messages`,
  `#compact-session-info` and the extension slots.

## How it maps to Pi

This is a frontend change only. The agent still emits the same thinking events;
the TUI chooses where to show them.

## Testing

```bash
uv run pytest tests/test_tui_thinking_toggle.py
uv run pytest tests/test_tui_app.py -k "thinking or completion"
```

`test_hidden_thinking_moves_nothing_on_screen` records every transcript row's
place and the prompt's region once hidden thinking starts. It streams the rest
of the thinking and checks that nothing moves, then that the answer's first line
lands on the notice's row with the rows above it and the prompt where they were.
Other tests check that a row mounted after the placeholder retires it in the
same call, that the placeholder goes when a run ends mid-thinking, and that the
notice keeps its own row when the transcript is scrolled back or queued messages
or an extension widget sit above the prompt.

`test_completion_list_floats_above_prompt_and_moves_nothing` opens and closes
the list and checks that the transcript, prompt and session-info regions stay
put, and that the list's bottom edge meets the prompt.
`test_completion_list_follows_prompt_after_resize` checks that the list moves
with the prompt when the terminal shrinks.
