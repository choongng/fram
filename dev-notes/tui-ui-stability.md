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

The notice now sits in the prompt's top border, as its `border_title`. The
border is always drawn, so the notice coming and going changes no layout.

- `TauTuiApp._hidden_thinking_in_progress()` decides from state: a run is going,
  thinking is hidden, no answer text is streaming, and the newest item is
  thinking.
- `_apply_activity_indicator` sets the title, along with the spinner and border
  colour it already drew. It runs on chrome refreshes, on each streamed message
  update and after Ctrl+T.
- `TranscriptView` no longer mounts the row. The helpers that tracked and
  removed it (`_hidden_thinking_line`, `_drop_hidden_thinking_placeholder`) are
  gone, and hidden thinking items render nothing in the transcript.

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
place and the prompt's region, streams hidden thinking, and checks that nothing
moves until the answer adds its own row. It fails on the old code, where the
status row appears at the end of the transcript.

`test_completion_list_floats_above_prompt_and_moves_nothing` opens and closes
the list and checks that the transcript, prompt and session-info regions stay
put, and that the list's bottom edge meets the prompt.
`test_completion_list_follows_prompt_after_resize` checks that the list moves
with the prompt when the terminal shrinks.
