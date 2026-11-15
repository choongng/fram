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

## Next steps

- Autocomplete suggestions mount as a row above the prompt and push the
  transcript up. They should float over the transcript instead
  (`overlay: screen`).
- Audit the other rows that change height while a turn runs: `#queued-messages`,
  `#compact-session-info` and the extension slots.

## How it maps to Pi

This is a frontend change only. The agent still emits the same thinking events;
the TUI chooses where to show them.

## Testing

```bash
uv run pytest tests/test_tui_thinking_toggle.py
uv run pytest tests/test_tui_app.py -k thinking
```

`test_hidden_thinking_moves_nothing_on_screen` records every transcript row's
place and the prompt's region, streams hidden thinking, and checks that nothing
moves until the answer adds its own row. It fails on the old code, where the
status row appears at the end of the transcript.
