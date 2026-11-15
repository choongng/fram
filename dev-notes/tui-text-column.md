# One text column for the prompt and the transcript

The TUI lines up typed text and transcript text in one column, column 2 of the
main pane, so the eye does not shift between what was typed and what came back.

## Layout

- `#prompt-row` has no margin and carries the prompt's top and bottom rules, so
  they run the full width of the main pane, across the logo column too.
  `_apply_activity_indicator` recolours the rules on the row while a turn runs,
  and the row takes the `-shell-mode` class with the prompt for shell commands.
- `#prompt-prefix` (the τ logo, or the spinner during a turn) is two columns
  wide and sits in column 0 on the prompt's first line. Column 1 is blank.
- `#prompt` has no left padding, so typed text starts in column 2. Its
  `max-height` is the six text lines alone, since the rules moved to the row.
- `#thinking-notice` above the prompt pads its text into column 2 as well.
- Bordered transcript rows (user, tool, status) have no left margin: the pipe is
  in column 0, under the logo, and one padding column follows it, so the text
  starts in column 2. Borderless rows (assistant, thinking) keep one margin
  column and one padding column, which also puts their text in column 2.

## User rows

User rows used to paint the prompt background (`#101419` in `tau-dark`) behind
the message. They now paint none, so the terminal's own background shows, as it
does for every other row. In `tau-dark` the user body is `#d8dee9 on #000000`;
`#000000` is the screen background, which a `terminal_background` theme maps to
the terminal default.

## Testing

`test_tui_prompt_grows_to_six_lines_then_scrolls` checks the row's height with
the rules, and the activity-indicator tests read the rule colours from
`#prompt-row`. `test_tau_dark_theme_uses_aqua_as_its_shared_accent` checks that
the user body has no background.
