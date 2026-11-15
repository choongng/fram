# TUI: keep the terminal's background

The TUI used to paint every cell with its theme's `screen_background`, so
`tau-dark` turned a tinted or translucent terminal solid black and `tau-light`
turned it white. Themes can now leave the terminal's own background showing.

## What was added

- A `terminal_background` theme field. `tau-dark` and `tau-light` set it;
  `high-contrast` does not, since it is about exact colors.
- `terminal_default_color(theme, color)` in `tui/themes/__init__.py`. When the
  flag is set, it returns `ansi_default`, Textual's name for the terminal's
  default color, for any color equal to `screen_background`.
  `theme_css_variables` passes every background variable through it, so the
  screen, transcript, sidebar and chrome show the terminal while the prompt,
  which has its own color, stays tinted.
- Parsing drops `on <screen_background>` from role bodies, so transcript rows do
  not repaint black over the terminal.
- `tui/terminal_background.py`, a Textual line filter and a modal backdrop, both
  installed by `TauTuiApp`.

## Why the filter is needed

Textual converts every color without an RGB value to a fixed truecolor before
writing it, using its ANSI theme (`textual.filter.ANSIToTruecolor`). The
terminal's default background is one of those colors, so `ansi_default` would
still arrive as black. `KeepTerminalBackground` subclasses that filter: it
converts as before, then puts the default background back. Two details:

- Dimmed text is blended toward the background, which needs an RGB value. When
  the background is the default, the filter blends toward the ANSI theme's color
  for it.
- It replaces the background by adding a style rather than rebuilding one.
  Rich's `without_color` drops a style's meta, which carries Textual's mouse
  handlers, and clicks stopped working.

Textual rebuilds its filter when the theme changes, so `TauTuiApp` overrides
`_refresh_truecolor_filter` to install the subclass again.

## Modal backdrops

A modal screen, such as the command palette, tints the screen behind it by
blending each cell's colors with its own translucent background. Textual counts
the default background as black in that blend, so `tau-light`'s backdrop came
out grey. `KeepDefaultBackgroundScreen` tints a default-background cell's text
and leaves its background to the terminal. `Screen.render` builds the backdrop
by the name `textual.screen.BackgroundScreen`, so `install_modal_backdrop`
replaces that name, for the whole process.

The project-trust screen tinted with `$tau-screen-background 60%`, which is now
`ansi_default` at 60%: a black overlay. It uses Textual's `$background`, which
stays a hex color.

## Rejected routes

Two other routes were rejected. `Theme(ansi=True)` makes Textual generate a
whole ANSI palette and drops the variables Tau's CSS uses. `App.ansi_color`
keeps the default background but also turns on Textual's `:ansi` widget styles,
which restyle the prompt, Markdown and toasts.

## What stays a real color

The theme keeps `screen_background` as a hex color. Tau still needs it where it
computes with the background: the tints blended from it (`transcript_background`
blends in `widgets.py`), the prompt spinner's fade, Textual's own derived shades
(the `background`, `surface` and `panel` slots of the Textual theme), and HTML
session exports. For those it is the color Tau assumes the terminal has.

Dimmed text is the exception. The filter blends it toward the ANSI theme's
default background, not toward `screen_background`, because that is the color
Textual's filter itself uses. The two can differ slightly.

## How it maps to Pi

This is a frontend change only. The theme flag and the filter live in
`tau_coding.tui`; the agent layers are untouched.

## Testing

```bash
uv run pytest tests/test_tui_terminal_background.py tests/test_tui_themes.py
uv run pytest tests/test_tui_app.py -k "terminal or backdrop"
```

`test_tui_app_leaves_the_terminal_background_showing` renders the app, runs a
transcript row through the app's filters and checks that only the default
background reaches the terminal, after a theme switch too.
`test_tui_app_modal_backdrop_keeps_the_terminal_background` opens the command
palette and checks the backdrop row the same way, in both themes.
