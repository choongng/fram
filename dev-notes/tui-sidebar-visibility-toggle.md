# Session-only TUI sidebar visibility

Tau's interactive TUI now exposes `/sidebar` in the slash-command palette. The
command toggles the detailed session sidebar without changing `~/.tau/tui.json`
or the configured `sidebar_position`.

## Behavior

- Configured `left` and `right` positions remain unchanged while the sidebar is
  hidden and shown.
- Responsive hiding still applies until the user explicitly changes visibility.
- Once the user hides the sidebar, resizing the terminal does not unexpectedly
  restore it.
- A configured `off` sidebar can be shown for the current session at the
  existing default right position. Restarting Tau honors the saved `off` value.

The visibility override belongs to `TauTuiApp`, alongside other frontend-only
state. It is intentionally not part of `TuiSettings`, so theme persistence and
other durable TUI settings cannot accidentally serialize it.

## Hidden at startup

The sidebar now starts hidden, so the transcript gets the full width. The
durable `sidebar_visible` setting (default `false`) shows it at startup;
`_update_responsive_layout` treats `false` like `off` for the initial layout,
and the session override above still wins once `/sidebar` or `Ctrl+B` is used.

`sidebar_position: "off"` already hid the sidebar, but it also makes
`supports_sidebar` false, which turns off extension sidebar sections. A separate
setting keeps those sections working while the sidebar is hidden, so showing it
reveals them.

## Tests

Deterministic Textual pilot tests cover left/right toggling, responsive resize
behavior, configured-off temporary showing, and the no-write guarantee. Run the
focused suite with:

```bash
uv run pytest tests/test_tui_app.py tests/test_commands.py
```
