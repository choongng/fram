---
title: Keyboard shortcuts
description: Default keys for the Tau TUI, and how to remap them.
---

These are the default keys in the interactive [TUI](/guides/tui/). Run
`/hotkeys` in-session to see them, and remap them in `~/.tau/tui.json` (see
[Configuration](/reference/configuration/#tui-settings)).

## Prompting

| Key | Action |
| --- | --- |
| `Enter` | Accept a highlighted non-file completion; otherwise submit the prompt exactly as typed (including `@` file-reference text) |
| `Ctrl+J` | Insert a newline |
| `Esc` | Cancel the active run |
| `Enter` (while running) | Queue text as steering for the current run |
| `Alt+Enter` | Queue a follow-up that waits until the run would stop |
| `Up` (empty prompt, running) | Edit the most recently queued follow-up |

## Navigation & pickers

| Key | Action |
| --- | --- |
| `Ctrl+K` | Open the command palette |
| `Ctrl+R` | Open the session picker |
| `Tab` | Accept any highlighted completion, including `@` file references |
| `Down` / `Up` | Move through completions |
| `Ctrl+E` (in `/prompts`) | Edit the selected prompt template |
| `Ctrl+S` (while editing a prompt template) | Save and reload resources |

## Models & thinking

| Key | Action |
| --- | --- |
| `Ctrl+P` | Cycle scoped (favorite) models forward |
| `Shift+Ctrl+P` | Cycle scoped (favorite) models backward |
| `Shift+Tab` | Cycle the thinking mode |
| `Ctrl+T` | Toggle display of thinking/reasoning tokens |
| `Ctrl+B` | Show or hide the session sidebar (session-only, like `/sidebar`) |

## Output & session

| Key | Action |
| --- | --- |
| `Ctrl+O` | Toggle exact tool commands and full output (vs. compact previews) |
| `PgUp` / `PgDn` | Page the transcript, even while the prompt has focus |
| `Ctrl+C` | Clear the prompt input |
| `Ctrl+D` | Quit |
| `Ctrl+Z` | Suspend to the shell; `fg` resumes (not supported on Windows) |
| `Ctrl+_` or `Ctrl+/` | Undo in the prompt and text editors (`Ctrl+Y` redoes) |

> [!NOTE] Remapping
> Keys use Textual's syntax (`ctrl+k`, `shift+tab`, `down`, `f2`, …). Tau rejects
> unknown names, empty keys, and duplicate assignments so mistakes fail early. A
> `tui.json` that gives another action a key now used by a default, such as
> `ctrl+b`, `ctrl+j` or `ctrl+z`, is rejected until one of the two is remapped. Any
> key you don't set keeps its default. `Ctrl+B` is also tmux's prefix, where only
> `Ctrl+B Ctrl+B` reaches Tau, so tmux users may prefer another `"toggle_sidebar"`
> key in `~/.tau/tui.json`. To insert newlines with a different key, set
> `insert_newline`:
>
> ```json
> {
>   "keybindings": {
>     "insert_newline": "f2"
>   }
> }
> ```
