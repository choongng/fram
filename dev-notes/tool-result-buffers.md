# Tool result buffers

## What was added

`src/tau_coding/tool_buffers.py` keeps oversized tool output out of the
transcript. Every result passes through `buffer_tool_result` from the session's
`after_tool_call` hook:

- A session with a file saves each result but a successful `read`, small ones
  too, as `<session>.outputs/<tool>-<call-id>.txt`.
- A result over the inline limit is replaced by a window of it and a marker
  naming the saved file. The tool declares which part the window keeps; see
  [Which part is kept](#which-part-is-kept).
- A session with no file saves only oversized results, under a content hash in a
  per-user temp directory.

The inline limit is 1/50 of the context window, clamped to 2,000-4,000 tokens,
at the fixed 4 characters a token: 8,000 characters on a model of 32K to 100K
tokens. Under 32K the floor is 1/16 of the window and never below 1,000 tokens
(1,000 at 8K or 16K, 1,500 at 24K), so one result cannot take a quarter of a
small window. The window of a cut result is half the limit.
`toolResultInlineTokens` in `settings.json` or `TAU_TOOL_RESULT_INLINE_TOKENS`
replaces the computed value.

`bash` no longer truncates its own output or writes a temp log. One mechanism
now covers every tool, including a command the user runs from the prompt.

## Why

A single large `git diff` or test log can take a real share of a small context
window, and it stays there until compaction. A saved result stays searchable
after it leaves context.

The marker names a file path, so a model pages through a result with `read` and
`grep` and needs no new tool. A failed save never fails the tool call; the
marker says what went wrong instead.

## Which part is kept

A tool says which part of an oversized result is worth keeping, in
`AgentTool.output_window`:

| Value | Kept | For |
|---|---|---|
| `"ends"` (default) | about 40% of the window from the start, 60% from the end | command output, logs, anything not known to be one of the others |
| `"head"` | the start, cut between file lines | `read` |
| `"tail"` | the last 60 lines | output where only the end matters |
| `"whole"` | all of it, up to 1/8 of the context window | a finished report, such as a sub-agent's |

`tau_agent` only carries the field. `buffer_tool_result` takes the value, and
the session looks it up from the tool a call names; a call to a tool the session
does not have gets `"ends"`. No tool name is special-cased in the buffer code.

Every cut result starts with one line saying it was cut and where the marker is:

```text
[Output cut: 145 lines, middle omitted, see marker below.]
```

An `"ends"` cut puts its marker where the middle was. Each end holds whole
lines, up to its share of the window's characters and of its 60 lines; an end
whose first line alone is too long holds what fits of that line.

```text
[… lines 25–88 omitted (3,100 chars); full output: <path>. Page it with read offset/limit or grep it. Omitted sections: Highlights (L30), Priorities (L41).]
```

The marker lists the headings or definitions the cut took out, each with its
line in the saved output, so a `read` offset can go straight to one. The list
has the same budget as the outline of a cut read: 80 entries and a quarter of
the window's characters. When the save failed the marker says why in place of
the path. A report that quotes a line of code is still a report: where the cut
text has a `##` or deeper heading, its headings are listed and its `def` and
`class` lines are not. A code fence opened before the cut still covers the lines
inside it.

`"whole"` exists for results that are useless with a piece missing. A result up
to 1/8 of the context window (16,000 characters on a 32K-token model) is shown
uncut, even though it is over the inline limit. A larger one is cut as `"ends"`
is, so no single result can take over the window. A `"whole"` result is saved
like any other.

### Why not the tail

The window used to be the last 60 lines for every tool but `read`, with the note
that the result was cut as its last line. A parent session on a 32K-token model
received a sub-agent's 4,700-character status report that way: the last 2,000
characters, headed by nothing. It read the tail as the whole report, never
opened the saved file, and wrote its own report from an example in a skill
instead. Three things changed in answer:

- The first line of a cut result says it was cut, so a reader cannot take a
  window for the whole.
- The default window keeps the start too. A report states its conclusion first
  and a command's output ends with its result; keeping both ends serves both.
- A tool whose result is a finished report asks for it whole, and the inline
  floor rose from 1,000 to 2,000 tokens on a window of 32K or more, so a short
  report fits on a small model in any case.

Windowing happens once, when the result comes back, and the transcript stores
the windowed text. A later request replays those bytes unchanged, so a server's
prompt cache still matches (`tests/test_tool_output_window.py`).

## How it maps to Pi

Pi truncates in each tool and points at a temp file. Tau keeps that shape for
the model but moves it into one place in the coding layer. `tau_agent` only
provides the `after_tool_call` hook and knows nothing about buffers, session
files or limits.

## How to test

```bash
uv run pytest tests/test_tool_buffers.py tests/test_tool_output_files.py \
  tests/test_tool_output_window.py -q
```

## Reading a cut file

`read` shows each line as an anchored row, defined in
`src/tau_coding/line_anchors.py`:

```text
12#3fa| the line's text
48#0b7| a line longer than 180 characters starts here
48.2|  and continues on numbered rows
```

`12` is the file line and `3fa` is a short hash of the line. `offset`, `limit`
and the 2,000-line cap all count file lines, not rows. A file's final newline no
longer counts as an extra empty line. `edit` retries an `oldText` copied with
its prefixes on, so a copied row still matches.

An oversized `read` is windowed by size, because a line can be a few characters
of code or a few thousand of converted prose. The cut falls between file lines
and the `[Read cut: ...]` note names the lines shown and the offset to continue
from in the source file. Like any cut result it starts with the
`[Output cut: ...]` line, here saying the end was omitted. A read is not saved:
its file is already on disk, and a copy would carry a second set of row
prefixes. An outline of the file's definitions and headings follows, with line
numbers a read offset can use, so a model can jump to a section it would
otherwise page towards and may never reach.

Pi's read tool pages with `offset` and `limit` too. The row format, cut note and
outline are Tau additions, and they stay in the coding layer.

```bash
uv run pytest tests/test_read_anchors.py tests/test_reference_outline.py -q
```
