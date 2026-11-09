# Tool result buffers

## What was added

`src/tau_coding/tool_buffers.py` keeps oversized tool output out of the
transcript. Every result passes through `buffer_tool_result` from the session's
`after_tool_call` hook:

- A session with a file saves each result but a successful `read`, small ones
  too, as `<session>.outputs/<tool>-<call-id>.txt`.
- A result over the inline limit is replaced by a window of it and a marker
  naming the saved file. The window is the tail by default, and the head for
  `read`.
- A session with no file saves only oversized results, under a content hash in a
  per-user temp directory.

The inline limit is 1/50 of the context window, clamped to 1,000-4,000 tokens,
at the fixed 4 characters a token. `toolResultInlineTokens` in `settings.json`
or `TAU_TOOL_RESULT_INLINE_TOKENS` replaces the computed value.

`bash` no longer truncates its own output or writes a temp log. One mechanism
now covers every tool, including a command the user runs from the prompt.

## Why

A single large `git diff` or test log can take a real share of a small context
window, and it stays there until compaction. A saved result stays searchable
after it leaves context.

The marker names a file path, so a model pages through a result with `read` and
`grep` and needs no new tool. A failed save never fails the tool call; the
marker says what went wrong instead.

## How it maps to Pi

Pi truncates in each tool and points at a temp file. Tau keeps that shape for
the model but moves it into one place in the coding layer. `tau_agent` only
provides the `after_tool_call` hook and knows nothing about buffers, session
files or limits.

## How to test

```bash
uv run pytest tests/test_tool_buffers.py tests/test_tool_output_files.py -q
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
from in the source file. A read is not saved: its file is already on disk, and a
copy would carry a second set of row prefixes. An outline of the file's
definitions and headings follows, with line numbers a read offset can use, so a
model can jump to a section it would otherwise page towards and may never reach.

Pi's read tool pages with `offset` and `limit` too. The row format, cut note and
outline are Tau additions, and they stay in the coding layer.

```bash
uv run pytest tests/test_read_anchors.py tests/test_reference_outline.py -q
```
