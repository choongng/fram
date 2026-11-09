---
title: Built-in tools
description: The read, write, edit, and bash tools the agent uses to work in your project.
---

Tools are the actions the agent can take in your working directory. The model
decides when to call them; Tau executes them and streams the results back. Tau
ships four built-in coding tools: `read`, `write`, `edit`, and `bash`.

All paths are resolved against the session's working directory (`--cwd`, or the
directory you launched Tau from).

> [!NOTE]
> This page documents tool *behavior* — what the model can do on your machine. To
> build a frontend or register your own tools, see
> [Building a custom frontend](/internals/custom-frontend/).

## `read`

Reads a file from disk.

```json
{ "path": "README.md", "offset": 1, "limit": 40 }
```

| Argument | Required | Type | Description |
| --- | --- | --- | --- |
| `path` | yes | string | File to read (relative to `cwd`). |
| `offset` | no | integer | 1-indexed start line (`0` = start of file). |
| `limit` | no | integer | Maximum number of lines to return. |

For text files, `read` shows each line as `N#hhh| text`, where `N` is its line
number and `hhh` a short hash; a line over 180 characters continues on `N.2| `
rows. These prefixes are not file content, and `edit` accepts old text copied
with them. `read` applies `offset`/`limit` to file lines and truncates to at
most 2,000 lines or 50 KB (whichever comes first), appending a hint like
`[42 more lines in file. Use offset=101 to continue.]`. In a session, a result
over the inline limit in [Context](../guides/context.md) is cut again between
lines, with a `[Read cut: ...]` note naming the offset to continue from and an
outline of the file's definitions or headings. Supported images (JPEG, static
PNG, GIF, WebP, and BMP) are detected from their file content and sent to
vision-capable models as image attachments. BMP files are converted to PNG. Tau
validates images and, when necessary, resizes them without upscaling or changing
their aspect ratio. The processed attachment is limited to 2,000 pixels on
either side and 5 MB. Processing also rejects source files above 50 MB or 40
million pixels. Animated PNG and JPEG XL inputs receive explicit
unsupported-format notices. If decoding, conversion, or resizing cannot produce
a safe attachment, Tau returns a clear omission notice.

When the active model does not accept images, `read` returns an explicit
text-only notice that says the image contents are unavailable and recommends
switching to a vision-capable model. It does not attach or process the image.
Provider serialization applies the same defensive downgrade to image blocks from
older sessions or other tools. This avoids invalid requests and reduces pressure
on text-only models to invent a visual description.

The Textual TUI shows this notice, but does not render images inline; native
terminal-image rendering remains outside the `read` tool's provider-neutral
contract.

Fails when `path` is missing/invalid, the file doesn't exist, the path is a
directory, `offset` is past the end, or the file is neither UTF-8 text nor a
supported image.

## `write`

Creates or overwrites a complete UTF-8 text file.

```json
{ "path": "src/example.py", "content": "print('hello')\n" }
```

| Argument | Required | Type | Description |
| --- | --- | --- | --- |
| `path` | yes | string | File to write (relative to `cwd`). |
| `content` | yes | string | Complete file contents. |

Creates missing parent directories and overwrites any existing file. Writes to
the same path are serialized within a process, so concurrent `write`/`edit` calls
on one file don't interleave.

## `edit`

Applies exact text replacements to one file.

```json
{
  "path": "src/example.py",
  "edits": [
    { "oldText": "print('hello')", "newText": "print('hello, Tau')" }
  ]
}
```

| Argument | Required | Type | Description |
| --- | --- | --- | --- |
| `path` | yes | string | File to edit (relative to `cwd`). |
| `edits` | yes | array | One or more `{oldText, newText}` replacements. |

If every edit names the same `path` and the top-level `path` is missing, that
path is used.

Each `oldText` must be non-empty, match exactly (whitespace included), appear
**exactly once**, and not overlap another edit. All edits validate before
anything is written — if any fails, the file is left unchanged. Line endings are
normalized for matching and the original dominant ending is restored. Successful
results include a diff, a unified patch, and the first changed line number.

## `bash`

Runs a shell command in the working directory.

```json
{ "command": "pytest -q", "timeout": 30 }
```

| Argument | Required | Type | Description |
| --- | --- | --- | --- |
| `command` | yes | string | Shell command to run. |
| `description` | no | string | Short summary of the command's purpose, shown as the call's label. |
| `timeout` | no | number | Max runtime in seconds (> 0). Default `600`. |

Combines stdout and stderr and succeeds on exit code `0`. Large output is
reduced to its **tail**, and the full text is saved to a file named in the
result (see [Large tool results](/guides/context/#large-tool-results)). On
POSIX, a timeout kills the whole process group.

A call without `timeout` is stopped after 600 seconds, and the result says the
default was hit so the model can retry with a larger value. Set
`TAU_BASH_TIMEOUT_SECONDS` to change that default for a session (an empty,
non-numeric or non-positive value keeps 600). Commands you type yourself with
`!` have no default timeout.

## `history`

Reads earlier entries of the current session branch from the session file,
including entries a compaction removed from context.

| Parameter | Description |
|---|---|
| `refs` | Entries by reference: `#12`, `#12-#40` or `12,15`. Returned in full, up to the size shown inline for any tool result (at most 20,000 characters); the answer names the first entry left out. |
| `query` | Case-insensitive search over entry text and tool arguments. |
| `tool` | Only calls and results of this tool. Combines with `query`. |

`#N` is the entry's position in the session file. References appear in the index
at the start of a compaction summary and in `history` results. A session built
with an explicit tool list does not get this tool.

## Choosing the right tool

- **`read`** — inspect files (instead of `cat`/`sed`).
- **`write`** — new files or complete rewrites.
- **`edit`** — precise changes to an existing file.
- **`bash`** — tests, linters, searches, project inspection.
