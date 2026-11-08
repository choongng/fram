"""Keep oversized tool results out of the transcript.

Every tool result passes through `buffer_tool_result` before it becomes a
transcript message. A result under the inline limit is returned unchanged. A
larger one is written whole to a buffer file, and the transcript gets a window
of it (the tail by default, the head for `read`) plus a marker that names the
file. The model pages the rest in with tools it already knows: `read` with
`offset`/`limit`, or `grep` through `bash`.

A session saves every tool result, small ones too, as
`<session>.outputs/<tool>-<call-id>.txt` (`output_file_name`), so old outputs
stay greppable after they leave context. Output with no call to name it by (a
user's terminal command) is named by a hash of its content, so the same output
always gets the same path. See dev-notes/tool-result-buffers.md.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from tau_agent.messages import ImageContent, TextContent
from tau_agent.tools import AgentToolResult
from tau_coding.context_window import CHARS_PER_TOKEN

#: The inline limit is this fraction of the model's context window...
INLINE_WINDOW_DIVISOR = 50
#: ...clamped to this many tokens, so a small window still shows a useful
#: result and a large one does not take in a whole diff. At 4,000 tokens a 200K
#: window inlines up to 16,000 chars, so a 26,000-char `git diff` is buffered.
#: See dev-notes/tool-result-buffers.md.
MIN_INLINE_TOKENS = 1_000
MAX_INLINE_TOKENS = 4_000
#: Set to a token count, this replaces the computed inline limit for the run.
INLINE_TOKENS_ENV = "TAU_TOOL_RESULT_INLINE_TOKENS"
#: Lines of an oversized result kept in the transcript.
WINDOW_LINES = 60
#: Tools whose oversized output is windowed from the head, not the tail. Test
#: output, tracebacks and build logs end with the part that matters; a file
#: read starts with it.
HEAD_WINDOW_TOOLS = frozenset({"read"})


@dataclass(frozen=True, slots=True)
class BufferLimits:
    """How much of a tool result stays inline, in characters."""

    inline_chars: int
    window_chars: int
    window_lines: int = WINDOW_LINES

    @classmethod
    def for_context_window(
        cls, context_window_tokens: int, inline_tokens: int | None = None
    ) -> BufferLimits:
        """The limits for a context window.

        `INLINE_TOKENS_ENV`, then `inline_tokens` (the settings value), replaces
        the computed limit.
        """
        tokens = _env_inline_tokens() or inline_tokens
        if tokens is None:
            tokens = min(
                MAX_INLINE_TOKENS,
                max(MIN_INLINE_TOKENS, context_window_tokens // INLINE_WINDOW_DIVISOR),
            )
        # A fixed 4 characters a token: the served model's measured ratio
        # describes its own output, not the tool output it reads.
        inline_chars = tokens * CHARS_PER_TOKEN
        return cls(inline_chars=inline_chars, window_chars=inline_chars // 2)


def _env_inline_tokens() -> int | None:
    try:
        tokens = int(os.environ.get(INLINE_TOKENS_ENV, ""))
    except ValueError:
        return None
    return tokens if tokens > 0 else None


def default_buffer_dir() -> Path:
    """Where buffers go when the session has no file of its own.

    Per user, because outputs can hold secrets and the temp directory is shared.
    """
    return Path(tempfile.gettempdir()) / f"tau-buffers-{_user_tag()}"


def session_outputs_dir(session_path: Path) -> Path:
    """The directory holding a session file's tool outputs, beside the file."""
    return session_path.with_name(f"{session_path.stem}.outputs")


_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def output_file_name(tool_name: str, tool_call_id: str) -> str:
    """The file a tool call's output is saved as, derived from the call alone.

    Code that holds a recorded tool result (the history tool, the compaction
    index) finds its file from the result's tool name and call id, with no path
    stored anywhere.
    """
    tool = _UNSAFE_NAME_CHARS.sub("_", tool_name).strip("._") or "tool"
    call = _UNSAFE_NAME_CHARS.sub("_", tool_call_id).strip("._") or "call"
    return f"{tool}-{call}.txt"


def _user_tag() -> str:
    getuid = getattr(os, "getuid", None)
    return str(getuid()) if getuid is not None else "user"


def buffer_tool_result(
    result: AgentToolResult,
    *,
    tool_name: str,
    buffer_dir: Path,
    limits: BufferLimits,
    output_name: str | None = None,
) -> AgentToolResult:
    """Return `result` unchanged if it fits inline, else a window plus a buffer path.

    With `output_name`, the full text is saved as that file in `buffer_dir`
    whatever its size, so every output stays on disk; without it, only an
    oversized one is saved, under a content hash. A failed save never fails the
    tool call: an inline result is returned as is.
    """
    texts = [block.text for block in result.content if isinstance(block, TextContent)]
    text = "\n".join(texts)
    location: str | None = None
    failure = ""
    if output_name is not None:
        try:
            location = str(_write_output(buffer_dir / output_name, text))
        except OSError as exc:
            failure = f"{type(exc).__name__}: {exc}"
    if len(text) <= limits.inline_chars:
        return result
    if output_name is None:
        try:
            location = str(_write_buffer(buffer_dir, text))
        except OSError as exc:
            failure = f"{type(exc).__name__}: {exc}"
    head = tool_name in HEAD_WINDOW_TOOLS
    shown, shown_lines = _window(text, head=head, limits=limits, max_lines=limits.window_lines)
    total_lines = text.count("\n") + 1
    where = "first" if head else "last"
    summary = (
        f"[Output truncated: the {where} {shown_lines:,} of {total_lines:,} lines "
        f"({len(shown):,} of {len(text):,} chars) are shown."
    )
    if location is not None:
        marker = (
            f"{summary} Full output: {location} — page it with read (offset/limit) or grep it.]"
        )
    else:
        marker = f"{summary} The full output could not be saved ({failure}).]"
    images = [block for block in result.content if isinstance(block, ImageContent)]
    return result.model_copy(update={"content": [TextContent(text=f"{shown}\n{marker}"), *images]})


def _window(
    text: str, *, head: bool, limits: BufferLimits, max_lines: int | None
) -> tuple[str, int]:
    """The kept part of `text` and how many of its lines it shows, whole or in part."""
    lines = text.split("\n")
    if max_lines is not None:
        lines = lines[:max_lines] if head else lines[-max_lines:]
    kept = lines
    shown = "\n".join(kept)
    if len(shown) > limits.window_chars:
        # A few very long lines (minified JSON, an event stream) would defeat a
        # line window on their own, so the character cap always applies too.
        shown = shown[: limits.window_chars] if head else shown[-limits.window_chars :]
    return shown, shown.count("\n") + 1


def _write_buffer(buffer_dir: Path, text: str) -> Path:
    data = text.encode("utf-8", "replace")
    path = buffer_dir / f"{sha256(data).hexdigest()[:16]}.txt"
    if path.exists():
        return path
    return _write_output(path, text)


def _write_output(path: Path, text: str) -> Path:
    """Write `text` to `path` atomically, in a directory only this user can read."""
    data = text.encode("utf-8", "replace")
    buffer_dir = path.parent
    buffer_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=buffer_dir, prefix=".buffer-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path
