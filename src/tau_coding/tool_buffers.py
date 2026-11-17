"""Keep oversized tool results out of the transcript.

Every tool result passes through `buffer_tool_result` before it becomes a
transcript message. A result under the inline limit is returned unchanged. A
larger one is written whole to a buffer file, and the transcript gets a window
of it plus a marker that names the file. Which part the window keeps is the
tool's own declaration, `AgentTool.output_window`: both ends by default, with
the marker where the middle was; the head for `read`; and all of it, up to a
ceiling, for a tool that says its result is useless with a piece missing. A cut
result starts with a line saying it was cut. The model pages the rest in with
tools it already knows: `read` with `offset`/`limit`, or `grep` through `bash`.

A session saves every tool result but a successful read's, small ones too, as
`<session>.outputs/<tool>-<call-id>.txt` (`output_file_name`), so old outputs
stay greppable after they leave context. A read is left out because its file is
already on disk. Output with no call to name it by (a
user's terminal command) is named by a hash of its content, so the same output
always gets the same path. See dev-notes/tool-result-buffers.md.
"""

from __future__ import annotations

import os
import re
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from itertools import count
from pathlib import Path

from tau_agent.messages import ImageContent, TextContent
from tau_agent.tools import AgentToolResult, ToolOutputWindow
from tau_coding.context_window import CHARS_PER_TOKEN
from tau_coding.line_anchors import row_line_number, strip_row_prefixes

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
#: A tool that declares `output_window="whole"` is shown whole up to this
#: fraction of the context window, whatever the inline limit is. Past it the
#: result is cut like any other, so one result cannot take the window over.
WHOLE_WINDOW_DIVISOR = 8
#: Lines of an oversized result kept in the transcript, for every window but
#: the head: a head window is cut by size alone (`window_chars`), since a
#: file's lines may be a few characters (code) or a few thousand (a paper
#: converted to text), so a line count says nothing about how much a window
#: holds.
WINDOW_LINES = 60
#: The share of the window, in characters and in lines, that a cut keeping both
#: ends gives the start. The end gets the rest: a command's output, a test run
#: and a report all tend to close with their result.
ENDS_HEAD_PERCENT = 40
#: A head-windowed result is followed by an outline of its def/class lines and
#: document headings, so the model can jump to one with read offset/limit
#: instead of paging. A result cut in the middle names the ones the cut took
#: out. Either list is capped at this many entries and a quarter of the
#: window's chars.
OUTLINE_MAX_ENTRIES = 80
OUTLINE_WINDOW_DIVISOR = 4
#: A numbered section's title is short; a longer line is a numbered sentence.
SECTION_TITLE_MAX_CHARS = 60
#: An optional line-number prefix (`17\t`, `17: `, `17|`, read's `17#3fa| `).
_PREFIX = r"^(?:\s*(?P<number>\d+)(?:#[0-9a-f]+\| ?|\t|[:|\u2502\u2192] ?))?"
_DEFINITION = re.compile(
    _PREFIX + r"(?P<indent>[ \t]*)(?P<kind>async\s+def|def|class)\s+(?P<name>\w+)"
)
#: A Markdown ATX heading: `#`..`######`, a space, then its text.
_MARKDOWN_HEADING = re.compile(
    _PREFIX + r" {0,3}(?P<hashes>#{1,6})[ \t]+(?P<title>[^#\s].*?)[ \t#]*$"
)
#: A numbered section line of a plain-text paper: `5`, `5.3` or `5.3.` then a title.
_NUMBERED_SECTION = re.compile(
    _PREFIX + r"[ \t]*(?P<section>\d{1,2}(?:\.\d{1,2})*)\.?[ \t]+(?P<title>[A-Z].*?)[ \t]*$"
)
#: A Markdown code fence; a heading-like line inside one is code.
_FENCE = re.compile(_PREFIX + r" {0,3}(?:```|~~~)")
#: A continuation row of a wrapped read line (`17.2| ...`): not a line start.
_CONTINUATION_ROW = re.compile(r"^\s*\d+\.\d+\| ")
#: The note the read tool appends when it stops at a limit; not a file line.
_READ_TOOL_NOTE = re.compile(r"\n\n\[(?:Showing lines |\d+ more lines in file\. )[^\n]*\]\Z")


@dataclass(frozen=True, slots=True)
class BufferLimits:
    """How much of a tool result stays inline, in characters."""

    inline_chars: int
    window_chars: int
    window_lines: int = WINDOW_LINES
    #: The most a "whole" result may hold and still be shown whole; never less
    #: than `inline_chars` in effect (see `whole_limit`).
    whole_chars: int = 0

    @property
    def whole_limit(self) -> int:
        """The size up to which a tool that asks for its result whole gets it."""
        return max(self.whole_chars, self.inline_chars)

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
        return cls(
            inline_chars=inline_chars,
            window_chars=inline_chars // 2,
            whole_chars=context_window_tokens * CHARS_PER_TOKEN // WHOLE_WINDOW_DIVISOR,
        )


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
    """The file a tool call's output is saved as, derived from the call alone."""
    tool = _UNSAFE_NAME_CHARS.sub("_", tool_name).strip("._") or "tool"
    call = _UNSAFE_NAME_CHARS.sub("_", tool_call_id).strip("._") or "call"
    return f"{tool}-{call}.txt"


def _user_tag() -> str:
    getuid = getattr(os, "getuid", None)
    return str(getuid()) if getuid is not None else "user"


def buffer_tool_result(
    result: AgentToolResult,
    *,
    output_window: ToolOutputWindow = "ends",
    buffer_dir: Path,
    limits: BufferLimits,
    first_line: int = 1,
    source_path: str | None = None,
    output_name: str | None = None,
) -> AgentToolResult:
    """Return `result` unchanged if it fits inline, else a window plus a buffer path.

    `output_window` is the part of an oversized result its tool asked to keep
    (`AgentTool.output_window`). "whole" keeps all of a result up to
    `limits.whole_limit`; a larger one is cut as "ends" is.

    `first_line` is the file line the text starts at (a read's offset), so the
    marker and outline of a head-windowed read name the file's own line numbers.
    A read is cut by characters only, never by `window_lines`, and between file
    lines where it can. `source_path` is the file a read came from: nothing is
    saved, because the file is already on disk, and the marker sends the model
    back there.

    With `output_name`, the full text is saved as that file in `buffer_dir`
    whatever its size, so every output stays on disk; without it, only an
    oversized one is saved, under a content hash. A failed save never fails the
    tool call: an inline result is returned as is.
    """
    texts = [block.text for block in result.content if isinstance(block, TextContent)]
    text = "\n".join(texts)
    location: str | None = None
    failure = ""
    if output_name is not None and source_path is None:
        try:
            location = str(_write_output(_unused_path(buffer_dir / output_name), text))
        except OSError as exc:
            failure = f"{type(exc).__name__}: {exc}"
    if len(text) <= limits.inline_chars:
        return result
    if output_window == "whole":
        if len(text) <= limits.whole_limit:
            return result
        output_window = "ends"
    if output_name is None and source_path is None:
        try:
            location = str(_write_buffer(buffer_dir, text))
        except OSError as exc:
            failure = f"{type(exc).__name__}: {exc}"
    # A newline that ends the output ends its last line; it does not start one.
    total_lines = text.count("\n") + (0 if text.endswith("\n") else 1)
    if output_window == "head":
        shown = _read_window(text, limits.window_chars)
        marker, total_lines = _read_marker(
            text,
            shown,
            first_line=first_line,
            source_path=source_path,
            location=location,
            failure=failure,
        )
        outline = _outline(
            text, first_line=first_line, max_chars=limits.window_chars // OUTLINE_WINDOW_DIVISOR
        )
        if outline:
            marker = f"{marker}\n{outline}"
        body, omitted = f"{shown}\n{marker}", "end"
    elif output_window == "tail":
        shown, shown_lines = _tail_window(text, limits)
        marker = (
            f"[Output truncated: the last {shown_lines:,} of {total_lines:,} lines "
            f"({len(shown):,} of {len(text):,} chars) are shown. "
            f"{_where_the_rest_is(location, failure, capital=True)}]"
        )
        body, omitted = f"{shown}\n{marker}", "start"
    else:
        ends = _ends_window(text, limits, location=location, failure=failure)
        if ends is None:
            return result
        body, omitted = ends, "middle"
    noun = "line" if total_lines == 1 else "lines"
    top = f"[Output cut: {total_lines:,} {noun}, {omitted} omitted, see marker below.]"
    images = [block for block in result.content if isinstance(block, ImageContent)]
    return result.model_copy(update={"content": [TextContent(text=f"{top}\n{body}"), *images]})


def _where_the_rest_is(location: str | None, failure: str, *, capital: bool = False) -> str:
    """The closing words of a cut's marker: the saved file, or why there is none."""
    if location is not None:
        rest = f"full output: {location}. Page it with read offset/limit or grep it."
    else:
        rest = f"the full output could not be saved ({failure})."
    return rest[0].upper() + rest[1:] if capital else rest


def _ends_window(
    text: str, limits: BufferLimits, *, location: str | None, failure: str
) -> str | None:
    """The start and end of `text` around a marker for the middle, or None if all fits.

    Each end holds whole lines, as many as fit its share of `window_chars` and
    of `window_lines`. An end whose first line alone is too long holds what
    fits of that line, so a few very long lines (minified JSON, an event
    stream) cannot defeat the window.
    """
    # The newline that ends the output is put back after the end that is kept,
    # so the end is the last line with text on it, not the empty one after it.
    closing = "\n" if text.endswith("\n") else ""
    text = text.removesuffix("\n")
    lines = text.split("\n")
    head_chars = limits.window_chars * ENDS_HEAD_PERCENT // 100
    tail_chars = limits.window_chars - head_chars
    head_lines = max(1, limits.window_lines * ENDS_HEAD_PERCENT // 100)
    tail_lines = max(1, limits.window_lines - head_lines)
    head_count = _lines_that_fit(lines, head_chars, head_lines)
    tail_count = _lines_that_fit(lines[::-1], tail_chars, tail_lines)
    if head_count + tail_count >= len(lines):
        return None
    head = "\n".join(lines[:head_count]) if head_count else lines[0][:head_chars]
    tail = "\n".join(lines[-tail_count:]) if tail_count else lines[-1][-tail_chars:]
    middle = text[len(head) : len(text) - len(tail)]
    if not middle:
        return None
    # The newline that ends a whole kept line, or starts one, is not left out.
    if head_count:
        middle = middle.removeprefix("\n")
    if tail_count:
        middle = middle.removesuffix("\n")
    first, last = head_count + 1, len(lines) - tail_count
    span = f"line {first:,}" if first == last else f"lines {first:,}\u2013{last:,}"
    # An end that holds part of a line leaves the rest of that line out.
    span = f"{span} omitted" if head_count and tail_count else f"{span} omitted in part"
    marker = f"[\u2026 {span} ({len(middle):,} chars); {_where_the_rest_is(location, failure)}"
    if not head_count:
        # The middle starts inside line 1; what is left of that line is not
        # the start of a line, so it cannot be a heading.
        _, _, middle = middle.partition("\n")
        first += 1
    sections = _omitted_outline(
        middle,
        first_line=first,
        in_fence=sum(1 for line in lines[:head_count] if _FENCE.match(line)) % 2 == 1,
        max_chars=limits.window_chars // OUTLINE_WINDOW_DIVISOR,
    )
    if sections:
        marker = f"{marker} {sections}"
    return f"{head}\n{marker}]\n{tail}{closing}"


def _lines_that_fit(lines: list[str], max_chars: int, max_lines: int) -> int:
    """How many of the first `lines`, joined by newlines, fit both limits."""
    used = 0
    for fitted, line in enumerate(lines[:max_lines]):
        used += len(line) + (1 if fitted else 0)
        if used > max_chars:
            return fitted
    return min(len(lines), max_lines)


def read_first_line(arguments: Mapping[str, object]) -> int:
    """The file line a read with these arguments starts at (its offset, 1-indexed)."""
    offset = arguments.get("offset")
    if isinstance(offset, int) and not isinstance(offset, bool) and offset > 1:
        return offset
    return 1


def read_source_path(details: object) -> str | None:
    """The file a read result came from, from the read tool's details."""
    path = details.get("path") if isinstance(details, Mapping) else None
    return path if isinstance(path, str) else None


def _read_window(text: str, max_chars: int) -> str:
    """The head of a read's `text` that fits `max_chars`, cut between file lines.

    Rows (see `tau_coding.line_anchors`) are kept while they fit; the cut then
    backs up to the end of the last whole file line, unless that would leave
    nothing, in which case the rows of the first line that fit are kept. Text
    without row prefixes is cut between lines the same way.
    """
    body = _READ_TOOL_NOTE.sub("", text)
    rows = body.split("\n")
    kept: list[str] = []
    used = 0
    for row in rows:
        cost = len(row) + (1 if kept else 0)
        if used + cost > max_chars:
            break
        kept.append(row)
        used += cost
    if not kept:
        return text[:max_chars]
    if len(kept) < len(rows):
        whole = len(kept)
        while whole > 0 and _continues(rows[whole]):
            whole -= 1
        if whole > 0:
            kept = kept[:whole]
    return "\n".join(kept)


def _continues(row: str) -> bool:
    """Whether `row` is a continuation row of a wrapped line."""
    number = row_line_number(row)
    return number is not None and not number[1]


def _read_marker(
    text: str,
    shown: str,
    *,
    first_line: int,
    source_path: str | None,
    location: str | None,
    failure: str,
) -> tuple[str, int]:
    """Which file lines a cut read shows, of those it returned, and where to go on.

    Returned with the number of file lines the read gave.
    """
    body = _READ_TOOL_NOTE.sub("", text).removesuffix("\n")
    rows = body.split("\n")
    numbers = [row_line_number(row) for row in rows]
    if numbers and all(number is not None for number in numbers):
        last_asked = max(number[0] for number in numbers if number is not None)
        shown_rows = shown.count("\n") + 1
        next_row = numbers[shown_rows] if shown_rows < len(numbers) else None
        last_row = numbers[shown_rows - 1]
        assert last_row is not None
        whole_last = shown == "\n".join(rows[:shown_rows]) and (next_row is None or next_row[1])
        last_shown = last_row[0] if whole_last else last_row[0] - 1
    else:
        last_asked = first_line + body.count("\n")
        whole = shown.count("\n") + (1 if text[len(shown) : len(shown) + 1] in ("\n", "") else 0)
        last_shown = min(first_line + whole - 1, last_asked)
    rest = ""
    if last_shown >= first_line:
        shown_part = f"lines {first_line}-{last_shown} of the file are shown"
        next_line = last_shown + 1
    else:
        # Not even one line fits. Going on from it would show the same part
        # again, so go on from the line after, and say how to see the rest.
        seen = strip_row_prefixes(shown)
        shown_part = f"line {first_line} is too long to show whole"
        next_line = first_line + 1
        where = source_path if source_path is not None else location
        if seen is not None and where is not None:
            shown_part = f"{shown_part}; its first {len(seen):,} characters are shown"
            rest = (
                f" The rest of line {first_line}: "
                f"`sed -n '{first_line}p' {where} | cut -b {len(seen.encode()) + 1}-` through bash."
            )
    returned = last_asked - first_line + 1
    marker = f"[Read cut: {shown_part} (of {first_line}-{last_asked} returned); "
    if source_path is not None:
        return f"{marker}continue with read offset={next_line} on {source_path}.{rest}]", returned
    marker = f"{marker}continue with offset={next_line}.{rest}"
    if location is not None:
        marker = f"{marker} Full output: {location}. Page it with read (offset/limit) or grep it.]"
    else:
        marker = f"{marker} The full output could not be saved ({failure}).]"
    return marker, returned


def _outline(text: str, *, first_line: int, max_chars: int) -> str:
    """The def/class lines and document headings of `text` with their file line numbers."""
    entries, headings = _outline_entries(text, first_line=first_line)
    if not entries:
        return ""
    if headings:
        header, noun = "Outline (jump with read offset=<line>):", "entries"
    else:
        header, noun = "Definitions (jump with read offset=<line>):", "definitions"
    kept: list[str] = []
    used = len(header)
    for line_number, label in entries[:OUTLINE_MAX_ENTRIES]:
        entry = f"  L{line_number} {label}"
        if used + 1 + len(entry) > max_chars:
            break
        kept.append(entry)
        used += 1 + len(entry)
    left_out = len(entries) - len(kept)
    tail = [f"  ... {left_out:,} more {noun} not listed"] if left_out else []
    return "\n".join([header, *kept, *tail])


def _omitted_outline(text: str, *, first_line: int, in_fence: bool, max_chars: int) -> str:
    """One sentence naming the sections or definitions in the cut-out `text`.

    `first_line` is the line of the whole output that `text` starts at, so each
    name carries a line number a read offset on the saved output can use.
    `in_fence` says whether `text` starts inside a code fence opened before it.
    """
    entries, headings = _outline_entries(
        text, first_line=first_line, row_numbers=False, in_fence=in_fence, report=True
    )
    if not entries:
        return ""
    noun = "sections" if headings else "definitions"
    kept: list[str] = []
    used = len(f"Omitted {noun}: .")
    for line_number, label in entries[:OUTLINE_MAX_ENTRIES]:
        entry = f"{label.lstrip(' #')} (L{line_number})"
        if used + 2 + len(entry) > max_chars:
            break
        kept.append(entry)
        used += 2 + len(entry)
    left_out = len(entries) - len(kept)
    if not kept:
        return f"Omitted {noun}: {left_out:,}, not listed."
    more = f", and {left_out:,} more" if left_out else ""
    return f"Omitted {noun}: {', '.join(kept)}{more}."


def _outline_entries(
    text: str,
    *,
    first_line: int,
    row_numbers: bool = True,
    in_fence: bool = False,
    report: bool = False,
) -> tuple[list[tuple[int, str]], bool]:
    """Each outline entry of `text` as (line number, indented label); whether any is a heading.

    A row that carries its own line number (a read's) is listed under it,
    unless `row_numbers` is off; any other is numbered by its place in `text`,
    counting from `first_line`. `in_fence` says whether `text` starts inside a
    code fence. `report` is for text that may be a report quoting a line of
    code: a `##` or deeper heading then makes it a document, whatever
    definitions it holds.
    """
    found: list[tuple[int, str, int, str]] = []
    for index, line in enumerate(text.split("\n")):
        line = line.removesuffix("\r")
        if _CONTINUATION_ROW.match(line):
            continue
        if _FENCE.match(line):
            in_fence = not in_fence
            continue

        def numbered(match: re.Match[str], place: int = first_line + index) -> int:
            return int(match["number"]) if row_numbers and match["number"] else place

        match = _DEFINITION.match(line)
        if match is not None:
            depth = len(match["indent"].expandtabs(4)) // 4
            kind = " ".join(match["kind"].split())
            label = f"{'  ' * depth}{kind} {match['name']}"
            found.append((index, "fenced" if in_fence else "code", numbered(match), label))
            continue
        if in_fence:
            continue
        match = _MARKDOWN_HEADING.match(line)
        if match is not None:
            label = f"{'  ' * (len(match['hashes']) - 1)}{match['hashes']} {match['title']}"
            found.append((index, "heading", numbered(match), label))
            continue
        match = _NUMBERED_SECTION.match(line)
        if match is not None and _is_section_title(match["title"]):
            label = f"{'  ' * match['section'].count('.')}{match['section']} {match['title']}"
            found.append((index, "section", numbered(match), label))
    # Code is outlined by its definitions, so its `# comments` are not headings.
    # A Markdown document marks its sections with headings, so its numbered
    # lines are list items, as is any numbered line next to another one.
    kinds = {kind for _, kind, _, _ in found}
    # A code comment is one `#`; two or more at the start of a line outside a
    # fence are a heading, and a document with one is not a source file.
    if report and any(
        kind == "heading" and label.lstrip().startswith("##") for _, kind, _, label in found
    ):
        found = [entry for entry in found if entry[1] != "code"]
        kinds.discard("code")
    numbered_lines = {index for index, kind, _, _ in found if kind == "section"}
    entries: list[tuple[int, str]] = []
    headings = False
    for index, kind, line_number, label in found:
        if kind == "heading" and "code" in kinds:
            continue
        if kind == "section" and (
            kinds & {"code", "heading"} or {index - 1, index + 1} & numbered_lines
        ):
            continue
        entries.append((line_number, label))
        headings = headings or kind in ("heading", "section")
    return entries, headings


def _is_section_title(title: str) -> bool:
    """A short title, not a numbered sentence of prose."""
    return len(title) <= SECTION_TITLE_MAX_CHARS and not title.endswith((".", ",", ";", ":"))


def _tail_window(text: str, limits: BufferLimits) -> tuple[str, int]:
    """The end of `text` that fits and how many whole lines it shows (at least 1)."""
    shown = "\n".join(text.split("\n")[-limits.window_lines :])
    if len(shown) > limits.window_chars:
        # A few very long lines (minified JSON, an event stream) would defeat a
        # line window on their own, so the character cap always applies too.
        shown = shown[-limits.window_chars :]
        return shown, max(shown.count("\n"), 1)
    return shown, shown.count("\n") + 1


def _unused_path(path: Path) -> Path:
    """`path`, or a numbered sibling if it is taken.

    Some providers reuse short call ids across responses, and an earlier
    transcript marker must keep pointing at the output it named.
    """
    candidate = path
    for number in count(2):
        if not candidate.exists():
            return candidate
        candidate = path.with_name(f"{path.stem}-{number}{path.suffix}")
    raise AssertionError("unreachable")


def _require_own_directory(directory: Path) -> None:
    """Refuse a directory another user made, or a symlink planted in its place.

    The default buffer directory has a predictable name in the shared temp
    directory, and outputs can hold secrets.
    """
    info = os.lstat(directory)
    getuid = getattr(os, "getuid", None)
    if stat.S_ISLNK(info.st_mode) or (getuid is not None and info.st_uid != getuid()):
        raise PermissionError(f"not a directory owned by this user: {directory}")


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
    _require_own_directory(buffer_dir)
    fd, tmp = tempfile.mkstemp(dir=buffer_dir, prefix=".buffer-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path
