"""Recall entries a compaction dropped: stable references and the history tool.

Every session entry has a reference, ``#N``: its 1-based position in the
session file. The file is append-only and shared by every branch, so a
position never changes and is unique. References are never added to ordinary
messages. They appear only in text the harness writes, the compaction index
and `history` results, and the `history` tool turns
one back into the entry. See dev-notes/session-recall-and-continuation-compaction.md.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    BashExecutionMessage,
    TextContent,
    ToolResultMessage,
    UserMessage,
    message_text,
)
from tau_agent.session import CompactionEntry, MessageEntry
from tau_agent.session.entries import CustomMessageEntry, SessionEntry
from tau_agent.tools import AgentTool, AgentToolResult
from tau_agent.types import JSONValue
from tau_coding.tool_buffers import output_file_name

HISTORY_TOOL_NAME = "history"
#: Matches a `history` search may return before it asks for a narrower one.
MAX_SEARCH_MATCHES = 40
#: An entry in a search or tool listing is cut to this many characters; asking
#: for it by reference shows it whole.
LISTING_ENTRY_CHARS = 600
#: Characters a `history` answer shows when the session sets no smaller limit.
#: The session passes its inline limit, so an answer is never windowed after it
#: is made: windowing keeps the tail, and an answer read from its end loses the
#: order its notes refer to.
REFS_MAX_CHARS = 20_000
#: Room kept in that limit for the note naming what was left out.
NOTE_CHARS = 200
#: Characters of one tool-call argument shown in an entry header or index line.
ARGUMENT_CHARS = 120

_REF_RANGE = re.compile(r"#?(\d+)\s*(?:-|\.\.|–)\s*#?(\d+)|#?(\d+)")


@dataclass(frozen=True, slots=True)
class NumberedEntry:
    """A session entry with its stable reference."""

    number: int
    entry: SessionEntry

    @property
    def ref(self) -> str:
        return f"#{self.number}"


def number_entries(entries: Sequence[SessionEntry]) -> dict[str, int]:
    """Map each entry id to its reference number (1-based file position)."""
    return {entry.id: index for index, entry in enumerate(entries, start=1)}


def numbered_branch(
    file_entries: Sequence[SessionEntry], branch: Sequence[SessionEntry]
) -> list[NumberedEntry]:
    """The entries of `branch` (root to leaf) numbered by their place in the file.

    An entry not yet in the file (a write still deferred) has no reference yet
    and is left out.
    """
    numbers = number_entries(file_entries)
    return [NumberedEntry(numbers[entry.id], entry) for entry in branch if entry.id in numbers]


def parse_refs(text: str) -> list[tuple[int, int]]:
    """Parse ``#12``, ``#12-#40``, ``12,15`` into inclusive ranges."""
    ranges: list[tuple[int, int]] = []
    for match in _REF_RANGE.finditer(text):
        if match.group(3) is not None:
            number = int(match.group(3))
            ranges.append((number, number))
        else:
            low, high = int(match.group(1)), int(match.group(2))
            ranges.append((min(low, high), max(low, high)))
    return ranges


def entry_message(entry: SessionEntry) -> AgentMessage | None:
    """The message an entry carries, if it is one the model saw."""
    if isinstance(entry, MessageEntry):
        return entry.message
    if isinstance(entry, CustomMessageEntry):
        return UserMessage(content=entry.content)
    return None


def short_argument(arguments: Mapping[str, JSONValue]) -> str:
    """The one argument that best names what a tool call did, shortened."""
    for key in ("command", "path", "file_path", "pattern", "query", "url", "prompt", "refs"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return _clip(" ".join(value.split()), ARGUMENT_CHARS)
    if not arguments:
        return ""
    return _clip(json.dumps(arguments, ensure_ascii=False, default=str), ARGUMENT_CHARS)


def render_entry(
    item: NumberedEntry, *, outputs_dir: Path | None, max_chars: int | None = None
) -> str | None:
    """One entry as the history tool shows it, or None for a metadata entry."""
    entry = item.entry
    if isinstance(entry, CompactionEntry):
        body = entry.summary
        header = f"{item.ref} compaction summary"
    else:
        message = entry_message(entry)
        if message is None:
            return None
        header, body = _message_header_and_body(message, outputs_dir=outputs_dir)
        header = f"{item.ref} {header}"
    if max_chars is not None:
        body = _clip(body, max_chars)
    return f"{header}\n{body}".rstrip()


def _message_header_and_body(message: AgentMessage, *, outputs_dir: Path | None) -> tuple[str, str]:
    if isinstance(message, AssistantMessage):
        calls = [
            f"  -> {call.name}({short_argument(call.arguments)}) [call {call.id}]"
            for call in message.tool_calls
        ]
        return "assistant", "\n".join([message.text, *calls]).strip()
    if isinstance(message, ToolResultMessage):
        header = f"tool result {message.tool_name}"
        if message.is_error:
            header += " (error)"
        header += f" [call {message.tool_call_id}]"
        if outputs_dir is not None:
            path = outputs_dir / output_file_name(message.tool_name, message.tool_call_id)
            if path.exists():
                header += f" output: {path}"
        return header, message.text
    if isinstance(message, BashExecutionMessage):
        return f"user terminal command `{message.command}`", message.output
    return message.role, message_text(message)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


@dataclass(frozen=True, slots=True)
class RecalledBranch:
    """What the history tool answers from."""

    entries: list[NumberedEntry]
    #: Where tool outputs are saved, if anywhere.
    outputs_dir: Path | None
    #: The most characters an answer may have.
    max_chars: int = REFS_MAX_CHARS


class RecallSource(Protocol):
    """Where the history tool reads the current branch from."""

    async def recall_branch(self) -> RecalledBranch:
        """The current branch with references, and how to answer from it."""
        ...


def history_text(
    branch: Sequence[NumberedEntry],
    *,
    outputs_dir: Path | None,
    refs: str | None = None,
    query: str | None = None,
    tool: str | None = None,
    max_chars: int = REFS_MAX_CHARS,
) -> str:
    """Answer one history request against the entries of the current branch.

    The answer is at most `max_chars` characters, and names what it left out.
    """
    if not branch:
        return "This session has no recorded entries yet."
    items = list(branch)
    limit: int | None = LISTING_ENTRY_CHARS
    if refs:
        ranges = parse_refs(refs)
        if not ranges:
            return f"No references in {refs!r}; use #12, #12-#40 or 12,15."
        items = [i for i in items if any(low <= i.number <= high for low, high in ranges)]
        if not items:
            last = branch[-1].number
            return f"No entries on the current branch match {refs} (references run to #{last})."
        limit = None
    if tool:
        items = [i for i in items if _involves_tool(i.entry, tool)]
    if query:
        needle = query.casefold()
        # Earlier history calls and answers hold the query itself; they are
        # searched only when history is the tool asked for.
        with_history = tool == HISTORY_TOOL_NAME
        items = [
            i
            for i in items
            if needle in _searchable_text(i.entry, with_history=with_history).casefold()
        ]
    if not refs and not tool and not query:
        return (
            "Ask for entries by `refs` (#12, #12-#40), search them with `query`, "
            f"or list one tool's calls with `tool`. References run #1-#{branch[-1].number}."
        )
    rendered = [
        text
        for text in (render_entry(i, outputs_dir=outputs_dir, max_chars=limit) for i in items)
        if text is not None
    ]
    if not rendered:
        return (
            "No entries on the current branch match. Only this session's branch is "
            "searched, not earlier sessions or other branches."
        )
    budget = max(0, max_chars - NOTE_CHARS)
    if limit is None:
        return _within_refs_budget(rendered, budget)
    shown = _whole_entries(rendered[:MAX_SEARCH_MATCHES], budget)
    if len(shown) == len(rendered):
        return "\n\n".join(shown)
    return "\n\n".join(shown) + (
        f"\n\n[{len(rendered) - len(shown)} more matches not shown; "
        "narrow the search or ask for a range by refs.]"
    )


def _whole_entries(rendered: Sequence[str], budget: int) -> list[str]:
    """The leading entries of `rendered` that fit `budget` whole."""
    shown: list[str] = []
    size = -2
    for text in rendered:
        size += len(text) + 2
        if size > budget:
            break
        shown.append(text)
    return shown


def _within_refs_budget(rendered: Sequence[str], budget: int) -> str:
    """Whole entries up to `budget` characters, naming the first one left out.

    A first entry too long for the budget is cut, and says where it stops.
    """
    shown = _whole_entries(rendered, budget)
    if not shown:
        first = rendered[0]
        cut = first[:budget].rstrip()
        note = (
            f"\n\n[{first.split(' ', 1)[0]} is cut at {len(cut):,} of {len(first):,} "
            "characters. A tool result names the file holding its full output; read it there."
        )
        if len(rendered) > 1:
            following = rendered[1].split(" ", 1)[0]
            note += f" {len(rendered) - 1} more entries, from {following}, are not shown."
        return cut + note + "]"
    if len(shown) == len(rendered):
        return "\n\n".join(shown)
    rest = rendered[len(shown)].split(" ", 1)[0]
    return "\n\n".join(shown) + (
        f"\n\n[{len(rendered) - len(shown)} more entries not shown, from {rest}: "
        f"the answer stops at {budget + NOTE_CHARS:,} characters. Ask for them by refs.]"
    )


def _involves_tool(entry: SessionEntry, tool: str) -> bool:
    message = entry_message(entry)
    if isinstance(message, ToolResultMessage):
        return message.tool_name == tool
    if isinstance(message, AssistantMessage):
        return any(call.name == tool for call in message.tool_calls)
    return False


def _searchable_text(entry: SessionEntry, *, with_history: bool = False) -> str:
    if isinstance(entry, CompactionEntry):
        return entry.summary
    message = entry_message(entry)
    if message is None:
        return ""
    if isinstance(message, AssistantMessage):
        calls = [
            json.dumps(call.arguments, ensure_ascii=False)
            for call in message.tool_calls
            if with_history or call.name != HISTORY_TOOL_NAME
        ]
        return "\n".join([message.text, *calls])
    if isinstance(message, ToolResultMessage) and message.tool_name == HISTORY_TOOL_NAME:
        return message_text(message) if with_history else ""
    if isinstance(message, BashExecutionMessage):
        return f"{message.command}\n{message.output}"
    return message_text(message)


HISTORY_DESCRIPTION = (
    "Read this session's earlier entries, including ones compaction removed from "
    "context; other sessions are not searched. Entries are referenced as #N (the "
    "compaction summary's index cites them). "
    "Pass `refs` (#12, #12-#40, 12,15) for those entries in full, `query` to search "
    "entry text and tool arguments, or `tool` to list one tool's calls and results; "
    "query and tool can combine. Tool results name the file holding their full output, "
    "which read and grep also work on."
)


def create_history_tool(source: RecallSource) -> AgentTool:
    """Create the `history` tool reading the current branch from `source`."""

    async def execute(
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: object = None,
        on_update: object = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        recalled = await source.recall_branch()
        text = history_text(
            recalled.entries,
            outputs_dir=recalled.outputs_dir,
            refs=_string_argument(arguments, "refs"),
            query=_string_argument(arguments, "query"),
            tool=_string_argument(arguments, "tool"),
            max_chars=recalled.max_chars,
        )
        return AgentToolResult(content=[TextContent(text=text)])

    return AgentTool(
        name=HISTORY_TOOL_NAME,
        label="History",
        description=HISTORY_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "refs": {"type": "string", "description": "Entry references: #12, #12-#40, 12,15."},
                "query": {"type": "string", "description": "Case-insensitive text to search for."},
                "tool": {"type": "string", "description": "Only this tool's calls and results."},
            },
        },
        execute_fn=execute,
        prompt_snippet="Read earlier session entries by #N reference or search",
    )


def _string_argument(arguments: Mapping[str, JSONValue], key: str) -> str | None:
    value = arguments.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return value.strip() or None if isinstance(value, str) else None


class LateRecallSource:
    """A recall source bound to its session after the tools are built.

    The session's tool list is built before the session object exists, so the
    history tool holds this and the session binds itself once constructed.
    """

    def __init__(self) -> None:
        self._read: Callable[[], Awaitable[RecalledBranch]] | None = None

    def bind(self, read: Callable[[], Awaitable[RecalledBranch]]) -> None:
        self._read = read

    async def recall_branch(self) -> RecalledBranch:
        if self._read is None:
            return RecalledBranch([], None)
        return await self._read()


#: Characters of a user message quoted on its index line.
INDEX_PROMPT_CHARS = 100
#: Tool calls listed on one index line before the rest are counted.
INDEX_CALLS_PER_TURN = 8
COMPACTION_INDEX_HEADING = "## Index of compacted entries"
INDEX_NOTE = (
    "Newest history is listed turn by turn, older history as captioned sections, "
    "chapters and parts, each with its #N range. `history` with refs=#N-#M shows a "
    "range in full; tool output files can be read or grepped."
)


def split_compaction_index(summary: str) -> tuple[str, str]:
    """Split a stored summary into the index it starts with and the rest."""
    if not summary.startswith(COMPACTION_INDEX_HEADING):
        return "", summary
    lines = summary.split("\n")
    position = 1
    while position < len(lines) and lines[position].strip():
        position += 1
    while position < len(lines) and not lines[position].strip():
        position += 1
    while position < len(lines) and lines[position].strip():
        position += 1
    return "\n".join(lines[:position]), "\n".join(lines[position:]).lstrip("\n")


def with_compaction_index(index: str, summary: str) -> str:
    """Put a compaction index before the summary text it belongs to."""
    return f"{index}\n\n{summary}" if index else summary


def index_turn_line(turn: Sequence[NumberedEntry], *, outputs_dir: Path | None) -> str:
    """One user turn as an index line: its range, what was asked, its tool calls."""
    first, last = turn[0].number, turn[-1].number
    span = f"#{first}" if first == last else f"#{first}-#{last}"
    opener = entry_message(turn[0].entry)
    if isinstance(opener, BashExecutionMessage):
        head = f"user ran `{_clip(' '.join(opener.command.split()), INDEX_PROMPT_CHARS)}`"
    elif isinstance(opener, UserMessage):
        head = f'user: "{_clip(_first_line(message_text(opener)), INDEX_PROMPT_CHARS)}"'
    else:
        # The rest of a turn an earlier compaction cut at its boundary.
        head = "(turn continued)"
    results: dict[str, ToolResultMessage] = {}
    calls: list[tuple[str, str, Mapping[str, JSONValue]]] = []
    for item in turn:
        message = entry_message(item.entry)
        if isinstance(message, AssistantMessage):
            calls.extend((call.id, call.name, call.arguments) for call in message.tool_calls)
        elif isinstance(message, ToolResultMessage):
            results[message.tool_call_id] = message
    parts: list[str] = []
    for call_id, name, arguments in calls[:INDEX_CALLS_PER_TURN]:
        argument = short_argument(arguments)
        part = f"{name} `{_clip(argument, 60)}`" if argument else name
        result = results.get(call_id)
        if result is not None and result.is_error:
            part += " (error)"
        if outputs_dir is not None and result is not None:
            path = outputs_dir / output_file_name(name, call_id)
            if path.exists():
                part += f" [{path.name}]"
        parts.append(part)
    if len(calls) > INDEX_CALLS_PER_TURN:
        parts.append(f"+{len(calls) - INDEX_CALLS_PER_TURN} more calls")
    return f"{span} {head}" + (f" -> {'; '.join(parts)}" if parts else "")


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return " ".join(line.split())
    return ""
