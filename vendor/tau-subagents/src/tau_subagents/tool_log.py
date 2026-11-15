"""The last few tool calls of a run, for a peek at what it has been doing.

The extension appends a record at each ``tool_execution_start`` and closes it
at the matching ``tool_execution_end``; ``get_subagent_result`` with
``tools: N`` prints the last N. Only the most recent ``RECENT_TOOLS_KEPT``
calls are kept, so the log costs nothing on a long run.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .phase import format_seconds

#: Tool calls kept per run; also the most get_subagent_result will show.
RECENT_TOOLS_KEPT = 20
#: Longest argument summary before it is cut with an ellipsis.
ARGS_SUMMARY_CHARS = 80
# The argument that says most about a call, for the tools that have one.
_PRIMARY_ARGS = ("command", "path", "file_path", "pattern", "query", "url", "prompt")


@dataclass
class ToolCallRecord:
    call_id: str
    name: str
    args: str
    started_at: float
    ended_at: float | None = None
    is_error: bool = False


def summarize_args(args: object) -> str:
    """One line naming a call's main argument (a command, a path, a pattern)."""
    if isinstance(args, Mapping):
        for key in _PRIMARY_ARGS:
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return _clip(value)
        if not args:
            return ""
    try:
        text = json.dumps(args, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        text = str(args)
    return _clip(text)


def _clip(text: str) -> str:
    line = " ".join(text.split())
    if len(line) <= ARGS_SUMMARY_CHARS:
        return line
    return line[: ARGS_SUMMARY_CHARS - 1] + "…"


def format_tool_log(
    records: Iterable[ToolCallRecord],
    limit: int,
    total: int,
    now: float,
    *,
    live: bool = True,
) -> str:
    """The last ``limit`` calls, oldest first, one line each.

    ``live`` is False once the run has ended: a call still open then was
    cut off by a stop or cancel rather than running.
    """
    shown = list(records)[-limit:] if limit > 0 else []
    if not shown:
        return "Recent tool calls: none."
    lines = [f"Recent tool calls (last {len(shown)} of {total}):"]
    for record in shown:
        call = f"`{record.args}`" if record.args else "()"
        if record.ended_at is None:
            state = "running" if live else "never finished; ran"
            outcome = f"{state} {format_seconds(now - record.started_at)}"
        else:
            took = format_seconds(record.ended_at - record.started_at)
            outcome = f"error after {took}" if record.is_error else f"ok in {took}"
        lines.append(f"- {record.name} {call} — {outcome}")
    return "\n".join(lines)


def tool_log_details(
    records: Iterable[ToolCallRecord], limit: int, now: float, *, live: bool = True
) -> list[dict[str, object]]:
    """The same calls as structured data for a tool result's details."""
    shown = list(records)[-limit:] if limit > 0 else []
    return [
        {
            "tool": record.name,
            "args": record.args,
            "running": live and record.ended_at is None,
            "finished": record.ended_at is not None,
            "is_error": record.is_error,
            "elapsed_s": round(
                (record.ended_at if record.ended_at is not None else now)
                - record.started_at,
                1,
            ),
        }
        for record in shown
    ]
