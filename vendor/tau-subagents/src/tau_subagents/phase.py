"""What a live subagent is doing right now, and for how long.

A run's phase tells a slow model from a slow tool from a wedged one:

- ``queued`` — waiting for a concurrency slot;
- ``awaiting_model`` — a request is out and no response has started (the
  elapsed time is time to first token, plus setup before the first request);
- ``streaming`` — the model is sending its response;
- ``in_tool`` — a tool call is executing.

The extension sets the phase from the child's agent events; the UI and
``get_subagent_result`` read it through the formatters here. This module
imports nothing from the extension, so the UI modules can use it too.
"""

from __future__ import annotations

from typing import Protocol

PHASE_QUEUED = "queued"
PHASE_AWAITING_MODEL = "awaiting_model"
PHASE_STREAMING = "streaming"
PHASE_IN_TOOL = "in_tool"
PHASES = (PHASE_QUEUED, PHASE_AWAITING_MODEL, PHASE_STREAMING, PHASE_IN_TOOL)
#: Characters per token for the estimate of a response still streaming.
CHARS_PER_TOKEN = 4


class PhasedRun(Protocol):
    status: str
    phase: str
    phase_since: float | None
    current_tool: str | None
    current_tool_started_at: float | None
    streamed_chars: int


def format_seconds(seconds: float) -> str:
    """Compact elapsed time: 42s, 3m05s, 2h10m."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def phase_elapsed(run: PhasedRun, now: float) -> float | None:
    """Seconds spent in the current phase (the current tool, when in one)."""
    if run.phase == PHASE_IN_TOOL and run.current_tool_started_at is not None:
        return now - run.current_tool_started_at
    if run.phase_since is None:
        return None
    return now - run.phase_since


def streamed_tokens(run: PhasedRun) -> int:
    """Estimated tokens of the response streaming now (0 when none is)."""
    if run.phase != PHASE_STREAMING:
        return 0
    return run.streamed_chars // CHARS_PER_TOKEN


def format_phase(run: PhasedRun, now: float) -> str:
    """Short phase text for a live run ("in bash 20s"); empty once it ends."""
    if run.status == "queued":
        return "queued"
    if run.status != "running":
        return ""
    elapsed = phase_elapsed(run, now)
    suffix = "" if elapsed is None else f" {format_seconds(elapsed)}"
    if run.phase == PHASE_IN_TOOL and run.current_tool is not None:
        return f"in {run.current_tool}{suffix}"
    if run.phase == PHASE_STREAMING:
        tokens = streamed_tokens(run)
        streamed = f", ~{tokens} tokens" if tokens else ""
        return f"streaming{suffix}{streamed}"
    return f"awaiting model{suffix}"
