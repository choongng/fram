"""JSONL output files for subagent transcripts (ADR 0003).

Each run streams its child transcript to a durable directory under the Tau
home so the transcript can be inspected outside the parent conversation and
survives reboots (upstream pi-subagents keeps child sessions durable too):

    ~/.tau/subagents/<cwd-slug>-<hash>/<parent-session>/tasks/<id>.jsonl

The directory name mirrors `tau_coding.paths.project_session_dir` but lives
outside `~/.tau/sessions/` so Tau's session index never sees these files.
By default transcripts are kept indefinitely. When `transcriptRetentionDays`
is positive, a retention sweep (`sweep_transcripts`, run on session_start)
deletes older files; setting it to 0 falls back to the old ephemeral location
in the system temp directory.

The initial entry is the prompt. New session messages are flushed when a
tool starts (so the call a long tool is running is on disk while it runs) and
at each `turn_end`. Each entry's `timestamp` is when its message was created,
not when it was flushed. A run that fails before it starts gets a stub file: the prompt and
an `error` entry. Write errors are swallowed and all IO runs in a thread.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from .paths import encode_cwd, project_dir_name, transcripts_root

__all__ = [
    "OutputFileWriter",
    "encode_cwd",
    "output_file_path",
    "project_dir_name",
    "sweep_transcripts",
    "transcripts_root",
]

_SECONDS_PER_DAY = 86_400


def output_file_path(
    cwd: Path,
    parent_session_id: str | None,
    agent_id: str,
    home: Path | None = None,
    *,
    durable: bool = True,
) -> Path:
    """Return the transcript path for one run.

    Durable (default): `~/.tau/subagents/<slug>-<hash>/<session>/tasks/`.
    Ephemeral (`durable=False`, i.e. `transcriptRetentionDays: 0`): the old
    per-uid directory in the system temp dir.
    """
    if durable:
        base = transcripts_root(home) / project_dir_name(cwd)
    else:
        uid = os.getuid() if hasattr(os, "getuid") else 0
        base = (
            Path(tempfile.gettempdir())
            / f"tau-subagents-{uid}"
            / encode_cwd(str(cwd))
        )
    return (
        base
        / (parent_session_id or "no-session")
        / "tasks"
        / f"{agent_id}.jsonl"
    )


def full_report_path(transcript: Path) -> Path:
    """Where a run keeps, as plain text, a final report it was asked to condense."""
    return transcript.with_suffix(".report.md")


def sweep_transcripts(
    retention_days: int,
    home: Path | None = None,
    *,
    now: float | None = None,
) -> int:
    """Delete durable transcripts older than `retention_days`; return count.

    Best-effort: unreadable entries are skipped, emptied `tasks`/session/
    project directories are pruned, and the root itself is kept. Blocking —
    callers run it in a thread.
    """
    if retention_days <= 0:
        return 0
    root = transcripts_root(home)
    if not root.is_dir():
        return 0
    cutoff = (now if now is not None else datetime.now(UTC).timestamp()) - (
        retention_days * _SECONDS_PER_DAY
    )
    removed = 0
    for transcript in root.glob("*/*/tasks/*.jsonl"):
        try:
            if transcript.stat().st_mtime < cutoff:
                transcript.unlink()
                full_report_path(transcript).unlink(missing_ok=True)
                removed += 1
        except OSError:
            continue
    # Prune now-empty directories, deepest first, never the root.
    for directory in sorted(
        (d for d in root.rglob("*") if d.is_dir()),
        key=lambda d: len(d.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()  # only succeeds when empty
        except OSError:
            continue
    return removed


class OutputFileWriter:
    """Appends transcript entries for one run to its JSONL output file."""

    def __init__(
        self, path: Path, agent_id: str, cwd: Path, *, inherited: int = 0
    ) -> None:
        self.path = path
        self._agent_id = agent_id
        self._cwd = cwd
        # index 0 is the prompt, written by write_initial. Forks additionally
        # seed `inherited` parent messages into the session; skipping them
        # keeps the whole parent transcript out of the durable file.
        self._inherited = inherited
        self._written = 1 + inherited
        self.started = False

    async def write_initial(self, prompt: str) -> None:
        self.started = True
        await asyncio.to_thread(self._write_initial_blocking, prompt)

    async def write_error(self, error: str) -> None:
        """Append an error entry for a run that failed outside the model."""
        await asyncio.to_thread(self._write_error_blocking, error)

    async def flush(self, messages: Sequence[object]) -> None:
        await asyncio.to_thread(self._flush_blocking, list(messages))

    async def compacted(self, messages: Sequence[object]) -> None:
        """Note a compaction and count the shorter context it left as written.

        Everything before it was flushed at the turn end it follows, but the
        context now holds fewer messages, so a count kept from the old one
        would skip every later message.
        """
        await asyncio.to_thread(self._compacted_blocking, list(messages))

    def _write_initial_blocking(self, prompt: str) -> None:
        try:
            root = self.path.parents[3]
            root.mkdir(parents=True, exist_ok=True)
            os.chmod(root, 0o700)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            entry = self._entry("user", {"role": "user", "content": prompt})
            if self._inherited:
                entry["inheritedMessages"] = self._inherited
            self._append(entry)
        except OSError:
            pass

    def _write_error_blocking(self, error: str) -> None:
        try:
            self._append(self._entry("error", {"role": "error", "content": error}))
        except OSError:
            pass

    def _flush_blocking(self, messages: list[object]) -> None:
        try:
            new_messages = messages[self._written :]
            if not new_messages:
                return
            for message in new_messages:
                self._append(
                    self._entry(
                        _entry_type(message),
                        _serialize_message(message),
                        at=_message_time(message),
                    )
                )
            self._written = len(messages)
        except OSError:
            pass

    def _compacted_blocking(self, messages: list[object]) -> None:
        summary = getattr(messages[0], "text", "") if messages else ""
        self._written = len(messages)
        try:
            self._append(
                self._entry("compaction", {"role": "compaction", "content": summary})
            )
        except OSError:
            pass

    def _entry(
        self, entry_type: str, message: object, *, at: datetime | None = None
    ) -> dict[str, object]:
        return {
            "isSidechain": True,
            "agentId": self._agent_id,
            "type": entry_type,
            "message": message,
            "timestamp": (at or datetime.now(UTC)).isoformat(),
            "cwd": str(self._cwd),
        }

    def _append(self, entry: dict[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def _entry_type(message: object) -> str:
    role = getattr(message, "role", None)
    if role in ("assistant", "user"):
        return str(role)
    return "toolResult"


def _message_time(message: object) -> datetime | None:
    """When a Tau message was created (its ``timestamp`` is Unix ms)."""
    stamp = getattr(message, "timestamp", None)
    if not isinstance(stamp, int) or isinstance(stamp, bool):
        return None
    try:
        return datetime.fromtimestamp(stamp / 1000, UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _serialize_message(message: object) -> object:
    dump = getattr(message, "model_dump", None)
    if dump is not None:
        try:
            return dump(mode="json")
        except Exception:  # noqa: BLE001 - transcripts degrade to repr
            pass
    return str(message)
