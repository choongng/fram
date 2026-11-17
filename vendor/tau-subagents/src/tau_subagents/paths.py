"""Where this extension keeps its files under the Tau home.

One place names the directories: the durable transcript root, the job
registry root, and the two path components that line our entries up with
Tau's own session directories. Nothing else computes a `~/.tau` path, so a
later change to the layout has one file to edit.

The extension assumes the `.tau` name under the user home; Tau does not
expose the configured `TauPaths` to extensions (ADR 0003), so
`TAU_SUBAGENTS_DIR` is the manual escape hatch for the transcript root.
"""

from __future__ import annotations

import os
import re
from hashlib import sha256
from pathlib import Path


def encode_cwd(cwd: str) -> str:
    """Encode a working directory into a single, never-empty path component."""
    return cwd.replace("/", "-").replace("\\", "-").lstrip("-") or "root"


def project_dir_name(cwd: Path) -> str:
    """Return `<slug>-<hash>` for a cwd, matching Tau's session-dir naming.

    A port of `tau_coding.paths.project_session_dir` naming (slug of the
    resolved path, home-relative when possible, plus the first 6 hex chars of
    its SHA-256) so `~/.tau/subagents/` entries line up visually with
    `~/.tau/sessions/` entries for the same project.
    """
    resolved = cwd.resolve()
    digest = sha256(str(resolved).encode("utf-8")).hexdigest()[:6]
    slug = _slugify_path(resolved)
    return f"{slug or 'project'}-{digest}"


def _slugify_path(path: Path, *, max_length: int = 72) -> str:
    parts = [part for part in path.parts if part not in (path.anchor, "")]
    try:
        relative_to_home = path.relative_to(Path.home())
    except ValueError:
        pass
    else:
        parts = ["home", *relative_to_home.parts]

    slug_parts = [
        normalized
        for part in parts
        if (normalized := re.sub(r"[^a-zA-Z0-9._-]+", "-", part).strip(".-_").lower())
    ]
    slug = "-".join(slug_parts)
    if len(slug) <= max_length:
        return slug

    suffix_parts: list[str] = []
    suffix_length = 0
    for part in reversed(slug_parts):
        next_length = suffix_length + len(part) + (1 if suffix_parts else 0)
        if next_length > max_length:
            break
        suffix_parts.append(part)
        suffix_length = next_length
    return "-".join(reversed(suffix_parts)) or slug[-max_length:].strip("-")


def tau_home(home: Path | None = None) -> Path:
    """Return the Tau home: `<home>/.tau` for an explicit `home`, else Tau's own.

    Tau's own is `TAU_HOME` when that is set, as it is for Fram (`~/.fram`),
    and `~/.tau` otherwise. An explicit `home` is the user home directory a
    test or caller pins, so it ignores the environment, as an explicit
    `TauPaths` does.
    """
    if home is not None:
        return home / ".tau"
    configured = os.environ.get("TAU_HOME", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".tau"


def project_tau_dir(cwd: Path) -> Path:
    """Return the project's Tau folder: `.tau`, or what `TAU_PROJECT_DIRNAME` names.

    Fram sets `TAU_PROJECT_DIRNAME` to `.fram`. Tau releases that predate the
    variable never set it, so `.tau` remains their folder.
    """
    name = os.environ.get("TAU_PROJECT_DIRNAME", "").strip() or ".tau"
    return cwd / name


def transcripts_root(home: Path | None = None) -> Path:
    """Return the durable transcript root under the Tau home.

    `TAU_SUBAGENTS_DIR` overrides the location entirely (pi's
    `PI_SUBAGENTS_TEMP_ROOT` precedent); it is also the test-isolation seam.
    Otherwise it is under the Tau home (`tau_home`), which follows `TAU_HOME`
    as Tau itself does; Tau does not expose its configured `TauPaths` to
    extensions (see ADR 0003).
    """
    configured = os.environ.get("TAU_SUBAGENTS_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return tau_home(home) / "subagents"


def jobs_home(home: Path | None = None) -> Path:
    """Return the job registry root: `<Tau home>/jobs`, or `TAU_JOBS_DIR` if set.

    Sibling of the transcript root rather than a directory under it, because
    a job is a command and not a transcript, and only one of the two is Tau
    session data. `TAU_JOBS_DIR` overrides it entirely, so the CLI, the
    extension's watcher, and a test each point at the same registry without
    threading a root through every caller.
    """
    configured = os.environ.get("TAU_JOBS_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    return tau_home(home) / "jobs"
