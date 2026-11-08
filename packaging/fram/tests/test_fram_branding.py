"""Tau's display text must read its name from the branding seam, or fram shows "Tau".

This scans Tau's sources for string literals that spell out Tau's name or
command, outside docstrings. Each one must be a deliberate keep, listed below.
When a Tau release adds display text that fails here, make it read
``BRANDING`` (see dev-notes/branding.md); if it names something that must stay
Tau's, list it.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "src"

# Tau's name, ``tau <word>`` or a bare ``tau`` as a command, or the τ mark.
# Closing punctuation counts only at a word's end, so ``tau.compaction`` and
# ``tau://`` stay identifiers.
TAU_NAME = re.compile(r"\bTau\b|(?<![\w./-])tau(?=[.,:;!?)\]]*(?:[\s`'\"]|$))|τ")

# Files whose every mention is identity or accurate about Tau itself.
KEPT_FILES = {
    "tau_coding/branding.py",  # Tau's own defaults
    "tau_coding/updater.py",  # updates tau-ai from PyPI; fram refuses `update`
}
KEPT_DIRS = ("tau_ai/",)  # provider layer: Codex originator and User-Agent

# Literals kept whole: a short name must not excuse new text that begins with it.
KEPT_EXACT = {
    ("tau_coding/extensions/loader.py", "tau"),  # the extension API's `tau` name
    ("tau_coding/extensions/loader.py", "(tau)` entry point"),  # the extension signature
    ("tau_coding/oauth.py", "tau"),  # OAuth originator
    ("tau_coding/rpc.py", "tau"),  # RPC source identifier
    ("tau_coding/update_check.py", "Tau "),  # tau-ai PyPI notice; fram turns it off
    ("tau_coding/update_check.py", "). Run `tau update` to upgrade."),
}

# Long literals kept by how they begin, as (file, start of the literal).
KEPT_PREFIXES = {
    ("tau_coding/extensions/api.py", "extension instance is stale after reload"),  # API object
    ("tau_coding/rpc.py", "Unhandled Tau session entry"),  # internal assertion
    ("tau_coding/rpc.py", "abort_bash is not supported by Tau"),  # RPC protocol error
    ("tau_coding/session_usage.py", "\n    (function () {"),  # a JS comment
    ("tau_coding/tui/app.py", "\n    Screen {"),  # CSS: $tau-* variables and a comment
}


def _docstring_ids(tree: ast.Module) -> set[int]:
    """Return ids of docstrings, including attribute docstrings after assignments."""
    ids = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for index, statement in enumerate(body):
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                continue
            if index == 0 or isinstance(body[index - 1], ast.Assign | ast.AnnAssign):
                ids.add(id(statement.value))
    return ids


def unbranded_strings() -> list[tuple[str, int, str]]:
    """Return (file, line, literal) for each Tau-naming literal not kept on purpose."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        if relative in KEPT_FILES or relative.startswith(KEPT_DIRS):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = _docstring_ids(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            if id(node) in docstrings or not TAU_NAME.search(node.value):
                continue
            if (relative, node.value) in KEPT_EXACT or any(
                relative == file and node.value.startswith(start) for file, start in KEPT_PREFIXES
            ):
                continue
            found.append((relative, node.lineno, node.value[:80]))
    return found


def test_tau_display_text_reads_the_branding_seam() -> None:
    assert unbranded_strings() == []


def test_the_scan_finds_a_hard_coded_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = tmp_path / "tau_coding" / "new_screen.py"
    module.parent.mkdir()
    module.write_text(
        '"""Mentions Tau in a docstring, which is fine."""\n'
        'MESSAGE = "Tau is ready."\n'
        'USAGE = f"Usage: tau run {1}"\n'
        'HINT = "Set the key, then run tau."\n'
        'SOURCE = "tau.compaction"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path)

    assert sorted((file, line) for file, line, _ in unbranded_strings()) == [
        ("tau_coding/new_screen.py", 2),
        ("tau_coding/new_screen.py", 3),
        ("tau_coding/new_screen.py", 4),
    ]
