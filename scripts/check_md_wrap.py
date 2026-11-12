"""Check, and optionally fix, the wrapping of Markdown prose Fram changed.

Fram's Markdown prose wraps at 80 columns, filled greedily: no line is longer
than 80 characters unless a single word makes it so, and no line ends early when
the next line's first word would have fitted. With one fill rule, anyone who
rewraps a paragraph gets the same lines, and a paragraph edited without being
rewrapped shows up here.

Only paragraphs that differ from the upstream Tau commit Fram last merged are
checked, so upstream's own text, which does not follow this rule everywhere, is
never rewrapped and upstream merges stay clean. A changed paragraph whose
unchanged lines are already longer than 80 columns, or that sits in such a
list, follows upstream's style there and is left alone.

Usage:
    uv run python scripts/check_md_wrap.py            # report, exit 1 if any
    uv run python scripts/check_md_wrap.py --fix      # rewrap them in place
    uv run python scripts/check_md_wrap.py --base REV [PATH ...]

``--base`` defaults to ``git merge-base HEAD upstream/main``. The check reads
the working tree, so uncommitted edits count. Headings, tables, code, HTML,
blockquotes, front matter, files under hidden directories and paragraphs with
hard line breaks are skipped.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

WIDTH = 80

_FENCE = re.compile(r"^\s*(```|~~~)")
_LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d+[.)])(\s+)")
_HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", re.M)
# Text that would turn into markup, or read badly, at the start of a line.
_NO_LINE_START = re.compile(r"^([-*+>|=#]+|\d+[.)]|—)$")
# Inline code spans stay whole; a break inside one is legal but hard to read. A
# backtick with no partner is an ordinary character.
_TOKEN = re.compile(r"(?:`[^`]*`|[^\s`]|`)+")


@dataclass
class Unit:
    """One wrappable run of prose lines: a paragraph or one list item."""

    start: int  # 0-based index of the first line in the file
    lines: list[str]
    first_prefix: str
    rest_prefix: str
    block: range  # the lines of the paragraph or whole list it belongs to


def _skipped(line: str) -> bool:
    stripped = line.strip()
    return (
        not stripped
        or stripped.startswith(("#", "|", ">", "<", "[^", "<!--"))
        or line.endswith(("  ", "\\"))
        or re.fullmatch(r"[-=*_]{3,}", stripped) is not None
        or re.match(r"^\s*\[[^\]]+\]:\s", line) is not None
    )


def units(text: str) -> list[Unit]:
    """Split a Markdown file into the units this check rewraps."""
    lines = text.split("\n")
    found: list[Unit] = []
    i = 0
    if lines and lines[0].strip() in ("---", "+++"):
        fence = lines[0].strip()
        i = next((k + 1 for k in range(1, len(lines)) if lines[k].strip() == fence), len(lines))
    in_fence = False
    block: list[int] = []

    def flush() -> None:
        found.extend(_block_units(lines, block))
        block.clear()

    while i < len(lines):
        line = lines[i]
        if _FENCE.match(line):
            flush()
            in_fence = not in_fence
        elif in_fence or not line.strip():
            flush()
        else:
            block.append(i)
        i += 1
    flush()
    return found


def _block_units(lines: list[str], block: list[int]) -> list[Unit]:
    """Cut a block of consecutive non-blank lines into list items or a paragraph."""
    if not block or any(_skipped(lines[k]) for k in block):
        return []
    groups: list[list[int]] = []
    for k in block:
        if _LIST_ITEM.match(lines[k]) or not groups:
            groups.append([k])
        else:
            groups[-1].append(k)
    result = []
    for group in groups:
        first = lines[group[0]]
        item = _LIST_ITEM.match(first)
        if item:
            first_prefix = item.group(0)
            rest_prefix = " " * len(first_prefix)
        else:
            first_prefix = rest_prefix = first[: len(first) - len(first.lstrip())]
            if len(rest_prefix) >= 4:
                continue  # an indented code block, or nesting this check can't see
        if any(not lines[k].startswith(rest_prefix) for k in group[1:]):
            continue
        span = range(block[0], block[-1] + 1)
        result.append(Unit(group[0], [lines[k] for k in group], first_prefix, rest_prefix, span))
    return result


def fill(unit: Unit, width: int = WIDTH) -> list[str]:
    """The unit's words refilled greedily to ``width`` columns."""
    body = " ".join(
        line[len(unit.first_prefix if n == 0 else unit.rest_prefix) :].strip()
        for n, line in enumerate(unit.lines)
    )
    words = _TOKEN.findall(body)
    out: list[str] = []
    current = unit.first_prefix
    has_word = False
    for word in words:
        fits = len(current) + 1 + len(word) <= width
        if has_word and not fits and not _NO_LINE_START.match(word):
            out.append(current)
            current = unit.rest_prefix + word
        else:
            current = f"{current} {word}" if has_word else current + word
        has_word = True
    out.append(current)
    return out


def _words(lines: list[str]) -> list[str]:
    return " ".join(lines).split()


def changed_lines(base: str, path: str) -> set[int]:
    """0-based line numbers of ``path`` in the working tree that differ from ``base``."""
    diff = subprocess.run(
        ["git", "diff", "-U0", base, "--", path], capture_output=True, text=True, check=True
    ).stdout
    if not diff and _untracked(path):
        return set(range(Path(path).read_text(encoding="utf-8").count("\n") + 1))
    changed: set[int] = set()
    for match in _HUNK.finditer(diff):
        start, count = int(match[1]), int(match[2] or 1)
        if count == 0:
            changed |= {start - 1, start}  # a deletion: the lines on either side
        else:
            changed |= set(range(start - 1, start - 1 + count))
    return changed


def _untracked(path: str) -> bool:
    out = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return bool(out.strip())


def check_file(path: str, changed: set[int], *, fix: bool) -> list[str]:
    """Report, and with ``fix`` rewrite, the changed units that are not filled."""
    text = Path(path).read_text(encoding="utf-8")
    lines = text.split("\n")
    problems = []
    edits: list[tuple[Unit, list[str]]] = []
    for unit in units(text):
        span = range(unit.start, unit.start + len(unit.lines))
        touched = [k for k in span if k in changed]
        if not touched:
            continue
        if any(len(lines[k]) > WIDTH for k in unit.block if k not in changed):
            continue  # upstream left this paragraph or list unwrapped; match it
        filled = fill(unit)
        if filled == unit.lines:
            continue
        assert _words(filled) == _words(unit.lines), (path, unit.start)
        long = [k for k in span if len(lines[k]) > WIDTH]
        where = long[0] if long else touched[0]
        reason = f"line is {len(lines[where])} columns" if long else "lines end early"
        problems.append(f"{path}:{unit.start + 1}: paragraph not wrapped at {WIDTH} ({reason})")
        edits.append((unit, filled))
    if fix and edits:
        for unit, filled in reversed(edits):
            lines[unit.start : unit.start + len(unit.lines)] = filled
        Path(path).write_text("\n".join(lines), encoding="utf-8")
    return problems


def _default_base() -> str:
    try:
        return subprocess.run(
            ["git", "merge-base", "HEAD", "upstream/main"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        sys.exit(
            "check_md_wrap: no upstream/main to compare with. Add the remote "
            "(git remote add upstream https://github.com/huggingface/tau.git; "
            "git fetch upstream main) or pass --base."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", help="Markdown files (default: all changed)")
    parser.add_argument("--base", help="the upstream commit to compare with")
    parser.add_argument("--fix", action="store_true", help="rewrap in place")
    args = parser.parse_args(argv)
    base = args.base or _default_base()
    paths = (
        args.paths
        or subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=AM", base, "--", "*.md"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    )
    problems = []
    for path in paths:
        if not args.paths and any(part.startswith(".") for part in Path(path).parts):
            continue  # tool state and templates, not prose
        if path.endswith(".md") and Path(path).is_file():
            problems += check_file(path, changed_lines(base, path), fix=args.fix)
    for problem in problems:
        print(problem)
    if problems and not args.fix:
        print(f"\n{len(problems)} paragraph(s); rewrap with: scripts/check_md_wrap.py --fix")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
