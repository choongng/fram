"""Anchored rows: how `read` numbers the lines of a text file.

Each line of a file is shown as one or more *rows*. A line's first row is

    N#hhh| text

where `N` is the line's 1-based number in the file and `hhh` is its *anchor
hash*: the first `ANCHOR_HEX` hex digits of the sha256 of the line's `key`.
`key` folds what a copy of the line may change without changing what it says:
NFKC normalisation, curly quotes straightened, and whitespace, hyphens and
dashes dropped. So `N#hhh` names a line, and an anchor that was not copied
from a read does not match.

A line longer than `WRAP_WIDTH` characters is wrapped. Its first row holds the
first fragment, and each further fragment gets a continuation row

    N.k| text

numbered by position (`k` = 2, 3, ...) with no hash, since a fragment's hash
would change with the width. A wrap prefers to break before a space near the
width, so a continuation row usually starts with that space; the fragments
joined in order are exactly the line. A line at or under the width is never
wrapped, so code reads as it is, apart from the prefix.

The row prefix (everything up to and including `| `) is not file content.
`strip_row_prefixes` undoes it, so an edit whose old text was copied with its
prefixes still finds the lines it names.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

#: Hex digits of a line's hash in its anchor: enough that a guessed number is
#: refused, few enough to copy without error.
ANCHOR_HEX = 3
#: Lines longer than this many characters are wrapped into continuation rows.
#: 180 keeps nearly all code on one row (a long code line is rarely past 120)
#: while a paper converted to text, whose paragraphs are single lines of up to
#: a few thousand characters, is shown as several rows instead of one.
WRAP_WIDTH = 180
#: How far back from the width a wrap looks for a space to break before.
WRAP_SLACK = 40

#: Characters folded before hashing: curly quotes to straight ones.
_FOLD = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "″": '"',
    }
)
#: Characters dropped before hashing besides whitespace: the hyphens and dashes
#: a line break or a typesetter adds or swaps.
_DROP = frozenset("-­‐‑‒–—―−")

#: A first row's prefix (`12#3fa| `) or a continuation row's (`12.2| `). The
#: space after the bar is optional so a copy that lost a trailing space on an
#: empty line still counts.
_FIRST_ROW = re.compile(r"^(\d+)#[0-9a-fA-F]+\| ?")
_CONTINUATION_ROW = re.compile(r"^(\d+)\.(\d+)\| ?")


def key(text: str) -> str:
    """The form a line is hashed in: NFKC, curly quotes straightened, and
    whitespace, hyphens and dashes dropped."""
    folded = unicodedata.normalize("NFKC", text).translate(_FOLD)
    return "".join(ch for ch in folded if not ch.isspace() and ch not in _DROP)


def anchor_hash(line: str) -> str:
    """The `hhh` of a line's anchor `N#hhh`."""
    return hashlib.sha256(key(line).encode()).hexdigest()[:ANCHOR_HEX]


def wrap_line(line: str, width: int = WRAP_WIDTH) -> list[str]:
    """`line` as fragments of at most `width` chars that join back to it.

    A line at or under `width` is one fragment. A longer one is broken before
    the last space within `WRAP_SLACK` of the width, or at the width when there
    is none (a URL, a run of symbols).
    """
    if len(line) <= width:
        return [line]
    fragments: list[str] = []
    rest = line
    while len(rest) > width:
        cut = rest.rfind(" ", max(1, width - WRAP_SLACK), width + 1)
        if cut <= 0:
            cut = width
        fragments.append(rest[:cut])
        rest = rest[cut:]
    fragments.append(rest)
    return fragments


def line_rows(number: int, line: str, width: int = WRAP_WIDTH) -> list[str]:
    """The rows line `number` (1-based) is shown as."""
    fragments = wrap_line(line, width)
    rows = [f"{number}#{anchor_hash(line)}| {fragments[0]}"]
    rows.extend(f"{number}.{k}| {fragment}" for k, fragment in enumerate(fragments[1:], 2))
    return rows


def anchored_rows(lines: list[str], first_line: int = 1, width: int = WRAP_WIDTH) -> str:
    """`lines`, the first being file line `first_line`, as anchored rows."""
    return "\n".join(
        "\n".join(line_rows(number, line, width)) for number, line in enumerate(lines, first_line)
    )


def row_line_number(row: str) -> tuple[int, bool] | None:
    """The file line a row shows and whether it is the line's first row."""
    match = _FIRST_ROW.match(row)
    if match is not None:
        return int(match[1]), True
    match = _CONTINUATION_ROW.match(row)
    if match is not None:
        return int(match[1]), False
    return None


def strip_row_prefixes(text: str) -> str | None:
    """`text` with its row prefixes removed and wrapped rows rejoined.

    Returns None unless every line of `text` starts with a row prefix: text
    that is only partly prefixed is not a copy of rows, and is left alone. One
    final newline is allowed and kept.
    A continuation row is joined to the row before it without a newline.
    """
    lines: list[str] = []
    body = text.removesuffix("\n")
    for row in body.split("\n"):
        match = _FIRST_ROW.match(row)
        if match is not None:
            lines.append(row[match.end() :])
            continue
        match = _CONTINUATION_ROW.match(row)
        if match is None:
            return None
        if lines:
            lines[-1] += row[match.end() :]
        else:
            lines.append(row[match.end() :])
    return "\n".join(lines) + text[len(body) :]
