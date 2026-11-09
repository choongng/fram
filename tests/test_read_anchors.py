"""`read` prints anchored rows that name each line of a file.

Each file line is `N#hhh| text` (N its line number, hhh its anchor hash); a
line over `WRAP_WIDTH` chars is
wrapped into `N.k| ` continuation rows. See dev-notes/tool-result-buffers.md.
"""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

import pytest

from tau_agent.messages import TextContent
from tau_agent.tools import AgentToolResult
from tau_coding.line_anchors import WRAP_WIDTH, anchor_hash, strip_row_prefixes
from tau_coding.tool_buffers import WINDOW_LINES, BufferLimits, buffer_tool_result
from tau_coding.tools import create_edit_tool, create_read_tool


@pytest.fixture
def anyio_backend() -> str:
    """One backend, unparametrized, so each test's node id is its plain name."""
    return "asyncio"


def _expected_hash(line: str) -> str:
    """The anchor hash, written out from its definition."""
    folded = unicodedata.normalize("NFKC", line).translate(
        str.maketrans({"’": "'", "“": '"', "”": '"'})
    )
    kept = "".join(ch for ch in folded if not ch.isspace() and ch not in "-–—")
    return hashlib.sha256(kept.encode()).hexdigest()[:3]


def _text(result: AgentToolResult) -> str:
    return "\n".join(b.text for b in result.content if isinstance(b, TextContent))


@pytest.mark.anyio
async def test_read_rows_carry_line_number_and_hash(tmp_path: Path) -> None:
    lines = ["def f():", "    return “it’s”", "", "x = a - b"]
    (tmp_path / "m.py").write_text("\n".join(lines) + "\n")

    result = await create_read_tool(cwd=tmp_path).execute("c", {"path": "m.py"})

    rows = result.text.split("\n")
    assert rows == [f"{n}#{_expected_hash(line)}| {line}" for n, line in enumerate(lines, 1)]
    # The hash ignores spacing, quote style and dashes.
    assert anchor_hash('    return "it\'s"') == anchor_hash(lines[1])
    assert anchor_hash("x = a  b") == anchor_hash("x = a — b")


@pytest.mark.anyio
async def test_long_lines_wrap_into_numbered_continuation_rows(tmp_path: Path) -> None:
    long_line = " ".join(f"word{i:03d}" for i in range(100))  # 799 chars
    (tmp_path / "paper.txt").write_text(f"Title\n{long_line}\nEnd\n")

    result = await create_read_tool(cwd=tmp_path).execute("c", {"path": "paper.txt"})

    rows = result.text.split("\n")
    assert rows[0].startswith("1#") and rows[-1].startswith("3#")
    wrapped = rows[1:-1]
    assert wrapped[0].startswith(f"2#{_expected_hash(long_line)}| word000 ")
    assert [row.split("|")[0] for row in wrapped[1:]] == [
        f"2.{k}" for k in range(2, len(wrapped) + 1)
    ]
    fragments = [row.split("| ", 1)[1] for row in wrapped]
    assert all(len(fragment) <= WRAP_WIDTH for fragment in fragments)
    # Breaks fall before a space, and the fragments join back to the line.
    assert all(fragment.startswith(" ") for fragment in fragments[1:])
    assert "".join(fragments) == long_line
    assert strip_row_prefixes("\n".join(wrapped)) == long_line


@pytest.mark.anyio
async def test_short_lines_are_never_wrapped(tmp_path: Path) -> None:
    exact = "y" * WRAP_WIDTH
    code = "    result = compute(alpha, beta, gamma)  # a comment that runs on a while"
    (tmp_path / "c.py").write_text(f"{exact}\n{code}\n")

    result = await create_read_tool(cwd=tmp_path).execute("c", {"path": "c.py"})

    assert result.text == (f"1#{_expected_hash(exact)}| {exact}\n2#{_expected_hash(code)}| {code}")
    assert ".2| " not in result.text


@pytest.mark.anyio
async def test_offset_and_limit_count_original_lines(tmp_path: Path) -> None:
    long_line = "lorem ipsum " * 60
    lines = [f"line {n}" if n != 3 else long_line for n in range(1, 11)]
    (tmp_path / "t.txt").write_text("\n".join(lines) + "\n")
    tool = create_read_tool(cwd=tmp_path)

    result = await tool.execute("c", {"path": "t.txt", "offset": 3, "limit": 2})

    body, note = result.text.split("\n\n")
    numbers = [row.split("|")[0] for row in body.split("\n")]
    assert numbers[0] == f"3#{_expected_hash(long_line)}"
    assert numbers[1:-1] == [f"3.{k}" for k in range(2, len(numbers))]
    assert numbers[-1] == f"4#{_expected_hash('line 4')}"
    # Ten lines in the file (the final newline starts no eleventh), two shown.
    assert note == "[6 more lines in file. Use offset=5 to continue.]"


@pytest.mark.anyio
async def test_edit_accepts_old_text_copied_with_row_prefixes(tmp_path: Path) -> None:
    long_line = "The quick brown fox " * 15
    path = tmp_path / "f.py"
    path.write_text(f"a = 1\nb = 2\n{long_line}\nc = 3\n")
    shown = (await create_read_tool(cwd=tmp_path).execute("c", {"path": "f.py"})).text
    rows = shown.split("\n")
    edit = create_edit_tool(cwd=tmp_path)

    # Old text copied with prefixes, new text written plain.
    await edit.execute(
        "e1", {"path": "f.py", "edits": [{"oldText": "\n".join(rows[:2]), "newText": "a = 10"}]}
    )
    assert path.read_text() == f"a = 10\n{long_line}\nc = 3\n"

    # A wrapped line copied as its rows, new text prefixed on every line too.
    wrapped = "\n".join(row for row in rows if row.startswith("3"))
    await edit.execute(
        "e2",
        {"path": "f.py", "edits": [{"oldText": wrapped, "newText": "2#abc| short = True"}]},
    )
    assert path.read_text() == "a = 10\nshort = True\nc = 3\n"

    # New text only partly prefixed is written as given.
    await edit.execute(
        "e3",
        {"path": "f.py", "edits": [{"oldText": rows[-1], "newText": "c = 4\n9#000| kept"}]},
    )
    assert path.read_text() == "a = 10\nshort = True\nc = 4\n9#000| kept\n"

    # Text that matches neither way keeps the existing message.
    with pytest.raises(Exception, match="Could not find the exact text"):
        await edit.execute(
            "e4", {"path": "f.py", "edits": [{"oldText": "1#fff| zzz", "newText": "q"}]}
        )


def test_an_oversized_read_is_windowed_by_size_not_line_count(tmp_path: Path) -> None:
    limits = BufferLimits(inline_chars=4_000, window_chars=3_000)
    short = "\n".join(f"{n}#abc| x" for n in range(1, 2_001))
    result = buffer_tool_result(
        AgentToolResult(content=[TextContent(text=short)]),
        tool_name="read",
        buffer_dir=tmp_path,
        limits=limits,
    )
    shown, marker = _text(result).split("\n[Read cut: ")
    # Short rows: far more than WINDOW_LINES of them fit the window's chars.
    assert shown.count("\n") + 1 > WINDOW_LINES
    assert len(shown) <= limits.window_chars
    last = shown.split("\n")[-1].split("#")[0]
    assert marker.startswith(f"lines 1-{last} of the file are shown (of 1-2000 returned)")
    assert f"continue with offset={int(last) + 1}." in marker

    # A wrapped line is not split between the window and what is left out.
    wrapped = "\n".join(
        f"{n}#abc| {'p' * 100}\n{n}.2| {'q' * 100}\n{n}.3| {'r' * 100}" for n in range(1, 50)
    )
    result = buffer_tool_result(
        AgentToolResult(content=[TextContent(text=wrapped)]),
        tool_name="read",
        buffer_dir=tmp_path,
        limits=limits,
    )
    shown, marker = _text(result).split("\n[Read cut: ")
    assert shown.split("\n")[-1].startswith(tuple(f"{n}.3| " for n in range(1, 50)))
    last = int(shown.split("\n")[-1].split(".")[0])
    assert marker.startswith(f"lines 1-{last} of the file are shown")


def test_a_cut_read_sends_the_model_back_to_the_source_file(tmp_path: Path) -> None:
    text = "\n".join(f"{n}#abc| line {n}" for n in range(1, 2_001))

    shown = buffer_tool_result(
        AgentToolResult(content=[TextContent(text=text)]),
        tool_name="read",
        buffer_dir=tmp_path,
        limits=BufferLimits(inline_chars=100, window_chars=500, window_lines=60),
        source_path="/work/big.py",
        output_name="read-c1.txt",
    ).content[0]

    assert isinstance(shown, TextContent)
    assert " on /work/big.py.]" in shown.text
    assert "Full output" not in shown.text


@pytest.mark.anyio
async def test_a_line_too_long_for_the_window_is_read_past(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text(f"short1\nshort2\n{'x' * 20_000}\nafter\n")
    read = await create_read_tool(cwd=tmp_path).execute("c", {"path": "a.jsonl", "offset": 3})

    shown = _text(
        buffer_tool_result(
            read,
            tool_name="read",
            buffer_dir=tmp_path,
            limits=BufferLimits(inline_chars=16_000, window_chars=8_000),
            source_path=str(tmp_path / "a.jsonl"),
            first_line=3,
        )
    )

    marker = shown.split("[Read cut: ")[1]
    assert marker.startswith("line 3 is too long to show whole; its first ")
    # Going on from line 3 would show the same part again.
    assert "continue with read offset=4 on " in marker
    seen = int(marker.split("its first ")[1].split(" characters")[0].replace(",", ""))
    assert 0 < seen < 20_000
    assert f"`sed -n '3p' {tmp_path / 'a.jsonl'} | cut -b {seen + 1}-`" in marker


def test_row_prefixes_come_off_text_ending_in_a_newline() -> None:
    assert strip_row_prefixes("1#abc| a = 1\n2#def| b = 2\n") == "a = 1\nb = 2\n"
