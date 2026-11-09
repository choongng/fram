"""A cut read's outline lists a document's headings, not only code definitions."""

from __future__ import annotations

from pathlib import Path

import anyio

from tau_agent.messages import TextContent
from tau_agent.tools import AgentToolResult
from tau_coding.line_anchors import anchor_hash
from tau_coding.tool_buffers import BufferLimits, buffer_tool_result, read_first_line
from tau_coding.tools import create_read_tool

LIMITS = BufferLimits(inline_chars=2_000, window_chars=8_000, window_lines=60)

PAPER_SECTIONS = {
    7: "1    Introduction",
    41: "2    Related Work",
    202: "5    Tree Structures",
    272: "5.1     An example",
    287: "5.2     Normal form",
    340: "5.3     Operations over Trees",
    345: "5.3.1    Comparison",
    373: " 5.3.3   Join",
    524: "7                Conclusions",
}


def _text(result: AgentToolResult) -> str:
    return "\n".join(b.text for b in result.content if isinstance(b, TextContent))


def _paper(total: int = 671) -> str:
    """A plain-text paper with numbered sections, with prose between sections."""
    lines = [
        PAPER_SECTIONS.get(n, f"Prose of the paper, line {n}, describing clocks and stamps.")
        for n in range(1, total + 1)
    ]
    return "\n".join(lines) + "\n"


def _execute_read(tmp_path: Path, arguments: dict[str, object]) -> AgentToolResult:
    """Run the read tool once; sync, so each test has one pytest id the check can name."""
    tool = create_read_tool(cwd=tmp_path)
    return anyio.run(tool.execute, "call", arguments)


def _cut_read(tmp_path: Path, name: str, source: str, arguments: dict[str, object]) -> str:
    """Read `name` with the read tool and buffer it as a session does."""
    (tmp_path / name).write_text(source)
    result = _execute_read(tmp_path, {"path": name, **arguments})
    first_line = read_first_line(arguments)
    buffered = buffer_tool_result(
        result,
        tool_name="read",
        buffer_dir=tmp_path / "buffers",
        limits=LIMITS,
        first_line=first_line,
    )
    shown = _text(buffered)
    assert "[Read cut" in shown
    return shown


def _outline(shown: str) -> str:
    return shown.split("Outline (jump with read offset=<line>):", 1)[1]


def test_a_cut_read_of_a_markdown_reference_lists_its_headings(
    tmp_path: Path,
) -> None:
    lines = ["# Tree Structures", ""]
    where: dict[str, int] = {"# Tree Structures": 1}
    for section in range(1, 9):
        heading = f"## Section {section}"
        where[heading] = len(lines) + 1
        lines.extend([heading, ""])
        lines.extend(f"Body text of section {section}, row {k}." for k in range(40))
        sub = f"### Detail {section}"
        where[sub] = len(lines) + 1
        lines.extend([sub, "", "```bash", "# not a heading", "```", ""])
        lines.extend(f"More text of section {section}, row {k}." for k in range(30))
    shown = _cut_read(tmp_path, "paper.md", "\n".join(lines) + "\n", {})

    outline = _outline(shown)
    assert "  L1 # Tree Structures" in outline
    assert f"  L{where['## Section 8']}   ## Section 8" in outline
    assert f"  L{where['### Detail 8']}     ### Detail 8" in outline
    assert where["## Section 8"] > 500
    assert "not a heading" not in outline


def test_a_cut_read_of_a_plain_text_paper_lists_its_numbered_sections(
    tmp_path: Path,
) -> None:
    shown = _cut_read(tmp_path, "paper.txt", _paper(), {})

    kept = shown.split("\n[Read cut")[0]
    last = int(kept.split("\n")[-1].split("#")[0])
    assert last < 340
    outline = _outline(shown)
    assert "  L340   5.3 Operations over Trees" in outline
    assert "  L202 5 Tree Structures" in outline
    assert "  L272   5.1 An example" in outline
    assert "  L345     5.3.1 Comparison" in outline
    assert "  L373     5.3.3 Join" in outline
    assert "  L524 7 Conclusions" in outline
    assert "Prose of the paper" not in outline


def test_numbered_prose_and_list_lines_are_not_sections(tmp_path: Path) -> None:
    lines = [f"Filler prose, line {n}, for the paper body." for n in range(1, 401)]
    lines[9] = "1 Introduction"
    lines[49] = "2 The clock is advanced each time an event is recorded by the replica."
    lines[89] = "3 the stamp is joined with its peer before forking again"
    lines[129] = "4 This sentence runs long enough that it cannot be a title of any section"
    lines[169:172] = ["1. Install it", "2. Run the tests", "3. Read the report"]
    lines[209] = "2024 Annual Report"
    lines[249] = "2 Related Work"
    shown = _cut_read(tmp_path, "notes.txt", "\n".join(lines) + "\n", {})

    outline = _outline(shown)
    assert "  L10 1 Introduction" in outline
    assert "  L250 2 Related Work" in outline
    for text in ["clock is advanced", "stamp is joined", "runs long", "Install", "Annual"]:
        assert text not in outline, text

    markdown = ["# Guide", "", *[f"Text row {n}." for n in range(300)]]
    markdown[150:151] = ["1. Install it", "", "Then configure it.", "", "2. Run it"]
    shown = _cut_read(tmp_path, "guide.md", "\n".join(markdown) + "\n", {})
    outline = _outline(shown)
    assert "  L1 # Guide" in outline
    assert "Install" not in outline and "Run it" not in outline


def test_code_definitions_are_still_listed(tmp_path: Path) -> None:
    lines = ["# A module comment, not a heading", "import os", ""]
    where: dict[str, int] = {}
    for c in range(40):
        where[f"class Thing{c}"] = len(lines) + 1
        lines.append(f"class Thing{c}:")
        where[f"def method_{c}"] = len(lines) + 1
        lines.append(f"    def method_{c}(self) -> int:")
        lines.append("        # 1 Step one of the method")
        lines.extend(f"        x = {k}" for k in range(10))
        lines.append("        return x")
        lines.append("")
    shown = _cut_read(tmp_path, "module.py", "\n".join(lines) + "\n", {})

    outline = shown.split("Definitions (jump with read offset=<line>):", 1)[1]
    assert f"  L{where['class Thing39']} class Thing39" in outline
    assert f"  L{where['def method_39']}   def method_39" in outline
    assert "module comment" not in outline and "Step one" not in outline
    assert "Outline (" not in shown


def test_the_outline_names_line_numbers_a_read_offset_can_use(
    tmp_path: Path,
) -> None:
    source = _paper()
    first = _cut_read(tmp_path, "paper.txt", source, {"offset": 101})
    assert "  L340   5.3 Operations over Trees" in _outline(first)
    assert "  L7 1 Introduction" not in _outline(first)

    (tmp_path / "paper.txt").write_text(source)
    jumped = _execute_read(tmp_path, {"path": "paper.txt", "offset": 340, "limit": 5})
    heading = PAPER_SECTIONS[340]
    assert _text(jumped).startswith(f"340#{anchor_hash(heading)}| {heading}\n")
