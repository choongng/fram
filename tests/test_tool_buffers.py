from __future__ import annotations

from pathlib import Path

import pytest

from pi_event_helpers import assistant_done, assistant_start
from tau_agent import AssistantMessage, ToolCall
from tau_agent.messages import ImageContent, TextContent, assistant_content
from tau_agent.session import JsonlSessionStorage
from tau_agent.tools import AgentToolResult
from tau_ai import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig, TauPaths, load_shell_settings
from tau_coding.line_anchors import anchor_hash
from tau_coding.shell_config import shell_settings_path
from tau_coding.tool_buffers import (
    INLINE_TOKENS_ENV,
    BufferLimits,
    buffer_tool_result,
    read_first_line,
)
from tau_coding.tools import create_read_tool

LIMITS = BufferLimits(inline_chars=1_000, window_chars=500, window_lines=10)


def _text(result: AgentToolResult) -> str:
    return "\n".join(b.text for b in result.content if isinstance(b, TextContent))


def test_a_result_that_fits_is_returned_unchanged(tmp_path: Path) -> None:
    result = AgentToolResult(content=[TextContent(text="short")])
    assert buffer_tool_result(result, buffer_dir=tmp_path, limits=LIMITS) is result
    assert list(tmp_path.iterdir()) == []


def test_an_oversized_result_keeps_both_ends_around_a_marker(tmp_path: Path) -> None:
    text = "\n".join(f"line {i}" for i in range(500))
    result = AgentToolResult(content=[TextContent(text=text)])

    buffered = buffer_tool_result(result, buffer_dir=tmp_path, limits=LIMITS)

    [path] = list(tmp_path.iterdir())
    assert path.read_text() == text
    # 40% of the 10-line window from the start, the rest from the end, the
    # marker where the middle was, and a first line that says there is one.
    omitted = len("\n".join(f"line {i}" for i in range(4, 494)))
    assert _text(buffered).split("\n") == [
        "[Output cut: 500 lines, middle omitted, see marker below.]",
        *[f"line {i}" for i in range(4)],
        f"[\u2026 lines 5\u2013494 omitted ({omitted:,} chars); full output: {path}. "
        "Page it with read offset/limit or grep it.]",
        *[f"line {i}" for i in range(494, 500)],
    ]
    again = buffer_tool_result(result, buffer_dir=tmp_path, limits=LIMITS)
    assert _text(again) == _text(buffered) and len(list(tmp_path.iterdir())) == 1


def test_both_ends_are_cut_between_lines_within_their_share_of_the_window(
    tmp_path: Path,
) -> None:
    # 100-char lines: a 500-char window holds one whole line in its 200-char
    # start (two need 201) and two in its 300-char end.
    lines = [f"{i:03d}" + "x" * 97 for i in range(1, 41)]
    limits = BufferLimits(inline_chars=1_000, window_chars=500, window_lines=60)

    shown = _text(
        buffer_tool_result(
            AgentToolResult(content=[TextContent(text="\n".join(lines))]),
            buffer_dir=tmp_path,
            limits=limits,
        )
    ).split("\n")

    assert shown[1:2] == lines[:1] and shown[3:] == lines[-2:]
    assert shown[2].startswith("[\u2026 lines 2\u201338 omitted (3,736 chars); full output: ")


def test_a_line_too_long_for_its_end_is_shown_in_part(tmp_path: Path) -> None:
    one_line = _text(
        buffer_tool_result(
            AgentToolResult(content=[TextContent(text="a" * 60_000 + "z" * 40_000)]),
            buffer_dir=tmp_path,
            limits=LIMITS,
        )
    ).split("\n")

    assert one_line[0] == "[Output cut: 1 line, middle omitted, see marker below.]"
    assert one_line[1] == "a" * 200 and one_line[3] == "z" * 300
    assert one_line[2].startswith("[\u2026 line 1 omitted in part (99,500 chars); full output: ")

    # A long last line after short ones: the start is whole lines, the end a part.
    text = "\n".join(["first", "second", "j" * 5_000])
    shown = _text(
        buffer_tool_result(
            AgentToolResult(content=[TextContent(text=text)]), buffer_dir=tmp_path, limits=LIMITS
        )
    ).split("\n")
    assert shown[1:3] == ["first", "second"] and shown[4] == "j" * 300
    assert shown[3].startswith("[\u2026 line 3 omitted in part (4,700 chars); ")


def test_the_marker_names_the_sections_the_cut_took_out(tmp_path: Path) -> None:
    sections = ["Summary", "Crux Check", "Highlights", "Priorities", "Risks", "Next Steps"]
    lines: list[str] = []
    for title in sections:
        lines += [f"## {title}", "", *[f"- {title} finding {n}" for n in range(30)], ""]
    limits = BufferLimits(inline_chars=2_000, window_chars=1_000, window_lines=60)

    shown = _text(
        buffer_tool_result(
            AgentToolResult(content=[TextContent(text="\n".join(lines))]),
            buffer_dir=tmp_path,
            limits=limits,
            output_name="report.txt",
        )
    )

    marker = next(line for line in shown.split("\n") if line.startswith("[\u2026"))
    # Each name carries its line in the saved output, for a read offset.
    saved = (tmp_path / "report.txt").read_text().split("\n")
    assert marker.endswith(
        "Omitted sections: Crux Check (L34), Highlights (L67), Priorities (L100), "
        "Risks (L133), Next Steps (L166).]"
    )
    assert saved[33] == "## Crux Check" and saved[165] == "## Next Steps"
    # A section whose heading is still in view is not listed.
    assert "## Summary" in shown and "Summary (L" not in marker
    assert "Next Steps finding 29" in shown


def _marker(text: str, tmp_path: Path, limits: BufferLimits = LIMITS) -> str:
    shown = _text(
        buffer_tool_result(
            AgentToolResult(content=[TextContent(text=text)]), buffer_dir=tmp_path, limits=limits
        )
    )
    return next(line for line in shown.split("\n") if line.startswith("[\u2026"))


def test_a_line_of_code_in_a_report_does_not_hide_its_sections(tmp_path: Path) -> None:
    lines = ["# Report", *["intro"] * 5, "## Findings", *["- a finding"] * 200]
    lines += ["def helper(): pass", "## Risks", *["- a risk"] * 200, "## Done"]

    marker = _marker("\n".join(lines), tmp_path)

    assert marker.endswith("Omitted sections: Findings (L7), Risks (L209).]")
    # Source code with comments is still outlined by its definitions.
    code = ["import os", *["x = 1"] * 10, "# a comment", "def helper():", *["    y = 2"] * 400]
    assert _marker("\n".join(code), tmp_path).endswith("Omitted definitions: def helper (L13).]")


def test_a_fence_opened_before_the_cut_still_covers_the_middle(tmp_path: Path) -> None:
    lines = ["Run this:", "```sh", *["echo step"] * 100, "# not a heading", "```"]
    lines += ["## After the script", *["text"] * 200]

    marker = _marker("\n".join(lines), tmp_path)

    assert marker.endswith("Omitted sections: After the script (L105).]")


def test_headings_with_windows_line_endings_are_named_without_them(tmp_path: Path) -> None:
    lines = ["intro"] * 10 + ["## Findings", *["text"] * 200, "## Risks", *["text"] * 200]

    marker = _marker("\r\n".join(lines), tmp_path)

    assert marker.endswith("Omitted sections: Findings (L11), Risks (L212).]")


def test_the_rest_of_a_line_cut_in_two_is_not_read_as_a_heading(tmp_path: Path) -> None:
    # The start share ends just before "## Not a heading", inside line 1.
    first = "x" * 200 + "## Not a heading"
    lines = [first, *["text"] * 100, "## Real", *["text"] * 200]

    marker = _marker("\n".join(lines), tmp_path)

    assert "omitted in part" in marker
    assert marker.endswith("Omitted sections: Real (L102).]")


def test_a_newline_at_the_end_is_not_the_end_that_is_kept(tmp_path: Path) -> None:
    shown = _text(
        buffer_tool_result(
            AgentToolResult(content=[TextContent(text="x" * 49_999 + "z\n")]),
            buffer_dir=tmp_path,
            limits=LIMITS,
        )
    )

    top, head, marker, tail, closing = shown.split("\n")
    assert top == "[Output cut: 1 line, middle omitted, see marker below.]"
    assert head == "x" * 200 and tail == "x" * 299 + "z" and closing == ""
    assert marker.startswith("[\u2026 line 1 omitted in part (49,500 chars); ")
    # The count is of lines, in a tail window's marker too.
    lines = _text(
        buffer_tool_result(
            AgentToolResult(content=[TextContent(text="line\n" * 500)]),
            output_window="tail",
            buffer_dir=tmp_path,
            limits=LIMITS,
        )
    )
    assert lines.startswith("[Output cut: 500 lines, start omitted")
    assert " of 500 lines " in lines


def test_the_list_of_omitted_sections_is_bounded(tmp_path: Path) -> None:
    lines: list[str] = []
    for n in range(400):
        lines += [f"## Section number {n}", "text"]
    limits = BufferLimits(inline_chars=2_000, window_chars=1_000, window_lines=60)

    shown = _text(
        buffer_tool_result(
            AgentToolResult(content=[TextContent(text="\n".join(lines))]),
            buffer_dir=tmp_path,
            limits=limits,
        )
    )

    marker = next(line for line in shown.split("\n") if line.startswith("[\u2026"))
    listed = marker.split("Omitted sections: ", 1)[1]
    assert len(listed) <= limits.window_chars // 4
    named = listed.count("Section number")
    assert 0 < named < 80
    assert listed.endswith(f", and {400 - 30 - named:,} more.]")
    # Code in the cut is named by its definitions, numbered by its place in
    # the output, not by a number the line happens to start with.
    code = "\n".join(["x = 1"] * 40 + ["7\tdef hidden():", "    pass"] + ["y = 2"] * 400)
    shown = _text(
        buffer_tool_result(
            AgentToolResult(content=[TextContent(text=code)]), buffer_dir=tmp_path, limits=limits
        )
    )
    assert "Omitted definitions: def hidden (L41).]" in shown


def test_a_tail_window_keeps_the_end_and_says_so_first(tmp_path: Path) -> None:
    text = "\n".join(f"line {i}" for i in range(500))
    result = AgentToolResult(content=[TextContent(text=text)])

    shown = _text(
        buffer_tool_result(result, output_window="tail", buffer_dir=tmp_path, limits=LIMITS)
    )

    [path] = list(tmp_path.iterdir())
    assert shown.split("\n") == [
        "[Output cut: 500 lines, start omitted, see marker below.]",
        *[f"line {i}" for i in range(490, 500)],
        f"[Output truncated: the last 10 of 500 lines (89 of {len(text):,} chars) are shown. "
        f"Full output: {path}. Page it with read offset/limit or grep it.]",
    ]


def test_a_result_the_window_holds_whole_is_left_alone(tmp_path: Path) -> None:
    # Only a window set larger than the inline limit can hold all of a result
    # that is over it; there is then nothing to leave out.
    limits = BufferLimits(inline_chars=100, window_chars=10_000, window_lines=60)
    result = AgentToolResult(content=[TextContent(text="\n".join(["short line"] * 30))])

    assert buffer_tool_result(result, buffer_dir=tmp_path, limits=limits) is result


def test_read_keeps_the_head_and_long_lines_are_capped(tmp_path: Path) -> None:
    head = buffer_tool_result(
        AgentToolResult(content=[TextContent(text="\n".join(f"row {i}" for i in range(500)))]),
        output_window="head",
        buffer_dir=tmp_path,
        limits=LIMITS,
    )
    # A read is windowed by its window's chars, not `window_lines`.
    top, *kept = _text(head).split("\n[Read cut")[0].split("\n")
    assert top == "[Output cut: 500 lines, end omitted, see marker below.]"
    assert kept[0] == "row 0" and len(kept) > LIMITS.window_lines
    assert f"lines 1-{len(kept)} of the file are shown" in _text(head)

    one_line = buffer_tool_result(
        AgentToolResult(content=[TextContent(text="x" * 100_000)]),
        buffer_dir=tmp_path,
        limits=LIMITS,
    )
    assert len(_text(one_line)) < 1_000


def test_an_unwritable_buffer_still_truncates_and_says_why(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    image = ImageContent(data="aGk=", mime_type="image/png")
    result = AgentToolResult(content=[TextContent(text="y\n" * 5_000), image])

    buffered = buffer_tool_result(result, buffer_dir=blocker / "sub", limits=LIMITS)

    shown = _text(buffered).split("\n")
    assert shown[0] == "[Output cut: 5,000 lines, middle omitted, see marker below.]"
    assert shown[1:5] == ["y"] * 4 and shown[6:] == ["y"] * 6 + [""]
    assert shown[5].startswith(
        "[\u2026 lines 5\u20134,994 omitted (9,979 chars); "
        "the full output could not be saved (NotADirectoryError: "
    )
    assert shown[5].endswith(").]")
    assert buffered.content[-1] == image

    tail = buffer_tool_result(
        result, output_window="tail", buffer_dir=blocker / "sub", limits=LIMITS
    )
    assert "are shown. The full output could not be saved (NotADirectoryError: " in _text(tail)


def _python_source(classes: int) -> tuple[str, dict[str, int]]:
    """A synthetic module and the 1-indexed file line of each definition."""
    lines = ['"""Synthetic module."""', "", "import os", ""]
    where: dict[str, int] = {}
    for c in range(classes):
        where[f"class Thing{c}"] = len(lines) + 1
        lines.append(f"class Thing{c}:")
        lines.append(f'    """Thing number {c}."""')
        lines.append("")
        where[f"def method_{c}"] = len(lines) + 1
        lines.append(f"    def method_{c}(self, value: int) -> int:")
        lines.extend(f"        value += {k}" for k in range(8))
        lines.append("        return value")
        lines.append("")
        where[f"async def fetch_{c}"] = len(lines) + 1
        lines.append(f"    async def fetch_{c}(self) -> None:")
        lines.extend(f"        await os.sleep({k})" for k in range(4))
        lines.append("")
        where[f"def helper_{c}"] = len(lines) + 1
        lines.append(f"def helper_{c}() -> int:")
        lines.extend(f"    x = {k}" for k in range(5))
        lines.append("    return x")
        lines.append("")
    return "\n".join(lines) + "\n", where


async def _read(tmp_path: Path, source: str, arguments: dict[str, object]) -> AgentToolResult:
    (tmp_path / "module.py").write_text(source)
    return await create_read_tool(cwd=tmp_path).execute("call", {"path": "module.py", **arguments})


@pytest.mark.anyio
async def test_cut_read_of_python_lists_definitions_with_line_numbers(tmp_path: Path) -> None:
    source, where = _python_source(classes=20)
    assert 500 < source.count("\n") < 600
    limits = BufferLimits(inline_chars=2_000, window_chars=8_000, window_lines=60)
    buffers = tmp_path / "buffers"

    for arguments in ({}, {"offset": 101}):
        result = await _read(tmp_path, source, arguments)
        buffered = buffer_tool_result(
            result,
            output_window="head",
            buffer_dir=buffers,
            limits=limits,
            first_line=read_first_line(arguments),
        )
        shown = _text(buffered)
        outline = shown.split("Definitions (", 1)[1]
        late = [name for name, line in where.items() if line > 200]
        for name in [
            "class Thing7",
            "  def method_7",
            "  async def fetch_7",
            "def helper_15",
            late[0],
        ]:
            line = where[name.strip()]
            assert f"L{line} {name}" in outline, (arguments, name, line)
        assert where["class Thing7"] > 60
        if arguments:
            assert "L5 class Thing0" not in outline
        else:
            assert "L5 class Thing0" in outline


def test_cut_read_outline_is_bounded(tmp_path: Path) -> None:
    source, where = _python_source(classes=200)
    result = AgentToolResult(content=[TextContent(text=source)])
    limits = BufferLimits(inline_chars=2_000, window_chars=4_000, window_lines=60)

    shown = _text(
        buffer_tool_result(result, output_window="head", buffer_dir=tmp_path, limits=limits)
    )

    outline = shown.split("Definitions (", 1)[1]
    assert len(outline) <= limits.window_chars // 4 + 100
    listed = outline.count("\n  L")
    assert 0 < listed <= 80
    assert f"{len(where) - listed:,} more definitions not listed" in outline

    bash = _text(buffer_tool_result(result, buffer_dir=tmp_path, limits=limits))
    assert "Definitions (" not in bash
    plain = AgentToolResult(content=[TextContent(text="no code here\n" * 500)])
    assert "Definitions (" not in _text(
        buffer_tool_result(plain, output_window="head", buffer_dir=tmp_path, limits=limits)
    )


@pytest.mark.anyio
async def test_cut_read_marker_names_file_lines_and_next_offset(tmp_path: Path) -> None:
    source = "".join(f"row {i}\n" for i in range(1, 2_001))
    arguments: dict[str, object] = {"offset": 100}
    result = await _read(tmp_path, source, arguments)
    first_line = read_first_line(arguments)

    shown = _text(
        buffer_tool_result(
            result,
            output_window="head",
            buffer_dir=tmp_path / "buffers",
            limits=BufferLimits(inline_chars=2_000, window_chars=8_000, window_lines=60),
            first_line=first_line,
        )
    )

    top, kept = shown.split("\n[Read cut")[0].split("\n", 1)
    assert top == "[Output cut: 1,901 lines, end omitted, see marker below.]"
    assert kept.startswith(f"100#{anchor_hash('row 100')}| row 100\n")
    assert len(kept) <= 8_000 and kept.count("\n") + 1 > 60
    last = int(kept.split("\n")[-1].split("#")[0])
    assert f"lines 100-{last} of the file are shown (of 100-2000 returned)" in shown
    assert f"continue with offset={last + 1}" in shown
    [path] = list((tmp_path / "buffers").iterdir())
    assert f"Full output: {path}" in shown


def test_char_cut_read_marker_names_the_last_whole_line(tmp_path: Path) -> None:
    text = "\n".join(f"{i:04d}" + "x" * 95 for i in range(1, 101))
    limits = BufferLimits(inline_chars=1_000, window_chars=450, window_lines=60)

    shown = _text(
        buffer_tool_result(
            AgentToolResult(content=[TextContent(text=text)]),
            output_window="head",
            buffer_dir=tmp_path,
            limits=limits,
            first_line=11,
        )
    )

    # 450 chars are four whole 100-char lines and part of a fifth.
    assert "lines 11-14 of the file are shown (of 11-110 returned)" in shown
    assert "continue with offset=15" in shown


@pytest.mark.anyio
async def test_read_with_explicit_limit_fills_the_char_window(tmp_path: Path) -> None:
    source = "".join(f"r{i}\n" for i in range(1, 1_001))
    arguments: dict[str, object] = {"offset": 1, "limit": 300}
    result = await _read(tmp_path, source, arguments)
    assert "more lines in file" in _text(result)
    first_line = read_first_line(arguments)

    shown = _text(
        buffer_tool_result(
            result,
            output_window="head",
            buffer_dir=tmp_path / "buffers",
            limits=BufferLimits(inline_chars=1_000, window_chars=8_000, window_lines=60),
            first_line=first_line,
        )
    )

    rows = [line for line in shown.split("\n") if "| r" in line]
    assert len(rows) == 300 > 60
    assert "lines 1-300 of the file are shown (of 1-300 returned)" in shown
    assert "continue with offset=301" in shown


def test_outline_reads_line_numbers_from_a_prefixed_read(tmp_path: Path) -> None:
    text = "\n".join(["17\tdef f():", *[f"{n}\t    pass" for n in range(18, 400)]])

    shown = _text(
        buffer_tool_result(
            AgentToolResult(content=[TextContent(text=text)]),
            output_window="head",
            buffer_dir=tmp_path,
            limits=LIMITS,
        )
    )

    assert "  L17 def f" in shown.split("Definitions (", 1)[1]


@pytest.mark.anyio
async def test_cut_read_through_session_outline_carries_file_line_numbers(
    tmp_path: Path,
) -> None:
    source, where = _python_source(classes=200)
    (tmp_path / "module.py").write_text(source)
    call = ToolCall(id="call-1", name="read", arguments={"path": "module.py", "offset": 101})
    provider = FakeProvider(
        [
            [
                assistant_start(model="fake"),
                assistant_done(
                    message=AssistantMessage(
                        content=assistant_content("Reading.", [call]), stop_reason="toolUse"
                    )
                ),
            ],
            [
                assistant_start(model="fake"),
                assistant_done(message=AssistantMessage(content="Done")),
            ],
        ]
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
        )
    )

    async for _ in session.prompt("Read it."):
        pass

    [result] = [m for m in provider.calls[1][2] if m.role == "toolResult"]
    shown = result.text
    line_101 = source.split("\n")[100]
    assert shown.split("\n", 1)[1].startswith(f"101#{anchor_hash(line_101)}| {line_101}\n")
    last = int(shown.split("\n[Read cut")[0].split("\n")[-1].split("#")[0])
    assert last > 160
    assert f"lines 101-{last} of the file are shown" in shown
    assert f"continue with read offset={last + 1}" in shown
    outline = shown.split("Definitions (", 1)[1]
    late = next(name for name, line in where.items() if line > 400 and "class" in name)
    assert f"L{where[late]} {late}" in outline
    assert "L5 class Thing0" not in outline


def _diff_like(chars: int) -> str:
    line = "+    value = compute(value)  # a changed line of a diff\n"
    return (line * (chars // len(line) + 1))[:chars]


def test_default_inline_limit_buffers_a_26k_char_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(INLINE_TOKENS_ENV, raising=False)
    limits = BufferLimits.for_context_window(200_000)
    diff = _diff_like(26_000)

    shown = _text(
        buffer_tool_result(
            AgentToolResult(content=[TextContent(text=diff)]),
            buffer_dir=tmp_path,
            limits=limits,
            output_name="bash-c1.txt",
        )
    )

    assert (tmp_path / "bash-c1.txt").read_text() == diff
    assert f"full output: {tmp_path / 'bash-c1.txt'}" in shown
    assert len(shown) < 10_000
    under = AgentToolResult(content=[TextContent(text=_diff_like(limits.inline_chars))])
    kept = buffer_tool_result(under, buffer_dir=tmp_path, limits=limits, output_name="bash-c2.txt")
    assert kept is under
    assert (tmp_path / "bash-c2.txt").read_text() == _text(under)


@pytest.mark.anyio
async def test_inline_limit_is_configurable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(INLINE_TOKENS_ENV, raising=False)
    paths = TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents")
    settings_file = shell_settings_path(paths)
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text('{"toolResultInlineTokens": 10000}', encoding="utf-8")
    settings = load_shell_settings(paths)
    assert settings.tool_result_inline_tokens == 10_000

    session = await CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider([]),
            model="fake",
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            tool_result_inline_tokens=settings.tool_result_inline_tokens,
        )
    )
    diff = AgentToolResult(content=[TextContent(text=_diff_like(26_000))])
    assert session._buffer_result(diff) is diff

    monkeypatch.setenv(INLINE_TOKENS_ENV, "500")
    shown = _text(session._buffer_result(diff))
    assert "full output: " in shown and len(shown) < 2_000
    assert BufferLimits.for_context_window(200_000).inline_chars == 2_000


def test_the_inline_floor_is_2000_tokens_and_less_on_a_small_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(INLINE_TOKENS_ENV, raising=False)
    inline = {
        window: BufferLimits.for_context_window(window).inline_chars // 4
        for window in (4_000, 8_000, 16_000, 24_000, 32_000, 100_000, 150_000, 1_000_000)
    }
    assert inline == {
        # 1/16 of a small window, and never under 1,000 tokens...
        4_000: 1_000,
        8_000: 1_000,
        16_000: 1_000,
        24_000: 1_500,
        # ...2,000 from 32K up, then 1/50 of the window up to 4,000.
        32_000: 2_000,
        100_000: 2_000,
        150_000: 3_000,
        1_000_000: 4_000,
    }
    limits = BufferLimits.for_context_window(32_000)
    assert (limits.inline_chars, limits.window_chars) == (8_000, 4_000)
