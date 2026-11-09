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
    assert (
        buffer_tool_result(result, tool_name="bash", buffer_dir=tmp_path, limits=LIMITS) is result
    )
    assert list(tmp_path.iterdir()) == []


def test_an_oversized_result_keeps_its_tail_and_names_the_full_file(tmp_path: Path) -> None:
    text = "\n".join(f"line {i}" for i in range(500))
    result = AgentToolResult(content=[TextContent(text=text)])

    buffered = buffer_tool_result(result, tool_name="bash", buffer_dir=tmp_path, limits=LIMITS)

    shown = _text(buffered)
    assert "line 499" in shown and "line 0\n" not in shown
    assert "the last 10 of 500 lines" in shown
    [path] = list(tmp_path.iterdir())
    assert str(path) in shown and path.read_text() == text
    again = buffer_tool_result(result, tool_name="bash", buffer_dir=tmp_path, limits=LIMITS)
    assert _text(again) == shown and len(list(tmp_path.iterdir())) == 1


def test_read_keeps_the_head_and_long_lines_are_capped(tmp_path: Path) -> None:
    head = buffer_tool_result(
        AgentToolResult(content=[TextContent(text="\n".join(f"row {i}" for i in range(500)))]),
        tool_name="read",
        buffer_dir=tmp_path,
        limits=LIMITS,
    )
    # A read is windowed by its window's chars, not `window_lines`.
    kept = _text(head).split("\n[Read cut")[0].split("\n")
    assert kept[0] == "row 0" and len(kept) > LIMITS.window_lines
    assert f"lines 1-{len(kept)} of the file are shown" in _text(head)

    one_line = buffer_tool_result(
        AgentToolResult(content=[TextContent(text="x" * 100_000)]),
        tool_name="bash",
        buffer_dir=tmp_path,
        limits=LIMITS,
    )
    assert len(_text(one_line)) < 1_000


def test_an_unwritable_buffer_still_truncates_and_says_why(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    image = ImageContent(data="aGk=", mime_type="image/png")
    result = AgentToolResult(content=[TextContent(text="y\n" * 5_000), image])

    buffered = buffer_tool_result(
        result, tool_name="bash", buffer_dir=blocker / "sub", limits=LIMITS
    )

    assert "could not be saved" in _text(buffered)
    assert buffered.content[-1] == image


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
            tool_name="read",
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

    shown = _text(buffer_tool_result(result, tool_name="read", buffer_dir=tmp_path, limits=limits))

    outline = shown.split("Definitions (", 1)[1]
    assert len(outline) <= limits.window_chars // 4 + 100
    listed = outline.count("\n  L")
    assert 0 < listed <= 80
    assert f"{len(where) - listed:,} more definitions not listed" in outline

    bash = _text(buffer_tool_result(result, tool_name="bash", buffer_dir=tmp_path, limits=limits))
    assert "Definitions (" not in bash
    plain = AgentToolResult(content=[TextContent(text="no code here\n" * 500)])
    assert "Definitions (" not in _text(
        buffer_tool_result(plain, tool_name="read", buffer_dir=tmp_path, limits=limits)
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
            tool_name="read",
            buffer_dir=tmp_path / "buffers",
            limits=BufferLimits(inline_chars=2_000, window_chars=8_000, window_lines=60),
            first_line=first_line,
        )
    )

    kept = shown.split("\n[Read cut")[0]
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
            tool_name="read",
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
            tool_name="read",
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
            tool_name="read",
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
    assert shown.startswith(f"101#{anchor_hash(line_101)}| {line_101}\n")
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
            tool_name="bash",
            buffer_dir=tmp_path,
            limits=limits,
            output_name="bash-c1.txt",
        )
    )

    assert (tmp_path / "bash-c1.txt").read_text() == diff
    assert f"Full output: {tmp_path / 'bash-c1.txt'}" in shown
    assert len(shown) < 10_000
    under = AgentToolResult(content=[TextContent(text=_diff_like(limits.inline_chars))])
    kept = buffer_tool_result(
        under, tool_name="bash", buffer_dir=tmp_path, limits=limits, output_name="bash-c2.txt"
    )
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
    assert session._buffer_result(diff, tool_name="bash") is diff

    monkeypatch.setenv(INLINE_TOKENS_ENV, "500")
    shown = _text(session._buffer_result(diff, tool_name="bash"))
    assert "Full output: " in shown and len(shown) < 2_000
    assert BufferLimits.for_context_window(200_000).inline_chars == 2_000
