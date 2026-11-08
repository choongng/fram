from __future__ import annotations

from pathlib import Path

import pytest

from tau_agent.messages import ImageContent, TextContent
from tau_agent.session import JsonlSessionStorage
from tau_agent.tools import AgentToolResult
from tau_ai import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig, TauPaths, load_shell_settings
from tau_coding.shell_config import shell_settings_path
from tau_coding.tool_buffers import (
    INLINE_TOKENS_ENV,
    BufferLimits,
    buffer_tool_result,
)

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
