"""Every tool result is saved to its own greppable file beside the session.

See dev-notes/tool-result-buffers.md.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from pi_event_helpers import assistant_done, assistant_start
from tau_agent import AgentTool, AgentToolResult, AssistantMessage, TextContent, ToolCall
from tau_agent.messages import assistant_content
from tau_agent.session import JsonlSessionStorage
from tau_ai import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig
from tau_coding.tool_buffers import (
    BufferLimits,
    buffer_tool_result,
    default_buffer_dir,
    output_file_name,
    session_outputs_dir,
)


def _echo_tool(text: str) -> AgentTool:
    async def run(*_: object) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text=text)])

    return AgentTool(
        name="echo", label="Echo", description="", parameters={"type": "object"}, execute_fn=run
    )


def _turn_calling(call: ToolCall) -> list[object]:
    return [
        assistant_start(model="fake"),
        assistant_done(
            message=AssistantMessage(content=assistant_content("", [call]), stop_reason="toolUse")
        ),
    ]


def _done() -> list[object]:
    return [assistant_start(model="fake"), assistant_done(message=AssistantMessage(content="ok"))]


async def _run(session: CodingSession, prompt: str) -> None:
    async for _ in session.prompt(prompt):
        pass


def test_output_file_name_is_derived_from_the_call_and_safe() -> None:
    assert output_file_name("bash", "toolu_01Abc") == "bash-toolu_01Abc.txt"
    assert output_file_name("mcp/fetch", "../x y") == "mcp_fetch-x_y.txt"
    assert output_file_name("", "") == "tool-call.txt"


def test_outputs_dir_sits_beside_the_session_file(tmp_path: Path) -> None:
    assert session_outputs_dir(tmp_path / "abc.jsonl") == tmp_path / "abc.outputs"


def test_default_buffer_dir_is_per_user() -> None:
    assert default_buffer_dir().name != "tau-buffers"
    assert default_buffer_dir().name.startswith("tau-buffers-")


@pytest.mark.anyio
async def test_a_small_tool_result_is_saved_and_left_inline(tmp_path: Path) -> None:
    call = ToolCall(id="call-7", name="echo", arguments={})
    provider = FakeProvider([_turn_calling(call), _done()])
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="s",
            storage=JsonlSessionStorage(tmp_path / "s.jsonl"),
            cwd=tmp_path,
            tools=[_echo_tool("needle in a small output")],
        )
    )

    await _run(session, "go")

    saved = tmp_path / "s.outputs" / "echo-call-7.txt"
    assert saved.read_text() == "needle in a small output"
    assert stat.S_IMODE(saved.parent.stat().st_mode) == 0o700
    [result] = [m for m in provider.calls[1][2] if m.role == "toolResult"]
    assert result.text == "needle in a small output"
    assert session.outputs_dir == tmp_path / "s.outputs"


def test_an_unwritable_outputs_dir_does_not_fail_a_small_result(tmp_path: Path) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("a file, not a directory")
    result = AgentToolResult(content=[TextContent(text="fine")])

    kept = buffer_tool_result(
        result,
        tool_name="echo",
        buffer_dir=blocker,
        limits=BufferLimits.for_context_window(200_000),
        output_name="echo-1.txt",
    )

    assert kept is result


def test_an_oversized_result_names_its_per_call_file(tmp_path: Path) -> None:
    text = "\n".join(f"line {i}" for i in range(50_000))
    result = AgentToolResult(content=[TextContent(text=text)])

    shown = buffer_tool_result(
        result,
        tool_name="bash",
        buffer_dir=tmp_path,
        limits=BufferLimits.for_context_window(200_000),
        output_name="bash-c1.txt",
    )

    assert (tmp_path / "bash-c1.txt").read_text() == text
    assert f"Full output: {tmp_path / 'bash-c1.txt'}" in shown.content[0].text  # type: ignore[union-attr]
    assert list(tmp_path.iterdir()) == [tmp_path / "bash-c1.txt"]


def test_a_reused_call_id_does_not_overwrite_the_earlier_output(tmp_path: Path) -> None:
    limits = BufferLimits.for_context_window(200_000)
    for text in ("first", "second"):
        buffer_tool_result(
            AgentToolResult(content=[TextContent(text=text)]),
            tool_name="bash",
            buffer_dir=tmp_path,
            limits=limits,
            output_name="bash-call_0.txt",
        )

    assert (tmp_path / "bash-call_0.txt").read_text() == "first"
    assert (tmp_path / "bash-call_0-2.txt").read_text() == "second"


def test_a_symlinked_buffer_directory_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = tmp_path / "buffers"
    link.symlink_to(target)
    text = "x" * 200_000

    shown = buffer_tool_result(
        AgentToolResult(content=[TextContent(text=text)]),
        tool_name="bash",
        buffer_dir=link,
        limits=BufferLimits.for_context_window(200_000),
        output_name="bash-c1.txt",
    )

    assert list(target.iterdir()) == []
    assert "could not be saved" in shown.content[0].text  # type: ignore[union-attr]


def test_a_tail_cut_inside_a_line_does_not_count_the_fragment(tmp_path: Path) -> None:
    text = "\n".join("y" * 300 for _ in range(1_000))
    limits = BufferLimits(inline_chars=100, window_chars=500, window_lines=60)

    shown = buffer_tool_result(
        AgentToolResult(content=[TextContent(text=text)]),
        tool_name="bash",
        buffer_dir=tmp_path,
        limits=limits,
    )

    assert "the last 1 of 1,000 lines" in shown.content[0].text  # type: ignore[union-attr]
