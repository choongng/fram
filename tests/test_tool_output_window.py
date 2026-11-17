"""A tool declares which part of an oversized result is worth keeping.

`AgentTool.output_window` is the declaration; the coding session reads it when
a result comes back and cuts the result accordingly. See
dev-notes/tool-result-buffers.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pi_event_helpers import assistant_done, assistant_start
from tau_agent import AgentTool, AgentToolResult, AssistantMessage, TextContent, ToolCall
from tau_agent.messages import assistant_content
from tau_agent.session import JsonlSessionStorage
from tau_agent.tools import ToolOutputWindow
from tau_ai import FakeProvider
from tau_coding import (
    CodingSession,
    CodingSessionConfig,
    OpenAICompatibleProviderConfig,
    ProviderSettings,
)
from tau_coding.extensions.runtime import ExtensionRuntime
from tau_coding.tool_buffers import INLINE_TOKENS_ENV, BufferLimits, buffer_tool_result
from tau_coding.tools import create_coding_tools


def _text(result: AgentToolResult) -> str:
    return "\n".join(b.text for b in result.content if isinstance(b, TextContent))


def _tool(name: str, text: str, output_window: ToolOutputWindow | None = None) -> AgentTool:
    async def run(*_: object) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text=text)])

    declared = {} if output_window is None else {"output_window": output_window}
    return AgentTool(
        name=name,
        label=name,
        description="",
        parameters={"type": "object"},
        execute_fn=run,
        **declared,  # type: ignore[arg-type]
    )


def _status_report(chars: int) -> str:
    """A sub-agent's status report: `##` sections, each a list of findings."""
    sections = ["Crux Check", "Highlights", "Priorities", "Risks", "Open Questions", "Next Steps"]
    lines = ["# Weekly status", ""]
    finding = 0
    while True:
        for title in sections:
            lines += [f"## {title}", ""]
            for _ in range(4):
                finding += 1
                lines.append(f"- Finding {finding}: the check passed and the note is filed.")
            lines.append("")
            if len("\n".join(lines)) >= chars:
                return "\n".join(lines)[:chars]


async def _session_calling(
    tmp_path: Path, tools: list[AgentTool], calls: list[ToolCall], *, context_window: int
) -> tuple[CodingSession, FakeProvider]:
    provider = FakeProvider(
        [
            [
                assistant_start(model="fake"),
                assistant_done(
                    message=AssistantMessage(
                        content=assistant_content("", calls), stop_reason="toolUse"
                    )
                ),
            ],
            [assistant_start(model="fake"), assistant_done(message=AssistantMessage(content="ok"))],
        ]
    )
    settings = ProviderSettings(
        default_provider="local",
        providers=(
            OpenAICompatibleProviderConfig(
                name="local",
                models=("fake",),
                default_model="fake",
                context_windows={"fake": context_window},
            ),
        ),
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Tau.",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path,
            provider_name="local",
            provider_settings=settings,
            tools=tools,
        )
    )
    return session, provider


def _results(provider: FakeProvider) -> dict[str, str]:
    return {m.tool_name: m.text for m in provider.calls[1][2] if m.role == "toolResult"}


def test_tools_keep_both_ends_unless_they_say_otherwise() -> None:
    assert _tool("plain", "").output_window == "ends"
    windows = {tool.name: tool.output_window for tool in create_coding_tools()}
    assert windows.pop("read") == "head"
    assert set(windows.values()) == {"ends"}


def test_a_whole_result_is_shown_whole_up_to_an_eighth_of_the_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(INLINE_TOKENS_ENV, raising=False)
    limits = BufferLimits.for_context_window(32_000)
    assert limits.whole_limit == 32_000 * 4 // 8 == 16_000

    at_ceiling = AgentToolResult(content=[TextContent(text="r" * 16_000)])
    assert (
        buffer_tool_result(at_ceiling, output_window="whole", buffer_dir=tmp_path, limits=limits)
        is at_ceiling
    )
    # The same result from a tool that did not ask is cut.
    assert buffer_tool_result(at_ceiling, buffer_dir=tmp_path, limits=limits) is not at_ceiling

    over = "\n".join(f"line {n}" for n in range(4_000))
    assert len(over) > limits.whole_limit
    cut = buffer_tool_result(
        AgentToolResult(content=[TextContent(text=over)]),
        output_window="whole",
        buffer_dir=tmp_path,
        limits=limits,
    )
    as_ends = buffer_tool_result(
        AgentToolResult(content=[TextContent(text=over)]), buffer_dir=tmp_path, limits=limits
    )
    assert _text(cut) == _text(as_ends) and len(_text(cut)) < limits.inline_chars


def test_a_whole_result_is_still_saved(tmp_path: Path) -> None:
    limits = BufferLimits(inline_chars=100, window_chars=50, whole_chars=1_000)
    report = AgentToolResult(content=[TextContent(text="finding\n" * 100)])

    kept = buffer_tool_result(
        report,
        output_window="whole",
        buffer_dir=tmp_path,
        limits=limits,
        output_name="agent-c1.txt",
    )

    assert kept is report
    assert (tmp_path / "agent-c1.txt").read_text() == "finding\n" * 100


def test_the_whole_ceiling_is_never_under_the_inline_limit(tmp_path: Path) -> None:
    limits = BufferLimits(inline_chars=1_000, window_chars=500)
    assert limits.whole_limit == 1_000
    raised = BufferLimits.for_context_window(8_000, inline_tokens=5_000)
    assert raised.whole_limit == raised.inline_chars == 20_000


@pytest.mark.anyio
async def test_a_status_report_reaches_a_small_model_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent on a 32K-token model once got only the last 2,000 characters of
    a sub-agent's 4,700-character report, never opened the saved file, and made
    its own report up. The report now arrives whole, and so does one several
    times its size."""
    monkeypatch.delenv(INLINE_TOKENS_ENV, raising=False)
    report = _status_report(4_700)
    long_report = _status_report(12_000)
    assert "## Crux Check" in report and "## Next Steps" in long_report
    calls = [
        ToolCall(id="c1", name="subagent", arguments={}),
        ToolCall(id="c2", name="long_subagent", arguments={}),
        ToolCall(id="c3", name="log", arguments={}),
    ]
    session, provider = await _session_calling(
        tmp_path,
        [
            _tool("subagent", report, "whole"),
            _tool("long_subagent", long_report, "whole"),
            _tool("log", long_report),
        ],
        calls,
        context_window=32_000,
    )
    assert len(long_report) > session._buffer_limits().inline_chars

    async for _ in session.prompt("Report on the week."):
        pass

    results = _results(provider)
    assert results["subagent"] == report
    assert results["long_subagent"] == long_report
    # A tool that did not declare it is cut as before, and every one is saved.
    assert len(results["log"]) < len(long_report)
    assert (tmp_path / "session.outputs" / "long_subagent-c2.txt").read_text() == long_report
    assert (tmp_path / "session.outputs" / "log-c3.txt").read_text() == long_report


@pytest.mark.anyio
async def test_a_call_to_an_unknown_tool_gets_the_default_window(tmp_path: Path) -> None:
    session, _ = await _session_calling(
        tmp_path, [_tool("known", "", "whole")], [], context_window=32_000
    )

    assert session._output_window("known") == "whole"
    assert session._output_window("no-such-tool") == "ends"


def test_an_extension_tool_keeps_its_declaration_through_the_hook_wrapper() -> None:
    runtime = ExtensionRuntime()

    [wrapped] = runtime.compose_tools([_tool("report", "", "whole")])

    assert wrapped.output_window == "whole"
