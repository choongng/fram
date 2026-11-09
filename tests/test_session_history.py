"""Stable #N entry references and the history tool.

See dev-notes/session-recall-and-continuation-compaction.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pi_event_helpers import assistant_done, assistant_start
from tau_agent import AgentTool, AgentToolResult, AssistantMessage, TextContent, ToolCall
from tau_agent.messages import ToolResultMessage, UserMessage, assistant_content
from tau_agent.session import CompactionEntry, JsonlSessionStorage, MessageEntry
from tau_ai import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig
from tau_coding.session_recall import (
    NOTE_CHARS,
    NumberedEntry,
    history_text,
    numbered_branch,
    parse_refs,
)
from tau_coding.tool_buffers import INLINE_TOKENS_ENV


def _entries() -> list[MessageEntry]:
    call = ToolCall(id="c1", name="bash", arguments={"command": "uv run pytest -q"})
    user = MessageEntry(message=UserMessage(content="fix the resume path"))
    assistant = MessageEntry(
        parent_id=user.id,
        message=AssistantMessage(content=assistant_content("Running tests.", [call])),
    )
    result = MessageEntry(
        parent_id=assistant.id,
        message=ToolResultMessage(
            tool_call_id="c1", tool_name="bash", content="FAILED test_resume", is_error=True
        ),
    )
    return [user, assistant, result]


def test_parse_refs_reads_single_ranges_and_lists() -> None:
    assert parse_refs("#12") == [(12, 12)]
    assert parse_refs("#40-#12, 15") == [(12, 40), (15, 15)]
    assert parse_refs("3..5") == [(3, 5)]
    assert parse_refs("nothing") == []


def test_references_are_file_positions_and_skip_other_branches() -> None:
    user, assistant, result = _entries()
    sibling = MessageEntry(parent_id=user.id, message=AssistantMessage(content="other branch"))
    file_entries = [user, assistant, sibling, result]

    branch = numbered_branch(file_entries, [user, assistant, result])

    assert [item.number for item in branch] == [1, 2, 4]


def test_refs_return_entries_whole_with_their_reference(tmp_path: Path) -> None:
    branch = [NumberedEntry(i + 1, entry) for i, entry in enumerate(_entries())]

    text = history_text(branch, outputs_dir=None, refs="#2-#3")

    assert text.startswith("#2 assistant\nRunning tests.\n  -> bash(uv run pytest -q) [call c1]")
    assert "#3 tool result bash (error) [call c1]\nFAILED test_resume" in text
    assert "#1 " not in text


def test_query_and_tool_filters_combine() -> None:
    branch = [NumberedEntry(i + 1, entry) for i, entry in enumerate(_entries())]

    assert history_text(branch, outputs_dir=None, query="PYTEST").startswith("#2 assistant")
    by_tool = history_text(branch, outputs_dir=None, tool="bash", query="failed")
    assert by_tool.startswith("#3 tool result bash")
    assert history_text(branch, outputs_dir=None, query="absent") == (
        "No entries on the current branch match. Only this session's branch is "
        "searched, not earlier sessions or other branches."
    )


def test_a_search_skips_earlier_history_calls_and_answers() -> None:
    user, assistant, result = _entries()
    call = ToolCall(id="h1", name="history", arguments={"query": "resume"})
    asked = MessageEntry(message=AssistantMessage(content=assistant_content("", [call])))
    answered = MessageEntry(
        message=ToolResultMessage(
            tool_call_id="h1", tool_name="history", content="#1 user\nfix the resume path"
        )
    )
    branch = [
        NumberedEntry(i + 1, entry)
        for i, entry in enumerate([user, assistant, result, asked, answered])
    ]

    text = history_text(branch, outputs_dir=None, query="resume")

    assert [line for line in text.splitlines() if line.startswith("#")] == [
        "#1 user",
        "#3 tool result bash (error) [call c1]",
    ]
    by_history = history_text(branch, outputs_dir=None, tool="history", query="resume")
    assert by_history.startswith("#4 assistant")


def _long_turns(count: int, size: int = 50) -> list[NumberedEntry]:
    entries = [
        MessageEntry(message=UserMessage(content=f"turn {n} " + "x" * size)) for n in range(count)
    ]
    return [NumberedEntry(i + 1, entry) for i, entry in enumerate(entries)]


def test_refs_stop_at_the_budget_and_name_the_rest() -> None:
    limit = 150 + NOTE_CHARS

    text = history_text(_long_turns(5), outputs_dir=None, refs="#1-#5", max_chars=limit)

    assert "#2 user" in text and "#3 user" not in text
    assert text.endswith(
        "[3 more entries not shown, from #3: the answer stops at 350 characters. "
        "Ask for them by refs.]"
    )
    assert len(text) <= limit


def test_a_first_entry_over_the_budget_is_cut_from_its_start() -> None:
    limit = 1_000

    text = history_text(_long_turns(3, 5_000), outputs_dir=None, refs="#1-#3", max_chars=limit)

    # It starts where the entry does, so it reads in order.
    assert text.startswith("#1 user\nturn 0 xxx")
    assert "#1 is cut at 800 of 5,015 characters." in text
    assert text.endswith("2 more entries, from #2, are not shown.]")
    assert len(text) <= limit


def test_a_search_listing_stops_at_the_budget() -> None:
    limit = 150 + NOTE_CHARS

    text = history_text(_long_turns(5), outputs_dir=None, query="turn", max_chars=limit)

    assert "#2 user" in text and "#3 user" not in text
    assert text.endswith(
        "[3 more matches not shown; narrow the search or ask for a range by refs.]"
    )
    assert len(text) <= limit


def test_a_missing_reference_says_where_references_run() -> None:
    branch = [NumberedEntry(i + 1, entry) for i, entry in enumerate(_entries())]

    assert "references run to #3" in history_text(branch, outputs_dir=None, refs="#9")


def test_a_tool_result_names_its_output_file(tmp_path: Path) -> None:
    (tmp_path / "bash-c1.txt").write_text("FAILED test_resume")
    branch = [NumberedEntry(i + 1, entry) for i, entry in enumerate(_entries())]

    text = history_text(branch, outputs_dir=tmp_path, refs="#3")

    assert f"output: {tmp_path / 'bash-c1.txt'}" in text


def test_compaction_summaries_are_searchable() -> None:
    compaction = CompactionEntry(summary="Decided to keep 1h TTL", first_kept_entry_id="x")
    text = history_text([NumberedEntry(7, compaction)], outputs_dir=None, query="1h ttl")

    assert text.startswith("#7 compaction summary\nDecided to keep 1h TTL")


def _echo_tool() -> AgentTool:
    async def run(*_: object) -> AgentToolResult:
        return AgentToolResult(content=[TextContent(text="the needle output")])

    return AgentTool(
        name="echo", label="Echo", description="", parameters={"type": "object"}, execute_fn=run
    )


@pytest.mark.anyio
async def test_the_session_offers_history_and_it_reads_the_session_file(tmp_path: Path) -> None:
    echo = ToolCall(id="e1", name="echo", arguments={})
    recall = ToolCall(id="h1", name="history", arguments={"query": "needle"})
    provider = FakeProvider(
        [
            [
                assistant_start(model="fake"),
                assistant_done(
                    message=AssistantMessage(
                        content=assistant_content("", [echo]), stop_reason="toolUse"
                    )
                ),
            ],
            [
                assistant_start(model="fake"),
                assistant_done(
                    message=AssistantMessage(
                        content=assistant_content("", [recall]), stop_reason="toolUse"
                    )
                ),
            ],
            [assistant_start(model="fake"), assistant_done(message=AssistantMessage(content="ok"))],
        ]
    )
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="s",
            storage=JsonlSessionStorage(tmp_path / "s.jsonl"),
            cwd=tmp_path,
            tools=None,
            extensions_enabled=False,
        )
    )
    session._harness.config.tools.append(_echo_tool())

    async for _ in session.prompt("find it"):
        pass

    assert "history" in [tool.name for tool in session.tools]
    [answer] = [
        m for m in provider.calls[2][2] if m.role == "toolResult" and m.tool_name == "history"
    ]
    assert "tool result echo [call e1]" in answer.text
    assert "the needle output" in answer.text
    assert f"output: {tmp_path / 's.outputs' / 'echo-e1.txt'}" in answer.text


@pytest.mark.anyio
async def test_a_session_holds_history_answers_to_its_inline_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(INLINE_TOKENS_ENV, "500")
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    parent: str | None = None
    for n in range(12):
        entry = MessageEntry(
            parent_id=parent, message=UserMessage(content=f"turn {n} " + "x" * 400)
        )
        await storage.append(entry)
        parent = entry.id
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider([]), model="fake", system="", storage=storage, cwd=tmp_path
        )
    )
    history = next(tool for tool in session.tools if tool.name == "history")

    answer = await history.execute("c", {"refs": "#1-#12"})

    # The answer fits inline, so the window, which keeps the end, leaves it whole.
    assert session._buffer_result(answer, tool_name="history") is answer
    text = answer.content[0]
    assert isinstance(text, TextContent)
    assert text.text.startswith("#1 user\nturn 0 ")
    assert "more entries not shown, from #" in text.text
