"""Compaction asks the live conversation for its summary, with a code-built index.

See dev-notes/session-recall-and-continuation-compaction.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pi_event_helpers import assistant_done, assistant_start
from tau_agent import AssistantMessage, ToolCall
from tau_agent.cache_hints import conversation_long_lived
from tau_agent.messages import UserMessage, assistant_content
from tau_agent.session import JsonlSessionStorage, MessageEntry
from tau_ai import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig
from tau_coding.context_window import (
    SUMMARIZATION_SYSTEM_PROMPT,
    UPDATE_SUMMARIZATION_PROMPT,
    build_continuation_summary_request,
)
from tau_coding.session_recall import COMPACTION_INDEX_HEADING, split_compaction_index


class _Recording(FakeProvider):
    def __init__(self, streams: list[list[object]]) -> None:
        super().__init__(streams)  # type: ignore[arg-type]
        self.long_lived: list[bool] = []

    def stream_response(self, **kwargs: object):  # type: ignore[no-untyped-def,override]
        self.long_lived.append(conversation_long_lived())
        return super().stream_response(**kwargs)  # type: ignore[arg-type]


def _reply(message: AssistantMessage) -> list[object]:
    return [assistant_start(model="fake"), assistant_done(message=message)]


def _cut_off(text: str) -> list[object]:
    """A reply that stopped at the output limit."""
    return [assistant_start(model="fake"), assistant_done(AssistantMessage(content=text), "length")]


async def _session(
    tmp_path: Path, provider: FakeProvider, *, turns: int = 3, long_lived: bool = False
) -> CodingSession:
    storage = JsonlSessionStorage(tmp_path / "s.jsonl")
    parent = None
    for n in range(turns):
        for message in (
            UserMessage(content=f"task {n} " + "context " * 2_000),
            AssistantMessage(content=f"answer {n}"),
        ):
            entry = MessageEntry(parent_id=parent, message=message)
            await storage.append(entry)
            parent = entry.id
    return await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            system="You are Tau.",
            storage=storage,
            cwd=tmp_path,
            tools=[],
            long_lived=long_lived,
            # A quarter of it is kept verbatim (the last turn); the context
            # stays under it, so only an explicit compact() runs.
            auto_compact_token_threshold=40_000,
        )
    )


def test_the_update_prompt_rewrite_still_matches() -> None:
    request = build_continuation_summary_request(last_ref=4, index="", previous_summary=True)

    assert "<previous-summary>" not in request
    assert UPDATE_SUMMARIZATION_PROMPT.split("\n\n", 1)[1] in request


@pytest.mark.anyio
async def test_the_summary_is_a_continuation_of_the_cached_conversation(tmp_path: Path) -> None:
    reply = "```captions\n#1-#2: Did task 0.\n```\n## Goal\nShip it"
    provider = _Recording([_reply(AssistantMessage(content=reply))])
    session = await _session(tmp_path, provider, long_lived=True)
    sent_before = list(session.messages)

    await session.compact()

    [(_model, system, messages, _tools)] = provider.calls
    assert system == "You are Tau."
    assert messages[: len(sent_before)] == sent_before
    request = messages[-1]
    assert isinstance(request, UserMessage)
    assert "Do not call a tool or continue the work" in request.text
    assert '#1-#2 user: "task 0' in request.text
    assert "```captions\n#1-#2: <caption>\n```" in request.text
    assert provider.long_lived == [True]
    summary = session.messages[0].text.removeprefix("Previous conversation summary:\n")
    index, rest = split_compaction_index(summary)
    assert rest == "## Goal\nShip it"
    assert '#1-#2 user: "task 0' in index
    [compaction] = [e for e in await session._config.storage.read_all() if e.type == "compaction"]
    assert compaction.index is not None
    assert compaction.index[0]["caption"] == "Did task 0."


@pytest.mark.anyio
async def test_a_tool_call_instead_of_a_summary_falls_back(tmp_path: Path) -> None:
    call = ToolCall(id="x", name="bash", arguments={"command": "ls"})
    provider = FakeProvider(
        [
            _reply(AssistantMessage(content=assistant_content("", [call]), stop_reason="toolUse")),
            _reply(AssistantMessage(content="Fallback summary")),
        ]
    )
    session = await _session(tmp_path, provider)

    await session.compact()

    assert len(provider.calls) == 2
    assert provider.calls[1][1] == SUMMARIZATION_SYSTEM_PROMPT
    summary = session.messages[0].text
    assert summary.startswith("Previous conversation summary:\n" + COMPACTION_INDEX_HEADING)
    assert summary.endswith("Fallback summary")


@pytest.mark.anyio
async def test_a_context_that_would_not_fit_uses_the_separate_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider([_reply(AssistantMessage(content="Separate summary"))])
    session = await _session(tmp_path, provider)
    monkeypatch.setattr(type(session), "context_window_tokens", property(lambda self: 1_000))

    await session.compact()

    [(_model, system, messages, tools)] = provider.calls
    assert system == SUMMARIZATION_SYSTEM_PROMPT
    assert tools == []
    # The messages carry the references the captions are asked by.
    assert "<message ref=#1 role=user>" in messages[0].text
    assert "carry their #N references" in messages[0].text


@pytest.mark.anyio
async def test_a_separate_request_after_a_compaction_keeps_the_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider(
        [
            _reply(AssistantMessage(content="first summary")),
            *(_reply(AssistantMessage(content=f"answer {n}")) for n in range(3, 6)),
            _reply(AssistantMessage(content="second summary")),
        ]
    )
    session = await _session(tmp_path, provider)
    await session.compact()
    for n in range(3, 6):
        async for _ in session.prompt(f"task {n} " + "context " * 2_000):
            pass
    monkeypatch.setattr(type(session), "context_window_tokens", property(lambda self: 1_000))

    await session.compact()

    prompt = provider.calls[-1][2][0].text
    assert "<previous-summary>" in prompt
    assert "<message ref=#3 role=user>" in prompt
    assert "index=1" not in prompt


@pytest.mark.anyio
async def test_a_context_too_big_with_its_tail_leaves_the_tail_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeProvider([_reply(AssistantMessage(content="## Goal\nShip it"))])
    session = await _session(tmp_path, provider)
    sent_before = list(session.messages)
    # Each turn is about 4k tokens and the last two are kept: the whole context
    # and the request do not fit, the summarized turn and the request do.
    monkeypatch.setattr(type(session), "context_window_tokens", property(lambda self: 19_000))

    await session.compact()

    [(_model, system, messages, _tools)] = provider.calls
    assert system == "You are Tau."
    assert messages[:-1] == sent_before[:2]
    assert "Do not call a tool or continue the work" in messages[-1].text
    assert session.messages[0].text.endswith("## Goal\nShip it")


@pytest.mark.anyio
async def test_a_summary_cut_off_at_the_output_limit_is_kept_and_marked(tmp_path: Path) -> None:
    reply = "```captions\n#1-#2: Did task 0.\n```\n## Goal\nShip"
    provider = FakeProvider([_cut_off(reply)])
    session = await _session(tmp_path, provider)

    await session.compact()

    assert len(provider.calls) == 1
    summary = session.messages[0].text
    assert summary.endswith("## Goal\nShip\n\n[Summary cut off at the output limit.]")


@pytest.mark.anyio
async def test_a_reply_cut_off_inside_its_captions_falls_back(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            _cut_off("```captions\n#1-#2: Did ta"),
            _reply(AssistantMessage(content="Fallback summary")),
        ]
    )
    session = await _session(tmp_path, provider)

    await session.compact()

    assert len(provider.calls) == 2
    assert provider.calls[1][1] == SUMMARIZATION_SYSTEM_PROMPT
    summary = session.messages[0].text
    assert summary.endswith("Fallback summary")
    assert "```captions" not in summary


@pytest.mark.anyio
async def test_a_second_compaction_carries_the_first_index_forward(tmp_path: Path) -> None:
    later = [f"task {n} " + "context " * 2_000 for n in range(3, 8)]
    provider = FakeProvider(
        [
            _reply(AssistantMessage(content="first summary")),
            *(_reply(AssistantMessage(content=f"answer {n}")) for n in range(3, 8)),
            _reply(AssistantMessage(content="second summary")),
        ]
    )
    session = await _session(tmp_path, provider, turns=3)
    await session.compact()
    for prompt in later:
        async for _ in session.prompt(prompt):
            pass

    await session.compact()

    request = provider.calls[-1][2][-1].text
    # The first chapter is shown from the stored tree, and the new turns are listed.
    assert '#1-#2 user: "task 0' in request
    assert 'user: "task 3' in request
    summary = session.messages[0].text.split("\n", 1)[1]
    index, rest = split_compaction_index(summary)
    assert rest == "second summary"
    assert index.splitlines()[3].startswith("#1")
    assert "first summary" not in summary
    [_, second] = [e for e in await session._config.storage.read_all() if e.type == "compaction"]
    assert second.index is not None and second.index[0]["first"] == 1
