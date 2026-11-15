"""Output files: written while a tool runs, stamped with event time."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tau_agent.messages import AssistantMessage, UserMessage
from tau_ai import FakeProvider
from test_extension import (
    RecordingSession,
    _agent_tool,
    _bash_call_stream,
    _extension_module,
    _load_runtime,
    _patch_provider_factory,
    _stop_tool,
    _text_stream,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _entries(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


async def test_tool_call_is_on_disk_while_the_tool_runs(tmp_path: Path) -> None:
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))
    module = _extension_module()
    _patch_provider_factory(
        module,
        FakeProvider([_bash_call_stream("sleep 30", "c1"), _text_stream("never")]),
    )

    spawned = await _agent_tool(runtime).execute(
        "call-1", {"prompt": "probe", "description": "probe", "run_in_background": True}
    )
    output_path = Path(str(spawned.details["output_file"]))
    for _ in range(500):
        if any(entry["type"] == "assistant" for entry in _entries(output_path)):
            break
        await asyncio.sleep(0.01)
    # The turn has not ended: the call is written because its tool started.
    assistant = next(e for e in _entries(output_path) if e["type"] == "assistant")
    assert "sleep 30" in json.dumps(assistant["message"])

    await _stop_tool(runtime).execute("call-2", {"agent_id": "agent-1"})


def test_entries_carry_the_message_time_not_the_flush_time(tmp_path: Path) -> None:
    from tau_subagents.output_file import OutputFileWriter

    path = tmp_path / "root" / "p" / "s" / "tasks" / "agent-1.jsonl"
    writer = OutputFileWriter(path, "agent-1", tmp_path)
    created = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    stamp_ms = int(created.timestamp() * 1000)

    async def write() -> None:
        await writer.write_initial("go")
        await writer.flush(
            [
                UserMessage(content="go"),
                AssistantMessage(content="done", timestamp=stamp_ms),
            ]
        )

    asyncio.run(write())
    entries = _entries(path)
    assert [entry["type"] for entry in entries] == ["user", "assistant"]
    assert entries[1]["timestamp"] == created.isoformat()
    # The prompt entry has no message of its own, so it keeps the write time.
    assert datetime.fromisoformat(str(entries[0]["timestamp"])) > created


async def test_a_resumed_run_appends_to_its_output_file(tmp_path: Path) -> None:
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))
    module = _extension_module()
    _patch_provider_factory(
        module, FakeProvider([_text_stream("First"), _text_stream("Second")])
    )
    agent_tool = _agent_tool(runtime)
    first = await agent_tool.execute("call-1", {"prompt": "start", "description": "d"})
    output_path = Path(str(first.details["output_file"]))

    await agent_tool.execute("call-2", {"resume": "agent-1", "prompt": "keep going"})

    texts = [json.dumps(entry["message"]) for entry in _entries(output_path)]
    # Match the reply text: Tau 0.4+ also writes `timing.timeToFirstOutputMs`.
    assert sum('"text": "First"' in text for text in texts) == 1
    assert any("keep going" in text for text in texts)
    assert any("Second" in text for text in texts)
