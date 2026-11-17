"""A very long final report goes back to its author to condense, not to Tau to cut."""

import asyncio
from pathlib import Path

import pytest
from tau_agent.messages import AssistantMessage
from tau_ai import AssistantErrorEvent, AssistantStartEvent
from test_extension import (
    CapturingProvider,
    RecordingSession,
    _agent_tool,
    _extension_module,
    _load_runtime,
    _patch_provider_factory,
    _steer_tool,
    _text_stream,
    _tool_call_stream,
    _wait_for,
)

pytestmark = pytest.mark.anyio

LONG = "\n".join(f"- Finding {n}: the check passed." for n in range(1_000))
SHORT = "\n".join(f"- Finding {n}: passed." for n in range(100))


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _error_stream() -> list[object]:
    return [
        AssistantStartEvent(partial=AssistantMessage(model="fake")),
        AssistantErrorEvent(
            reason="error",
            error=AssistantMessage(
                model="fake", stop_reason="error", error_message="boom"
            ),
        ),
    ]


def _user_texts(call: dict) -> list[str]:
    return [m.text for m in call["messages"] if getattr(m, "role", None) == "user"]


def _result_tool(runtime):  # noqa: ANN001, ANN202
    return next(
        tool for tool in runtime.extension_tools if tool.name == "get_subagent_result"
    )


def _setup(tmp_path: Path, streams: list[list[object]]):  # noqa: ANN202
    runtime = _load_runtime(tmp_path)
    session = RecordingSession(tmp_path)
    runtime.bind(session)
    module = _extension_module()
    provider = CapturingProvider(streams)
    _patch_provider_factory(module, provider)
    return runtime, session, module, provider


async def test_a_long_report_is_condensed_by_the_agent_that_wrote_it(
    tmp_path: Path,
) -> None:
    runtime, _session, module, provider = _setup(
        tmp_path, [_text_stream(LONG), _text_stream(SHORT)]
    )
    assert len(LONG) > module.CONDENSE_OVER_CHARS  # type: ignore[attr-defined]
    assert module.CONDENSE_REQUEST_MESSAGE == (  # type: ignore[attr-defined]
        "Condense your report to under 8,000 characters; keep every finding,"
        " drop narration. Reply with the condensed report itself; do not call"
        " tools."
    )

    result = await _agent_tool(runtime).execute(
        "call-1", {"prompt": "p", "description": "d"}
    )

    assert len(provider.calls) == 2
    assert _user_texts(provider.calls[1])[-1] == module.CONDENSE_REQUEST_MESSAGE  # type: ignore[attr-defined]
    assert "[completed]" in result.text
    # The extra turn is not one of the run's own.
    assert "turns=1" in result.text
    assert result.text.endswith(SHORT) and LONG not in result.text
    note = result.text.split("\n\n")[-2]
    assert note.startswith(
        f"[The agent condensed this report from {len(LONG):,} to {len(SHORT):,}"
        " characters. The full report is in "
    )
    # The file the note names holds the report as it was first written, as
    # text a `read` can page, beside the transcript.
    full_report = Path(note.rsplit(" is in ", 1)[1].removesuffix(".]"))
    assert full_report.read_text() == LONG
    assert full_report.name == "agent-1.report.md"
    assert (full_report.parent / "agent-1.jsonl").exists()

    # A later fetch of the result gets the condensed report too.
    fetched = await _result_tool(runtime).execute("call-2", {"agent_id": "agent-1"})
    assert fetched.text.endswith(SHORT) and "condensed this report" in fetched.text


async def test_a_report_under_the_threshold_is_returned_as_written(
    tmp_path: Path,
) -> None:
    report = "x" * 12_000
    runtime, _session, module, provider = _setup(tmp_path, [_text_stream(report)])
    assert len(report) == module.CONDENSE_OVER_CHARS  # type: ignore[attr-defined]

    result = await _agent_tool(runtime).execute(
        "call-1", {"prompt": "p", "description": "d"}
    )

    assert len(provider.calls) == 1
    assert result.text.endswith(report) and "condensed" not in result.text


@pytest.mark.parametrize(
    "answer",
    [
        pytest.param(_error_stream(), id="provider-error"),
        pytest.param(_text_stream(""), id="empty"),
        pytest.param(_text_stream("y" * 12_000), id="still-too-long"),
        pytest.param(_text_stream("Done. " * 49), id="a-remark"),
        pytest.param(None, id="raises"),
    ],
)
async def test_a_failed_condense_turn_leaves_the_original_report(
    tmp_path: Path, answer: list[object] | None
) -> None:
    # With no second stream the fake provider raises when asked again.
    streams = [_text_stream(LONG)] + ([answer] if answer is not None else [])
    runtime, _session, _module, provider = _setup(tmp_path, streams)

    result = await _agent_tool(runtime).execute(
        "call-1", {"prompt": "p", "description": "d"}
    )

    assert len(provider.calls) == 2
    assert "[completed]" in result.text
    assert result.text.endswith(LONG) and "condensed" not in result.text


async def test_a_background_run_is_condensed_before_it_reports(
    tmp_path: Path,
) -> None:
    runtime, _session, _module, provider = _setup(
        tmp_path, [_text_stream(LONG), _text_stream(SHORT)]
    )

    await _agent_tool(runtime).execute(
        "call-1", {"prompt": "p", "description": "d", "run_in_background": True}
    )
    fetched = await _result_tool(runtime).execute(
        "call-2", {"agent_id": "agent-1", "wait": True}
    )

    assert len(provider.calls) == 2
    assert "[completed]" in fetched.text
    assert fetched.text.endswith(SHORT) and "condensed this report" in fetched.text


async def test_a_resumed_run_is_condensed_too(tmp_path: Path) -> None:
    runtime, _session, _module, provider = _setup(
        tmp_path,
        [_text_stream("First report"), _text_stream(LONG), _text_stream(SHORT)],
    )
    agent = _agent_tool(runtime)

    first = await agent.execute("call-1", {"prompt": "p", "description": "d"})
    assert first.text.endswith("First report") and len(provider.calls) == 1
    resumed = await agent.execute("call-2", {"resume": "agent-1", "prompt": "more"})

    assert len(provider.calls) == 3
    assert resumed.text.endswith(SHORT) and "condensed this report" in resumed.text


async def test_a_run_that_failed_is_not_asked_to_condense(tmp_path: Path) -> None:
    runtime, _session, _module, provider = _setup(tmp_path, [_error_stream()])

    result = await _agent_tool(runtime).execute(
        "call-1", {"prompt": "p", "description": "d"}
    )

    assert len(provider.calls) == 1
    assert "[error]" in result.text and "boom" in result.text


async def test_a_run_out_of_turns_is_not_asked_to_condense(tmp_path: Path) -> None:
    # The turn limit's own wrap-up request is the second call; no third follows.
    runtime, _session, _module, provider = _setup(
        tmp_path, [_text_stream("Working on it"), _text_stream(LONG)]
    )

    result = await _agent_tool(runtime).execute(
        "call-1", {"prompt": "p", "description": "d", "max_turns": 1}
    )

    assert len(provider.calls) == 2
    assert "[turn_limit]" in result.text
    assert result.text.endswith(LONG)


async def test_a_tool_call_in_the_condense_turn_ends_it(tmp_path: Path) -> None:
    # "Saved to report.md" is not the report, and neither is whatever the
    # child says after the tool call: the turn is cancelled at the call. Tau
    # 0.3 still makes the next request after a cancel; later versions do not.
    runtime, _session, _module, provider = _setup(
        tmp_path,
        [
            _text_stream(LONG),
            _tool_call_stream("Saved to report.md. " * 20, "c1"),
            _text_stream(SHORT),
        ],
    )

    result = await _agent_tool(runtime).execute(
        "call-1", {"prompt": "p", "description": "d"}
    )

    assert len(provider.calls) in (2, 3)
    assert "[completed]" in result.text and "turns=1" in result.text
    assert result.text.endswith(LONG) and "condensed" not in result.text


class _HeldSecondAnswer(CapturingProvider):
    """Plays its streams, holding the second until released."""

    def __init__(self, streams: list[list[object]]) -> None:
        super().__init__(streams)
        self.release = asyncio.Event()

    def stream_response(self, **kwargs):  # noqa: ANN003, ANN202
        events = super().stream_response(**kwargs)
        held = len(self.calls) == 2

        async def iterator():  # noqa: ANN202
            if held:
                await self.release.wait()
            async for event in events:
                yield event

        return iterator()


async def test_a_run_condensing_its_report_takes_no_steering(tmp_path: Path) -> None:
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))
    provider = _HeldSecondAnswer([_text_stream(LONG), _text_stream(SHORT)])
    _patch_provider_factory(_extension_module(), provider)

    await _agent_tool(runtime).execute(
        "call-1", {"prompt": "p", "description": "d", "run_in_background": True}
    )
    await _wait_for(lambda: len(provider.calls) == 2)
    steered = await _steer_tool(runtime).execute(
        "call-2", {"agent_id": "agent-1", "message": "Also check the docs."}
    )
    provider.release.set()
    fetched = await _result_tool(runtime).execute(
        "call-3", {"agent_id": "agent-1", "wait": True}
    )

    assert "has finished and is condensing its report" in steered.text
    # The steer was not queued: no third request, and its text went nowhere.
    assert len(provider.calls) == 2
    assert "Also check the docs." not in str(provider.calls)
    assert fetched.text.endswith(SHORT) and "condensed this report" in fetched.text
