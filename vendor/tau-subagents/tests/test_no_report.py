"""A child that ends with an empty message is asked once for its report."""

from pathlib import Path

import pytest
from test_extension import (
    CapturingProvider,
    RecordingSession,
    _agent_tool,
    _extension_module,
    _load_runtime,
    _patch_provider_factory,
    _text_stream,
    _tool_call_stream,
    _wait_for,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _last_user_text(call: dict) -> str:
    users = [m for m in call["messages"] if getattr(m, "role", None) == "user"]
    return users[-1].text


async def test_an_empty_ending_gets_one_reprompt_and_the_report(
    tmp_path: Path,
) -> None:
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))
    module = _extension_module()
    provider = CapturingProvider([_text_stream(""), _text_stream("The report")])
    _patch_provider_factory(module, provider)

    result = await _agent_tool(runtime).execute(
        "call-1", {"prompt": "p", "description": "d"}
    )

    assert "[completed]" in result.text
    assert result.text.endswith("The report")
    assert len(provider.calls) == 2
    assert _last_user_text(provider.calls[1]) == module.REPORT_REQUEST_MESSAGE  # type: ignore[attr-defined]


async def test_still_empty_after_the_reprompt_falls_back_and_says_so(
    tmp_path: Path,
) -> None:
    runtime = _load_runtime(tmp_path)
    session = RecordingSession(tmp_path)
    runtime.bind(session)
    module = _extension_module()
    # Three responses: a remark with a tool call, then empty twice. A second
    # re-prompt would ask for a fourth and fail the run.
    provider = CapturingProvider(
        [
            _tool_call_stream("Looking into it", "c1"),
            _text_stream(""),
            _text_stream(""),
        ]
    )
    _patch_provider_factory(module, provider)

    await _agent_tool(runtime).execute(
        "call-1", {"prompt": "p", "description": "d", "run_in_background": True}
    )
    await _wait_for(lambda: session.followed_up)
    note = session.followed_up[0]

    assert len(provider.calls) == 3
    assert "<status>completed_no_report</status>" in note
    assert "<result>Looking into it</result>" in note
    output_file = note.split("<output-file>", 1)[1].split("</output-file>", 1)[0]
    assert f"read its transcript at {output_file}" in note
    custom_type, details = session.followed_up_custom[0]
    assert details["status"] == "completed_no_report"

    from tau_subagents.notification_render import render_notification

    class View:
        pass

    view = View()
    view.details = details  # type: ignore[attr-defined]
    rendered = render_notification(view, View())
    assert "completed (no report)" in rendered


async def test_a_tool_call_ending_is_not_an_empty_report(tmp_path: Path) -> None:
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))
    module = _extension_module()
    provider = CapturingProvider(
        [_tool_call_stream("", "c1"), _text_stream("Done")]
    )
    _patch_provider_factory(module, provider)

    result = await _agent_tool(runtime).execute(
        "call-1", {"prompt": "p", "description": "d"}
    )
    assert "[completed]" in result.text
    assert len(provider.calls) == 2
