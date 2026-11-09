"""`tau -p --max-turns N` reaches the session and ends the run with a report."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from pi_event_helpers import assistant_done, assistant_start, text_delta, tool_call_end
from tau_agent import AssistantMessage, ToolCall
from tau_ai import FakeProvider
from tau_coding import cli
from tau_coding.cli import app, run_print_mode


@pytest.mark.anyio
async def test_print_mode_max_turns_ends_with_a_report_turn(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    call = ToolCall(id="call-1", name="read", arguments={"path": "notes.txt"})
    report = AssistantMessage(content="read notes.txt; nothing left", model="fake")
    streams = [
        [
            assistant_start(),
            tool_call_end(call),
            assistant_done(AssistantMessage(content=[call], model="fake"), "toolUse"),
        ],
        [assistant_start(), text_delta(report.text), assistant_done(report)],
    ]
    provider = FakeProvider(streams + [streams[1]] * 4)

    ok = await run_print_mode(
        prompt="Read the notes",
        model="fake",
        cwd=tmp_path,
        provider=provider,
        max_turns=1,
    )

    assert ok is True
    assert "nothing left" in capsys.readouterr().out
    assert "Turn limit reached (1 used)" in provider.calls[1][2][-1].text


def test_max_turns_flag_reaches_print_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    async def fake_print_mode(*_args: object, **kwargs: object) -> bool:
        seen.update(kwargs)
        return True

    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "run_openai_print_mode", fake_print_mode)

    result = CliRunner().invoke(app, ["-p", "--max-turns", "3", "Say hello"])

    assert result.exit_code == 0, result.output
    assert seen["max_turns"] == 3


def test_max_turns_without_print_mode_is_an_error() -> None:
    result = CliRunner().invoke(app, ["--max-turns", "3", "Say hello"])

    assert result.exit_code != 0
    assert "only supported in print mode" in result.output
