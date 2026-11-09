"""`tau -p` writes each extension that failed to load to stderr.

This runs print mode as the command line does, through the typer app, with two
extensions that fail to load for different reasons, and reads stdout and stderr
apart.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from pi_event_helpers import assistant_done, assistant_start, text_delta
from tau_agent import AssistantMessage
from tau_ai import FakeProvider
from tau_coding import cli, session
from tau_coding.cli import app
from tau_coding.provider_config import OpenAICompatibleProviderConfig, ProviderSettings
from tau_coding.resources import ResourceDiagnostic


def test_print_mode_writes_each_failed_extension_to_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    extensions = tmp_path / "extensions"
    extensions.mkdir()
    broken = extensions / "broken.py"
    broken.write_text("import no_such_module_for_this_test\n", encoding="utf-8")
    no_setup = extensions / "nosetup.py"
    no_setup.write_text("VALUE = 1\n", encoding="utf-8")

    settings = ProviderSettings(
        default_provider="local",
        providers=(
            OpenAICompatibleProviderConfig(
                name="local",
                base_url="http://localhost:11434/v1",
                api_key_env="LOCAL_API_KEY",
                models=("fake",),
                default_model="fake",
            ),
        ),
    )

    class ClosableFakeProvider(FakeProvider):
        async def aclose(self) -> None:
            return None

    created: list[ClosableFakeProvider] = []

    def fake_provider(*args: object, **kwargs: object) -> ClosableFakeProvider:
        reply = AssistantMessage(content="Hello")
        stream = [
            assistant_start(model="fake"),
            text_delta(delta="Hello"),
            assistant_done(message=reply),
        ]
        # Every call gets the reply: the run's session-naming request is one of them.
        created.append(ClosableFakeProvider([stream] * 8))
        return created[-1]

    monkeypatch.setattr(cli, "_startup_update_notice", lambda: None)
    monkeypatch.setattr(cli, "load_provider_settings", lambda: settings)
    for module in (cli, session):
        monkeypatch.setattr(module, "create_model_provider", fake_provider)
    monkeypatch.setenv("LOCAL_API_KEY", "test-key")
    monkeypatch.chdir(project)

    result = CliRunner().invoke(
        app,
        [
            "-p",
            "--provider",
            "local",
            "--model",
            "fake",
            "-e",
            str(broken),
            "-e",
            str(no_setup),
            "Say hello",
        ],
    )

    assert result.exit_code == 0, result.output
    # The prompt reached the model: the run went on past the failures.
    assert any(provider.calls for provider in created)
    assert result.stdout == "Hello\n"
    errors = [line for line in result.stderr.splitlines() if line.startswith("error extension")]
    assert len(errors) == 2, result.stderr
    [import_line] = [line for line in errors if " broken:" in line]
    assert "failed to import extension" in import_line
    assert "no_such_module_for_this_test" in import_line
    assert str(broken) in import_line
    [setup_line] = [line for line in errors if " nosetup:" in line]
    assert "does not define a callable" in setup_line
    assert str(no_setup) in setup_line


def test_print_mode_lists_failed_extensions_and_trust_notes_only() -> None:
    failed = ResourceDiagnostic(
        kind="extension", name="broken", message="failed to import", severity="error"
    )
    shadowed = ResourceDiagnostic(kind="extension", name="other", message="shadowed")
    trust = ResourceDiagnostic(kind="project-trust", message="untrusted", severity="error")
    skill = ResourceDiagnostic(kind="skill", message="bad frontmatter", severity="error")

    lines = cli.print_mode_diagnostic_lines([failed, shadowed, trust, skill])

    assert lines == [failed.format(), trust.format()]
