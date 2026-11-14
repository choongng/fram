"""Extensions that installed distributions declare through entry points."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from tau_coding.extensions import ExtensionRuntime, discover_extensions, load_extensions
from tau_coding.extensions import loader as loader_module
from tau_coding.extensions.loader import (
    DISABLE_PACKAGE_EXTENSIONS_ENV,
    PACKAGE_EXTENSION_GROUP,
)
from tau_coding.resources import TauResourcePaths

PACKAGE_SETUP = """
SETUP_CALLS = []


def setup(tau):
    SETUP_CALLS.append(tau)


NOT_CALLABLE = 42
"""


def _paths(tmp_path: Path) -> TauResourcePaths:
    return TauResourcePaths(
        root=tmp_path / "home-tau",
        cwd=tmp_path / "project",
        agents_root=tmp_path / "home-agents",
    )


@pytest.fixture
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[EntryPoint]]:
    """Stand in for the installed distributions' entry points.

    A test appends `EntryPoint`s to the yielded list; `pkgext` is an importable
    module whose `setup` they can point at.
    """
    site = tmp_path / "site"
    site.mkdir()
    (site / "pkgext.py").write_text(PACKAGE_SETUP, encoding="utf-8")
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.delenv(DISABLE_PACKAGE_EXTENSIONS_ENV, raising=False)
    declared: list[EntryPoint] = []

    def fake_entry_points(*, group: str) -> list[EntryPoint]:
        return [entry for entry in declared if entry.group == group]

    monkeypatch.setattr(loader_module, "entry_points", fake_entry_points)
    yield declared
    sys.modules.pop("pkgext", None)


def _entry(name: str, value: str = "pkgext:setup") -> EntryPoint:
    return EntryPoint(name=name, value=value, group=PACKAGE_EXTENSION_GROUP)


def _write_user_extension(paths: TauResourcePaths, name: str) -> None:
    directory = paths.root / "extensions"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.py").write_text("def setup(tau):\n    pass\n", encoding="utf-8")


def test_package_extensions_load_after_directories_and_explicit_paths(
    tmp_path: Path, installed: list[EntryPoint]
) -> None:
    paths = _paths(tmp_path)
    installed.append(_entry("from_package"))
    _write_user_extension(paths, "from_user")
    explicit = tmp_path / "from_explicit.py"
    explicit.write_text("def setup(tau):\n    pass\n", encoding="utf-8")

    result = load_extensions(paths, extra_paths=(explicit,))

    assert [extension.name for extension in result.extensions] == [
        "from_user",
        "from_explicit",
        "from_package",
    ]
    package = result.extensions[-1]
    assert package.source == "package"
    assert package.path is None
    assert package.source_id == "package:pkgext:setup"
    assert result.diagnostics == ()


def test_a_same_named_directory_extension_shadows_the_package(
    tmp_path: Path, installed: list[EntryPoint]
) -> None:
    paths = _paths(tmp_path)
    installed.append(_entry("shared"))
    _write_user_extension(paths, "shared")

    discovered, diagnostics = discover_extensions(paths)

    assert [(entry.name, entry.source) for entry in discovered] == [("shared", "user")]
    assert [diagnostic.message for diagnostic in diagnostics] == [
        "duplicate extension name ignored (first-loaded wins)"
    ]


@pytest.mark.parametrize(
    "flags",
    [{"include_resource_dirs": False}, {"include_user_dir": False}, {"include_packages": False}],
)
def test_package_extensions_follow_the_user_resource_switches(
    tmp_path: Path, installed: list[EntryPoint], flags: dict[str, bool]
) -> None:
    installed.append(_entry("from_package"))

    discovered, _ = discover_extensions(_paths(tmp_path), **flags)

    assert discovered == ()


def test_the_environment_disables_package_extensions_by_name(
    tmp_path: Path, installed: list[EntryPoint], monkeypatch: pytest.MonkeyPatch
) -> None:
    installed.extend([_entry("kept"), _entry("dropped"), _entry("also_dropped")])
    monkeypatch.setenv(DISABLE_PACKAGE_EXTENSIONS_ENV, "dropped, also_dropped")

    discovered, _ = discover_extensions(_paths(tmp_path))

    assert [entry.name for entry in discovered] == ["kept"]


def test_a_broken_package_extension_is_reported_and_the_rest_load(
    tmp_path: Path, installed: list[EntryPoint]
) -> None:
    installed.extend(
        [
            _entry("missing", "no_such_module:setup"),
            _entry("not_callable", "pkgext:NOT_CALLABLE"),
            _entry("works"),
        ]
    )

    result = load_extensions(_paths(tmp_path))

    assert [extension.name for extension in result.extensions] == ["works"]
    messages = {diagnostic.name: diagnostic.message for diagnostic in result.diagnostics}
    assert messages["missing"].startswith("failed to import no_such_module:setup")
    assert messages["not_callable"] == (
        "package extension entry point pkgext:NOT_CALLABLE is not callable"
    )
    assert {diagnostic.severity for diagnostic in result.diagnostics} == {"error"}


def test_the_runtime_runs_and_lists_a_package_extension(
    tmp_path: Path, installed: list[EntryPoint]
) -> None:
    installed.append(_entry("from_package"))
    runtime = ExtensionRuntime(built_in_extensions=())

    runtime.load(_paths(tmp_path))

    import pkgext  # type: ignore[import-not-found]

    assert len(pkgext.SETUP_CALLS) == 1
    assert runtime.extension_names == ("from_package",)
    (metadata,) = runtime.extension_metadata
    assert (metadata.source, metadata.path, metadata.hidden) == ("package", None, False)
