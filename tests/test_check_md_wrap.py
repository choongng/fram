"""The Markdown wrap check rewraps only the prose a change touched."""

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_md_wrap.py"
_spec = importlib.util.spec_from_file_location("check_md_wrap", SCRIPT)
assert _spec is not None and _spec.loader is not None
wrap = importlib.util.module_from_spec(_spec)
sys.modules["check_md_wrap"] = wrap
_spec.loader.exec_module(wrap)

LONG = "word " * 30


def _check(
    tmp_path: Path, text: str, changed: set[int], *, fix: bool = True
) -> tuple[list[str], str]:
    path = tmp_path / "doc.md"
    path.write_text(text, encoding="utf-8")
    problems = wrap.check_file(str(path), changed, fix=fix)
    return problems, path.read_text(encoding="utf-8")


def test_a_long_changed_line_is_filled_to_80_columns(tmp_path: Path) -> None:
    problems, text = _check(tmp_path, LONG.strip() + "\n", {0})

    assert len(problems) == 1
    assert all(len(line) <= 80 for line in text.splitlines())
    assert text.split() == LONG.split()


def test_a_line_that_ends_early_is_joined_with_the_next(tmp_path: Path) -> None:
    problems, text = _check(tmp_path, "One short line.\nAnd the rest of it.\n", {1})

    assert "lines end early" in problems[0]
    assert text == "One short line. And the rest of it.\n"


def test_a_filled_paragraph_passes(tmp_path: Path) -> None:
    problems, _ = _check(tmp_path, "Already filled.\n", {0}, fix=False)

    assert problems == []


def test_unchanged_paragraphs_are_left_alone(tmp_path: Path) -> None:
    text = "Short.\nLines.\n\nShort.\nLines.\n"

    problems, after = _check(tmp_path, text, {3})

    assert after == "Short.\nLines.\n\nShort. Lines.\n"
    assert len(problems) == 1


def test_a_paragraph_upstream_left_unwrapped_keeps_its_style(tmp_path: Path) -> None:
    text = f"- {LONG.strip()}\n- a new short item\n  that ends early\n"

    problems, after = _check(tmp_path, text, {1, 2})

    assert problems == []
    assert after == text


def test_list_items_keep_their_markers_and_indent(tmp_path: Path) -> None:
    text = f"1. {LONG.strip()}\n2. second\n   item\n"

    _, after = _check(tmp_path, text, {0, 1, 2})

    lines = after.splitlines()
    assert lines[0].startswith("1. word")
    assert all(line.startswith("   word") for line in lines[1:-1])
    assert lines[-1] == "2. second item"


def test_code_tables_headings_and_hard_breaks_are_skipped(tmp_path: Path) -> None:
    text = f"```\n{LONG}\n```\n\n| a | {LONG} |\n\n# {LONG}\n\nA hard  \nbreak.\n"

    problems, after = _check(tmp_path, text, set(range(12)))

    assert problems == []
    assert after == text


def test_inline_code_is_never_split_across_lines(tmp_path: Path) -> None:
    code = "`uv run pytest tests/test_one.py tests/test_two.py`"
    text = "x " * 30 + code + "\n"

    _, after = _check(tmp_path, text, {0})

    assert any(code in line for line in after.splitlines())


def test_a_marker_word_never_starts_a_line(tmp_path: Path) -> None:
    text = "a" * 79 + " - b\n"

    _, after = _check(tmp_path, text, {0})

    assert not any(line.startswith("- ") for line in after.splitlines())


def test_an_unmatched_backtick_is_kept(tmp_path: Path) -> None:
    text = "an `unclosed span " + "word " * 20 + "\n"

    _, after = _check(tmp_path, text, {0})

    assert after.split() == text.split()


def test_toml_front_matter_is_skipped(tmp_path: Path) -> None:
    text = f'+++\nsummary = "{LONG}"\n+++\n\nShort.\n'

    problems, after = _check(tmp_path, text, {0, 1, 2, 4})

    assert problems == []
    assert after == text
