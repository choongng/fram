"""Calls whose arguments are only unparsed text say what arrived and what didn't."""

from pathlib import Path

import pytest

from tau_coding import create_bash_tool, create_write_tool
from tau_coding.tools import ToolInputError


async def _write_error(tmp_path: Path, text: str) -> str:
    tool = create_write_tool(cwd=tmp_path)
    with pytest.raises(ToolInputError) as caught:
        await tool.execute("call-1", {"_raw_arguments": text})
    return str(caught.value)


@pytest.mark.anyio
async def test_cut_off_write_names_the_cut_off_not_the_path(tmp_path: Path) -> None:
    text = '{"path":"x.txt","content":"abc'

    message = await _write_error(tmp_path, text)

    assert f"after {len(text)} characters" in message
    assert "cut off" in message
    assert "nothing was written" in message
    assert "path must be a string" not in message
    assert not (tmp_path / "x.txt").exists()


@pytest.mark.anyio
async def test_cut_off_bash_does_not_run(tmp_path: Path) -> None:
    tool = create_bash_tool(cwd=tmp_path)

    with pytest.raises(ToolInputError) as caught:
        await tool.execute("call-1", {"_raw_arguments": '{"command":"touch MARK'})

    assert "the bash tool did not run" in str(caught.value)
    assert not (tmp_path / "MARK").exists()


@pytest.mark.anyio
async def test_parsed_arguments_still_reach_the_tool(tmp_path: Path) -> None:
    tool = create_write_tool(cwd=tmp_path)

    await tool.execute("call-1", {"path": "x.txt", "content": "abc"})

    assert (tmp_path / "x.txt").read_text(encoding="utf-8") == "abc"


@pytest.mark.anyio
async def test_cut_off_write_names_what_arrived_and_what_never_came(tmp_path: Path) -> None:
    # `content` came first and ran on past the file's text; `path` never came.
    text = '{"content":"first line\\nsecond line\\nand the text that kept going\\n'

    message = await _write_error(tmp_path, text)

    assert "`content` opened and never closed: it ran" in message
    assert 'ending "…' in message and "text that kept going" in message
    assert "`path` never came." in message
    assert "Send `path` first" in message and "reasoning before the call" in message


@pytest.mark.anyio
async def test_cut_off_with_its_required_arguments_closed_names_only_the_open_one(
    tmp_path: Path,
) -> None:
    message = await _write_error(tmp_path, '{"path":"x.txt","content":"abc')

    assert "`content` opened and never closed" in message
    assert "never came" not in message and "Send `path` first" not in message


@pytest.mark.anyio
@pytest.mark.parametrize(
    "text",
    [
        '{"content": "PAT = re.compile(\'\\d+\')\\n", "path": "a.py"}',
        '{"content": "line1\nline2", "path": "a.py"}',
    ],
)
async def test_a_bad_escape_is_not_called_a_cut_off(tmp_path: Path, text: str) -> None:
    message = await _write_error(tmp_path, text)

    assert "Every argument arrived (`content` and `path`)" in message
    assert "never closed" not in message and "never came" not in message
    assert r"(so `\d` is `\\d`)" in message


@pytest.mark.anyio
@pytest.mark.parametrize("text", ["[1, 2]", "", '{"path":"a","content":"x"} trailing'])
async def test_text_that_is_no_single_object_is_not_called_a_cut_off(
    tmp_path: Path, text: str
) -> None:
    message = await _write_error(tmp_path, text)

    assert "single JSON object" in message
    assert "cut off" not in message


@pytest.mark.anyio
async def test_a_key_with_no_value_is_named_as_such(tmp_path: Path) -> None:
    message = await _write_error(tmp_path, '{"path":')

    assert "`path` has no value." in message
    assert "never closed" not in message
