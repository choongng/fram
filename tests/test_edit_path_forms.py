"""An edit call with `path` in the wrong place is repaired or refused with the form."""

from pathlib import Path

import pytest

from tau_agent.loop import _checked_arguments
from tau_agent.messages import ToolCall
from tau_agent.tools import AgentTool, AgentToolResult
from tau_agent.types import JSONValue
from tau_coding import create_edit_tool
from tau_coding.tools import ToolInputError

FILE = "tests/test_core.py"
ORIGINAL = "def test_zero():\n    pass\n"
EDIT: dict[str, JSONValue] = {
    "oldText": ORIGINAL,
    "newText": "def test_zero():\n    assert compute(0)\n",
}
RIGHT_FORM = '{"path": "' + FILE + '", "edits": [{"oldText": "...", "newText": "..."}]}'


async def _call(tool: AgentTool, arguments: dict[str, JSONValue]) -> AgentToolResult:
    """Run `arguments` as the loop does: prepared and schema-checked, then executed."""
    prepared, problems = _checked_arguments(
        tool, ToolCall(id="call-1", name="edit", arguments=arguments)
    )
    if problems:
        raise ToolInputError("; ".join(problems))
    return await tool.execute("call-1", prepared)


def _target(tmp_path: Path) -> Path:
    target = tmp_path / FILE
    target.parent.mkdir()
    target.write_text(ORIGINAL, encoding="utf-8")
    return target


@pytest.mark.anyio
@pytest.mark.parametrize("edits", [[EDIT, -1, 10, FILE, FILE], [EDIT, FILE, FILE]])
async def test_bare_values_in_edits_are_refused_with_the_form(
    tmp_path: Path, edits: list[JSONValue]
) -> None:
    target = _target(tmp_path)
    tool = create_edit_tool(cwd=tmp_path)

    with pytest.raises(ToolInputError) as caught:
        await _call(tool, {"edits": edits})

    message = str(caught.value)
    assert not message.startswith("`path` is missing")
    assert f'`"{FILE}"`' in message
    assert "Put `path` at the top level, once, beside an `edits` list" in message
    assert message.endswith(RIGHT_FORM)
    assert target.read_text(encoding="utf-8") == ORIGINAL


@pytest.mark.anyio
async def test_a_path_shared_by_every_edit_object_is_taken(tmp_path: Path) -> None:
    target = _target(tmp_path)
    tool = create_edit_tool(cwd=tmp_path)

    result = await _call(tool, {"edits": [{**EDIT, "path": FILE}]})

    assert result.details is not None and result.details["path"] == str(target)
    assert "assert compute(0)" in target.read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_edits_naming_two_files_are_refused(tmp_path: Path) -> None:
    _target(tmp_path)
    tool = create_edit_tool(cwd=tmp_path)
    one = {"oldText": "a", "newText": "b", "path": FILE}
    two = {"oldText": "c", "newText": "d", "path": "core.py"}

    with pytest.raises(ToolInputError) as caught:
        await _call(tool, {"edits": [one, two]})

    message = str(caught.value)
    assert "one edit call changes one file" in message
    assert message.endswith(RIGHT_FORM)


@pytest.mark.anyio
async def test_a_top_level_path_wins_over_one_inside_an_edit(tmp_path: Path) -> None:
    target = _target(tmp_path)
    tool = create_edit_tool(cwd=tmp_path)

    await _call(tool, {"path": FILE, "edits": [{**EDIT, "path": "elsewhere.py"}]})

    assert "assert compute(0)" in target.read_text(encoding="utf-8")
    assert not (tmp_path / "elsewhere.py").exists()
