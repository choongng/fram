"""The tools that return a child's report ask Tau not to cut it."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from tau_agent.tools import AgentTool
from test_extension import _extension_module, _load_runtime

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_report_tools_ask_tau_for_the_whole_result_where_it_can_be_asked(
    tmp_path: Path,
) -> None:
    runtime = _load_runtime(tmp_path)
    module = _extension_module()
    options = module.report_tool_options  # type: ignore[attr-defined]

    @dataclass
    class OlderTool:
        name: str

    @dataclass
    class NewerTool:
        name: str
        output_window: str = "ends"

    # A Tau without the field is given no argument it would reject.
    assert options(OlderTool) == {}
    assert OlderTool(name="agent", **options(OlderTool)).name == "agent"
    assert NewerTool(name="agent", **options(NewerTool)).output_window == "whole"

    # On the Tau these tests run against, whichever it is.
    expected = "whole" if "output_window" in AgentTool.__dataclass_fields__ else None
    declared = {
        tool.name: getattr(tool, "output_window", None)
        for tool in runtime.extension_tools
    }
    assert declared["agent"] == declared["get_subagent_result"] == expected
    del declared["agent"], declared["get_subagent_result"]
    assert set(declared.values()) <= {None, "ends"}
