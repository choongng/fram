"""Every child is told it is a sub-agent that does its own task."""

from pathlib import Path

import pytest
from test_extension import (
    CapturingProvider,
    RecordingSession,
    _agent_tool,
    _extension_module,
    _fork_parent,
    _load_runtime,
    _patch_fork_resolution,
    _patch_provider_factory,
    _prompts_module,
    _text_stream,
)
from test_nesting import Delegators, delegating  # noqa: F401 - a fixture

pytestmark = pytest.mark.anyio

SPAWN_RULE = "Do not hand your task, or a reworded copy of it, to another agent."
NO_AGENT_TOOL = "You have no agent tool; ignore any mention of one."
SNIPPET = "never to pass on your own task"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _role(*, can_spawn: bool = True, agent_listed: bool = False) -> str:
    return _prompts_module().sub_agent_role(  # type: ignore[attr-defined]
        can_spawn=can_spawn, agent_listed=agent_listed
    )


def _tau_rebuilds_prompts() -> bool:
    """Whether this Tau can rebuild a prompt after tools are taken away."""
    from tau_coding import CodingSession

    return hasattr(CodingSession, "refresh_system_prompt")


async def _spawn_system_prompt(
    tmp_path: Path, arguments: dict[str, object]
) -> str:
    """Spawn one child and return the system prompt its provider was sent."""
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))
    provider = CapturingProvider([_text_stream("done")])
    _patch_provider_factory(_extension_module(), provider)
    await _agent_tool(runtime).execute(
        "call-1", {"prompt": "go", "description": "d", **arguments}
    )
    return str(provider.calls[0]["system"])


def _write_agent(tmp_path: Path, name: str, text: str) -> None:
    (tmp_path / ".tau" / "agents").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".tau" / "agents" / f"{name}.md").write_text(text)


async def test_role_text(tmp_path: Path) -> None:
    _load_runtime(tmp_path)
    assert _role() == (
        "<sub_agent_role>\n"
        "You are a sub-agent. Another agent gave you this task to do, not to pass on.\n"
        "Do the work yourself with your own tools, then report what you found or did.\n"
        "Do not hand your task, or a reworded copy of it, to another agent. Spawn an\n"
        "agent only for a clearly separate piece of work that can run alongside yours,\n"
        "and never when you could do it with a few tool calls.\n"
        "</sub_agent_role>"
    )
    # A child with no `agent` tool is not told when to use one.
    assert SPAWN_RULE not in _role(can_spawn=False)
    assert "Do the work yourself" in _role(can_spawn=False)
    assert NO_AGENT_TOOL not in _role(can_spawn=False)
    assert _role(can_spawn=False, agent_listed=True).endswith(
        f"found or did.\n{NO_AGENT_TOOL}\n</sub_agent_role>"
    )
    # A child that has the tool is not told it has none.
    assert _role(agent_listed=True) == _role()


async def test_general_keeps_the_default_prompt_and_gains_the_role(
    tmp_path: Path,
) -> None:
    system = await _spawn_system_prompt(tmp_path, {})

    # `general` has no prompt of its own: Tau's default coding prompt stays.
    assert system.startswith("You are an expert coding assistant")
    assert _role() in system


async def test_a_custom_replace_prompt_gains_the_role(tmp_path: Path) -> None:
    _write_agent(tmp_path, "writer", "---\ndescription: d\n---\nWrite carefully.")

    system = await _spawn_system_prompt(tmp_path, {"subagent_type": "writer"})

    assert system.startswith("Write carefully.")
    assert "expert coding assistant" not in system
    assert _role() in system


async def test_an_append_prompt_gains_the_role(tmp_path: Path) -> None:
    _write_agent(
        tmp_path,
        "helper",
        "---\ndescription: d\nprompt_mode: append\n---\nHelp out.",
    )

    system = await _spawn_system_prompt(tmp_path, {"subagent_type": "helper"})

    bridge = _prompts_module().SUB_AGENT_BRIDGE  # type: ignore[attr-defined]
    assert system.startswith(f"You are Tau.\n\n{bridge}\n\n{_role()}\n\n<active_agent")
    assert system.count("<sub_agent_role>") == 1


async def test_a_child_that_cannot_spawn_is_not_told_when_to(
    tmp_path: Path,
) -> None:
    # `explore` is limited to read and bash; an isolated child has no
    # extension tools at all.
    # Only the first can have a prompt that names a tool it has lost, and
    # only on a Tau that cannot rebuild the prompt.
    explore_listed = not _tau_rebuilds_prompts()
    for arguments, agent_listed in (
        ({"subagent_type": "explore"}, explore_listed),
        ({"isolated": True}, False),
    ):
        system = await _spawn_system_prompt(tmp_path, arguments)
        assert _role(can_spawn=False, agent_listed=agent_listed) in system
        assert SPAWN_RULE not in system


async def test_a_child_at_the_limit_is_not_left_with_a_prompt_that_lists_agent(
    tmp_path: Path,
    delegating: Delegators,  # noqa: F811
) -> None:
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))

    await _agent_tool(runtime).execute(
        "call-1", {"prompt": "do the task", "description": "d"}
    )

    first = str(delegating[0].calls[0]["system"])
    last = str(delegating[2].calls[0]["system"])
    assert _role() in first and SNIPPET in first
    assert SPAWN_RULE not in last
    if _tau_rebuilds_prompts():
        # The prompt was rebuilt without the tools the child lost.
        assert SNIPPET not in last
        assert _role(can_spawn=False) in last
    else:
        # The prompt still lists the tool, and the role says it is not there.
        assert SNIPPET in last
        assert _role(can_spawn=False, agent_listed=True) in last


async def test_nested_append_children_have_one_role_each(
    tmp_path: Path,
    delegating: Delegators,  # noqa: F811
) -> None:
    _write_agent(
        tmp_path,
        "helper",
        "---\ndescription: d\nprompt_mode: append\n---\nHelp out.",
    )
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))
    # general, then a helper under it, then a helper under that.
    delegating.spawn_arguments["subagent_type"] = "helper"

    await _agent_tool(runtime).execute(
        "call-1", {"prompt": "do the task", "description": "d"}
    )

    general, helper, last = (str(p.calls[0]["system"]) for p in delegating)
    bridge = _prompts_module().SUB_AGENT_BRIDGE  # type: ignore[attr-defined]
    for system in (general, helper, last):
        assert system.count("<sub_agent_role>") == 1
    for system in (helper, last):
        assert system.startswith("You are an expert coding assistant")
        assert system.count(bridge) == 1
    assert _role() in general
    assert _role() in helper
    # The last is at the limit. Its prompt is its parent's, which lists the
    # `agent` tool, and it does not keep its parent's rule on when to spawn.
    assert _role(can_spawn=False, agent_listed=True) in last
    assert SPAWN_RULE not in last


async def test_a_fork_is_told_in_its_task_and_keeps_the_parent_prompt(
    tmp_path: Path,
) -> None:
    runtime = _load_runtime(tmp_path)
    runtime.bind(_fork_parent(tmp_path))
    provider = CapturingProvider([_text_stream("done")])
    _patch_fork_resolution(_extension_module(), provider)

    await _agent_tool(runtime).execute(
        "call-1", {"prompt": "go", "description": "d", "subagent_type": "fork"}
    )

    call = provider.calls[0]
    # Byte-identical to the parent's, for the shared prompt cache.
    assert call["system"] == "You are Tau."
    task = call["messages"][-1].text  # type: ignore[index]
    assert task.startswith("<fork_task>\n")
    assert (
        "Do\nthis task yourself with your own tools: do not hand it, or a"
        " reworded copy\nof it, to another agent." in task
    )


DELEGATE_LINE = (
    "You are a sub-agent yourself: spawn only for a separate subtask, never to"
    " pass on the task you were given."
)


def _agent_description(tools: list[object]) -> str:
    return next(tool.description for tool in tools if tool.name == "agent")  # type: ignore[attr-defined]


async def test_agent_tool_says_when_not_to_use_it(tmp_path: Path) -> None:
    runtime = _load_runtime(tmp_path)

    description = _agent_tool(runtime).description

    assert "When not to use: do the task yourself when a few tool calls" in description
    assert "Never forward your whole task to a single agent." in description
    assert DELEGATE_LINE not in description  # the top session is no sub-agent
    assert "- general: " in description and "does the work itself." in description


async def test_a_childs_agent_tool_says_it_is_a_sub_agent(
    tmp_path: Path,
    delegating: Delegators,  # noqa: F811
) -> None:
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))
    parent_description = _agent_tool(runtime).description

    await _agent_tool(runtime).execute(
        "call-1", {"prompt": "do the task", "description": "d"}
    )

    child_description = _agent_description(delegating[0].calls[0]["tools"])  # type: ignore[arg-type]
    assert DELEGATE_LINE in child_description
    assert child_description.replace(f" {DELEGATE_LINE}", "") == parent_description
    # Tau's default prompt lists each tool by its snippet.
    assert "never to pass on your own task" in str(delegating[0].calls[0]["system"])


async def test_a_forks_agent_tool_matches_its_parents(
    tmp_path: Path,
    delegating: Delegators,  # noqa: F811
) -> None:
    """A fork shares its parent's prompt cache, which is keyed on the tools."""
    runtime = _load_runtime(tmp_path)
    runtime.bind(_fork_parent(tmp_path))
    parent_description = _agent_tool(runtime).description
    # The top session forks, and the fork spawns a general child.
    delegating.spawn_arguments["subagent_type"] = "general"

    await _agent_tool(runtime).execute(
        "call-1",
        {"prompt": "do the task", "description": "d", "subagent_type": "fork"},
    )

    fork, child = delegating[0], delegating[1]
    assert _agent_description(fork.calls[0]["tools"]) == parent_description  # type: ignore[arg-type]
    assert DELEGATE_LINE in _agent_description(child.calls[0]["tools"])  # type: ignore[arg-type]
