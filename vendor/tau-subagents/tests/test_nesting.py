"""The nesting ceiling: a backstop against sub-agents that re-delegate."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tau_agent.messages import AssistantMessage, ToolCall
from tau_ai import AssistantDoneEvent, AssistantStartEvent
from test_extension import (
    EXTENSION_DIR,
    RecordingSession,
    _agent_tool,
    _extension_module,
    _fork_parent,
    _load_runtime,
    _submodule,
    _text_stream,
    _wait_for,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class DelegatingProvider:
    """A model that passes its task on whenever it has an `agent` tool.

    It calls `agent` in answer to each prompt if the tool is offered, and
    otherwise answers; it answers each tool result, so each run ends.
    """

    def __init__(self, arguments: dict[str, object]) -> None:
        self._arguments = arguments
        self.calls: list[dict[str, object]] = []

    def stream_response(self, *, model, system, messages, tools, signal=None, **_):  # noqa: ANN001, ANN003, ANN202
        tool_list = list(tools or ())
        self.calls.append(
            {"system": system, "messages": list(messages), "tools": tool_list}
        )
        offered = any(tool.name == "agent" for tool in tool_list)
        if messages[-1].role == "user" and offered:
            events = [
                AssistantStartEvent(partial=AssistantMessage(model="fake")),
                AssistantDoneEvent(
                    reason="toolUse",
                    message=AssistantMessage(
                        content=[
                            ToolCall(
                                id=f"call-{id(self)}",
                                name="agent",
                                arguments=dict(self._arguments),
                            )
                        ],
                        stop_reason="toolUse",
                    ),
                ),
            ]
        else:
            events = _text_stream("did it myself")

        async def iterator():  # noqa: ANN202
            for event in events:
                yield event

        return iterator()

    def tool_names(self, call: int = 0) -> set[str]:
        return {tool.name for tool in self.calls[call]["tools"]}  # type: ignore[union-attr]


class Delegators(list[DelegatingProvider]):
    """The providers sessions created, in order, and what each one spawns."""

    def __init__(self) -> None:
        super().__init__()
        self.spawn_arguments: dict[str, object] = {
            "prompt": "do the task",
            "description": "d",
        }


@pytest.fixture
def delegating(
    _isolate_home: Path, monkeypatch: pytest.MonkeyPatch
) -> Delegators:
    """Make every session, at any depth, run a DelegatingProvider.

    Each child session imports its own copy of the extension, so the fakes
    are patched where every copy imports them from, before any copy loads.
    Returns the providers in the order sessions created them.
    """
    import tau_coding
    import tau_coding.provider_runtime

    extensions_dir = _isolate_home / ".tau" / "extensions"
    extensions_dir.mkdir(parents=True)
    (extensions_dir / "tau-subagents").symlink_to(EXTENSION_DIR)

    created = Delegators()

    def create(provider, model, thinking_level):  # noqa: ANN001, ANN202
        created.append(DelegatingProvider(created.spawn_arguments))
        return created[-1]

    monkeypatch.setattr(tau_coding, "load_provider_settings", lambda: None)
    monkeypatch.setattr(
        tau_coding,
        "resolve_provider_selection",
        lambda settings, provider_name=None, model=None: SimpleNamespace(
            provider=SimpleNamespace(name=provider_name or "fake"),
            model=model or "fake",
        ),
    )
    monkeypatch.setattr(tau_coding.provider_runtime, "create_model_provider", create)
    return created


def _set_max_depth(cwd: Path, value: int, **others: object) -> None:
    (cwd / ".tau").mkdir(parents=True, exist_ok=True)
    (cwd / ".tau" / "subagents.json").write_text(
        json.dumps({"maxDepth": value, **others})
    )


async def test_a_chain_of_delegating_agents_stops_at_the_default_limit(
    tmp_path: Path, delegating: Delegators
) -> None:
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))

    result = await _agent_tool(runtime).execute(
        "call-1", {"prompt": "do the task", "description": "d"}
    )

    # Three levels below the parent, and the third cannot spawn a fourth.
    assert len(delegating) == 3
    assert "agent" in delegating[0].tool_names()
    assert "agent" in delegating[1].tool_names()
    spawn_tools = _submodule("nesting").SPAWN_TOOLS  # type: ignore[attr-defined]
    assert not spawn_tools & delegating[2].tool_names()
    assert "jobs" in delegating[2].tool_names()  # only the spawn tools go
    assert "did it myself" in result.text


async def test_max_depth_setting_moves_the_limit(
    tmp_path: Path, delegating: Delegators
) -> None:
    _set_max_depth(tmp_path, 1)
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))

    await _agent_tool(runtime).execute(
        "call-1", {"prompt": "do the task", "description": "d"}
    )

    assert len(delegating) == 1
    assert "agent" not in delegating[0].tool_names()


async def test_a_fork_at_the_limit_keeps_its_tools_and_is_refused(
    tmp_path: Path, delegating: Delegators
) -> None:
    _set_max_depth(tmp_path, 1)
    runtime = _load_runtime(tmp_path)
    runtime.bind(_fork_parent(tmp_path))
    module = _extension_module()

    await _agent_tool(runtime).execute(
        "call-1",
        {"prompt": "do the task", "description": "d", "subagent_type": "fork"},
    )

    # The fork's tool list must match its parent's, so the tool stays and
    # the call is what gets refused.
    assert len(delegating) == 1
    fork = delegating[0]
    assert "agent" in fork.tool_names()
    refusal = fork.calls[1]["messages"][-1]  # type: ignore[index]
    assert refusal.role == "toolResult"
    assert refusal.text == module.depth_limit_message(1)  # type: ignore[attr-defined]
    assert "Do the task yourself" in refusal.text


async def test_a_background_child_inherits_its_depth(
    tmp_path: Path, delegating: Delegators
) -> None:
    _set_max_depth(tmp_path, 2)
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))
    # Every level spawns in the background, so each run starts in a task of
    # its own rather than inside its parent's tool call.
    delegating.spawn_arguments["run_in_background"] = True

    await _agent_tool(runtime).execute(
        "call-1",
        {"prompt": "do the task", "description": "d", "run_in_background": True},
    )
    # The second level has answered, and the first has answered its result.
    await _wait_for(
        lambda: len(delegating) == 2
        and len(delegating[1].calls) == 1
        and len(delegating[0].calls) == 2
    )

    assert "agent" in delegating[0].tool_names()
    assert "agent" not in delegating[1].tool_names()


async def test_a_queued_child_starts_at_its_own_depth(
    tmp_path: Path, delegating: Delegators
) -> None:
    _set_max_depth(tmp_path, 2, maxConcurrent=1)
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))
    background = {
        "prompt": "do the task",
        "description": "d",
        "run_in_background": True,
    }

    await _agent_tool(runtime).execute("call-1", background)
    second = await _agent_tool(runtime).execute("call-2", background)
    # The second waits, and starts from the task that ends the first.
    assert second.details["queued"] is True
    await _wait_for(
        lambda: len(delegating) == 4 and all(len(p.calls) >= 1 for p in delegating)
    )

    # Each child is at depth 1 and its own child at depth 2, the limit.
    child, grandchild, queued_child, queued_grandchild = delegating
    assert "agent" in child.tool_names()
    assert "agent" not in grandchild.tool_names()
    assert "agent" in queued_child.tool_names()
    assert "agent" not in queued_grandchild.tool_names()


async def test_a_resumed_child_counts_from_its_own_depth(
    tmp_path: Path, delegating: Delegators
) -> None:
    _set_max_depth(tmp_path, 2)
    runtime = _load_runtime(tmp_path)
    runtime.bind(RecordingSession(tmp_path))

    await _agent_tool(runtime).execute(
        "call-1", {"prompt": "do the task", "description": "d"}
    )
    assert len(delegating) == 2
    # The resume runs from the top session's context, outside the scope the
    # child was loaded in. The child still spawns at depth 2, the limit.
    await _agent_tool(runtime).execute(
        "call-2", {"prompt": "do it again", "description": "d", "resume": "agent-1"}
    )

    assert len(delegating) == 3
    assert len(delegating[0].calls) == 4  # the child ran a second time
    assert "agent" not in delegating[2].tool_names()


async def test_one_max_depth_holds_for_the_whole_tree(
    tmp_path: Path, delegating: Delegators
) -> None:
    _set_max_depth(tmp_path, 1)
    runtime = _load_runtime(tmp_path)
    module = _extension_module()
    nesting = _submodule("nesting")
    api = SimpleNamespace(context=SimpleNamespace(cwd=tmp_path))

    # A session below a top session whose limit is 5 uses 5, whatever the
    # settings it can read itself say.
    with nesting.nested(nesting.Nesting(depth=1, delegate=True, max_depth=5)):  # type: ignore[attr-defined]
        assert module.SubagentManager(api).max_depth == 5  # type: ignore[attr-defined]
    top = module.SubagentManager(api)  # type: ignore[attr-defined]
    assert top.max_depth == 1
    general = SimpleNamespace(fork=False)
    assert top._child_nesting(general) == (1, True, 1)  # noqa: SLF001


async def test_nesting_is_shared_between_copies_and_inherited_by_tasks(
    tmp_path: Path,
) -> None:
    _load_runtime(tmp_path)
    first = _submodule("nesting")
    _load_runtime(tmp_path)
    second = _submodule("nesting")
    assert first is not second  # two imports of the extension

    assert second.current_nesting() == (0, False, None)  # type: ignore[attr-defined]
    with first.nested(first.Nesting(depth=2, delegate=True)):  # type: ignore[attr-defined]
        assert second.current_nesting() == (2, True, None)  # type: ignore[attr-defined]

        async def read():  # noqa: ANN202
            return second.current_nesting()  # type: ignore[attr-defined]

        assert await asyncio.create_task(read()) == (2, True, None)
        assert await asyncio.to_thread(second.current_nesting) == (2, True, None)  # type: ignore[attr-defined]
    assert second.current_nesting() == (0, False, None)  # type: ignore[attr-defined]


async def test_max_depth_setting_parses(tmp_path: Path) -> None:
    _load_runtime(tmp_path)
    load = _extension_module().load_subagent_settings  # type: ignore[attr-defined]
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    assert load(cwd, home=home).max_depth == 3

    _set_max_depth(cwd, 5)
    assert load(cwd, home=home).max_depth == 5
    for dropped in (0, -1, 65, "4", True, 2.5):
        _set_max_depth(cwd, dropped)  # type: ignore[arg-type]
        assert load(cwd, home=home).max_depth == 3
