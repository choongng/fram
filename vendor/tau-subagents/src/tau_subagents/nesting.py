"""How deep in a chain of sub-agents the running code is.

Each child session loads its own copy of this extension (ADR 0004), and Tau
imports an extension loaded from a path under a fresh module name each time,
so a module-level variable here is a different object in the parent's copy
and the child's. The one `ContextVar` is therefore kept on a synthetic module
in `sys.modules`, which every copy finds by name. Its value is a plain tuple
for the same reason: a class defined here would be a different class in each
copy. The attribute name carries a version, so that a copy with a different
tuple shape uses a different variable instead of misreading this one.

A manager sets the variable around a child's load and run. The child's copy
of the extension reads it once, in `setup()`, which Tau calls during the
load. Code that runs in a copy of the context also sees it: an asyncio task,
or a call through `asyncio.to_thread`. A thread started any other way does
not.

The value is per process: a `tau` command that a sub-agent starts from a
shell is a top session again.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from contextvars import ContextVar
from types import ModuleType
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterator

_SHARED_MODULE = "_tau_subagents_shared"
#: Change the suffix whenever the shape of the stored tuple changes.
_SHARED_ATTRIBUTE = "nesting_v1"

#: The tools that start or manage sub-agents. A child at the nesting limit
#: loses all of them: without `agent` the other three have nothing to act on.
SPAWN_TOOLS = frozenset(
    {"agent", "get_subagent_result", "steer_subagent", "stop_subagent"}
)


class Nesting(NamedTuple):
    """Where a session sits in a chain of sub-agents."""

    #: 0 for the session a person started, 1 for its sub-agents, and so on.
    depth: int = 0
    #: Whether the session was given its task by another agent. False for
    #: the top session, and for a fork of it: a fork must send the same tool
    #: descriptions as its parent to share the parent's prompt cache.
    delegate: bool = False
    #: The `maxDepth` of the top session, which every session below it uses,
    #: so that a child reading other settings (in a worktree, say) cannot
    #: disagree with its parent. None in the top session: it reads its own.
    max_depth: int | None = None


def _variable() -> ContextVar[tuple[int, bool, int | None]]:
    shared = sys.modules.get(_SHARED_MODULE)
    if shared is None:
        shared = ModuleType(_SHARED_MODULE)
        sys.modules[_SHARED_MODULE] = shared
    variable = getattr(shared, _SHARED_ATTRIBUTE, None)
    if variable is None:
        variable = ContextVar(_SHARED_ATTRIBUTE, default=tuple(Nesting()))
        setattr(shared, _SHARED_ATTRIBUTE, variable)
    return variable


def current_nesting() -> Nesting:
    """The nesting of the session whose code is running."""
    return Nesting(*_variable().get())


@contextmanager
def nested(nesting: Nesting) -> Iterator[None]:
    """Run the body as a session at `nesting`."""
    variable = _variable()
    token = variable.set(tuple(nesting))
    try:
        yield
    finally:
        variable.reset(token)
