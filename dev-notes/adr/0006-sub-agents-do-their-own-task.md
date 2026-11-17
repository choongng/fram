# ADR 0006 — Sub-agents do their own task; nesting has a ceiling

## Status

Accepted (2026-11-17). This record replaces the "No depth limit" paragraph of
ADR 0004. The other decisions of ADR 0004 stay.

## Context

A parent session asked a `general` sub-agent to test a skill. The sub-agent did
not do the test. Its only action was a call to the `agent` tool with a reworded
copy of its task. The next sub-agent did the same. Two chains became 95 and 113
sessions deep before one session made a real tool call. One of them took 561
seconds for one tool use.

Four facts caused this:

- Children load the extension (ADR 0004). Thus each child has the `agent` tool.
- The `general` type has no system prompt. The child ran on Tau's default coding
  prompt. No text told the child that it was a sub-agent. Only append-mode types
  got the `<sub_agent_context>` block.
- The description of the `agent` tool told the model how to delegate. It did not
  tell the model when not to delegate.
- There was no depth limit. ADR 0004 said that a child's copy of the extension
  could not learn its depth.

## Decision

The primary fix is clear instruction. The depth limit is only a backstop.

### 1. Each child gets a role section

`prompts.sub_agent_role` makes this section:

```text
<sub_agent_role>
You are a sub-agent. Another agent gave you this task to do, not to pass on.
Do the work yourself with your own tools, then report what you found or did.
Do not hand your task, or a reworded copy of it, to another agent. Spawn an
agent only for a clearly separate piece of work that can run alongside yours,
and never when you could do it with a few tool calls.
</sub_agent_role>
```

The last three lines are left out when the child has no `agent` tool. If the
prompt of that child can still name the tool (see Consequences), this line takes
their place: "You have no agent tool; ignore any mention of one."

The section goes into each prompt shape:

- No prompt of its own (`general`): Tau's default prompt stays. The section is
  passed as `CodingSessionConfig.append_system_prompt`, and Tau adds it after
  the default prompt.
- A replace-mode prompt: the same field. Tau adds the section after the agent
  body.
- An append-mode prompt: the section is put after `<sub_agent_context>` in the
  full override. The override starts with the prompt of the session that spawns.
  If that session is a sub-agent, its prompt has a role section and possibly a
  `<sub_agent_context>` block already. The builder removes them first. Thus the
  child has one of each, and its role section is the one for its own depth.

A fork is different. Its system prompt must be the same bytes as the parent's
prompt, or the two do not share a prompt cache. Thus the instruction goes at the
start of the fork's task message, in `<fork_task>`.

### 2. The `agent` tool says when not to use it

The description has a new paragraph for all sessions:

```text
When not to use: do the task yourself when a few tool calls would do it. Use
agents for independent pieces of work that can run in parallel, or to keep
heavy reading out of your context. Never forward your whole task to a single
agent.
```

In a child session, one more sentence follows:

```text
You are a sub-agent yourself: spawn only for a separate subtask, never to
pass on the task you were given.
```

A fork keeps the wording of its parent, because the tool descriptions are part
of the cached prefix. The description of `general` now ends with "does the work
itself".

### 3. Nesting stops at `maxDepth`

The `maxDepth` setting has a default of 3. The session that a person started has
depth 0. A child at depth `maxDepth` cannot spawn:

- A child that is not a fork loses the `agent`, `get_subagent_result`,
  `steer_subagent`, and `stop_subagent` tools. The same filter that applies a
  `tools:` allow-list removes them.
- A fork keeps its tools, because its tool list must match the parent's. The
  `agent` call returns a plain message. The message tells the model to do the
  task itself.

### How a child learns its depth

ADR 0004 found no channel for this. `nesting.py` adds one. It is a `ContextVar`
that holds a tuple of the depth, a "delegate" flag, and the limit. The top
session reads `maxDepth` from its settings and each session below it uses that
value. A child that reads other settings, for example in a worktree, thus cannot
disagree with its parent.

Tau imports a path-loaded extension under a new module name for each session. A
module-level variable would thus be a different object in each copy. The
`ContextVar` is kept on a synthetic module in `sys.modules`, and each copy finds
it by name. When Tau loads the extension from a package entry point, there is
only one copy, and the same code works. The name of the attribute has a version
suffix. A later copy with a different tuple shape uses a different variable, and
does not misread this one.

The manager sets the variable around the load and the run of each child. The
child's manager reads it one time, in `setup()`, which Tau calls during the
load. The manager does not read it again at spawn time. A queued run or a
scheduled run can start from a different task context, and the value there could
be wrong. A resumed run (`resume=<id>`) also runs outside the scope. Code that
runs in a copy of the context sees the value: an asyncio task, or a call through
`asyncio.to_thread`. A thread that starts in a different way does not.

The limit is for one process. If a sub-agent starts a `tau` command from a
shell, that command is a top session at depth 0.

## Consequences

- A chain of sessions that only delegate now stops at depth 3. Before that
  point, the prompt and the tool description tell each session not to start the
  chain.
- A child's request has a different `agent` description from the parent's
  request. This costs nothing: a child that is not a fork has a different system
  prompt already, and shares no cache with the parent.
- Tau builds the system prompt, with its list of tools, before the filter for
  the limit or for a `tools:` allow-list runs. On a Tau that has
  `CodingSession.refresh_system_prompt`, the extension calls it after the
  filter, and the prompt lists only the tools that stay. On an older Tau, and in
  an append-mode child (its prompt is the parent's prompt), the list can still
  name `agent`. There, the role section says that the tool is not available. The
  tool is not in the request in each case, so the model cannot call it.
- A custom agent type cannot turn the role section off. If a type must
  orchestrate other agents, its body can say so, and the section still allows a
  spawn for separate work.

## How to test

`tests/test_delegate_role.py` checks the role section in each prompt shape, the
fork task, and the tool description in a parent, a child, and a fork.
`tests/test_nesting.py` runs a fake model that calls `agent` each time the tool
is offered. It checks that the chain stops at the limit, that `maxDepth` moves
the limit, that a fork at the limit is refused, and that a background child, a
queued child, and a resumed child each count from the correct depth.
