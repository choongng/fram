---
title: Build your own frontend
description: Advanced — drive Tau's coding session from your own UI by consuming its event stream.
---

> [!CAUTION] Advanced
> This page is for building a *new frontend* on top of Tau's core. If you just want
> to use Tau, see [The interactive session](/guides/tui/). The APIs here are
> Python and assume you've read the [architecture overview](/internals/architecture/).

Tau's Textual app is one frontend, not the architecture. A custom UI plugs into
the same primitives the built-in TUI uses:

```text
CodingSession   — owns the coding-agent environment
AgentEvent      — describes assistant text, tool calls, results, errors
Frontend state  — belongs to your UI
```

The reusable `tau_agent` package stays independent of terminal frameworks,
widgets, keybindings, config paths, and slash-command UX. Build against
`tau_coding.session.CodingSession`, not Textual widgets.

`CodingSession` provides the environment (provider/model, tools, persistence,
skills, prompt templates, project context, slash-command handling, compaction).
Your frontend provides the interface (prompt input, transcript rendering, command
entry, cancellation, pickers).

## Minimal event loop

```python
async for event in session.prompt(user_text):
    render_event(event)
```

The stream yields provider-neutral `CodingSessionEvent` values: portable
`AgentEvent` values from `tau_agent.events` plus session-level values from
`tau_coding.events` (see [the agent loop](/internals/agent-loop/)).
Render from these, never from provider-specific chunks. Use `agent_start` to
enter the running state and `agent_settled`—not merely `agent_end`—to leave it,
because automatic compaction, retry, or queued continuation may follow an
`agent_end`. Provider failures arrive as assistant messages whose
`stop_reason` is `"error"`, followed by the normal turn/run lifecycle.

## Steering and follow-ups

If the user submits while a run is active, queue instead of starting a second
run:

```python
async for event in session.prompt(user_text, streaming_behavior="steer"):
    adapter.apply(event); redraw(state)
```

Use `streaming_behavior="follow_up"` for a prompt that waits until the run would
otherwise stop. Overlapping `session.prompt(...)` calls without
`streaming_behavior` are rejected so two runs can't mutate one transcript.
`QueueUpdateEvent` carries pending queued text for badges/status.

`session.prompt(...)` queues only while the agent is running. At any other time
it runs a whole turn, and the agent can stop while the session awaits its
`input` hooks, after your own check. A frontend that queues from an input
handler, and must not run a turn there, calls `queue_prompt` instead:

```python
result = await session.queue_prompt(user_text, streaming_behavior="steer")
if result.status == "refused":
    held = (user_text, result.prepared_text)
```

It never starts a turn. `result.status` is `queued`, `handled` when an `input`
hook consumed the message, or `refused` when no run would deliver it. The call
yields no events, so read `session.queue_update_event()` afterwards to refresh
badges.

A run stops reading its queues a little before `session.is_running` turns false:
from its `agent_end` event on, and once it has been cancelled. `queue_prompt`
refuses there too. `session.can_queue_messages` is the same test without the
call: false once the current run will not read its queues again. True is not a
promise. A run that ends on a provider error or at its turn limit does not read
its queues after that turn, and a message queued during it stays queued for the
next prompt.

On a refusal, keep the message and submit it as the next run. If
`result.prepared_text` is set, the hooks and prompt expansion have already run,
and that text is their result. Deliver it with `prepared=True`, which skips
both, so a hook does not act twice on one message:

```python
user_text, prepared_text = held
if prepared_text is None:
    events = session.prompt(user_text)
else:
    events = session.prompt(prepared_text, prepared=True)
```

`queue_prompt` takes `prepared=True` as well, for offering the message to a
later agent run. Keep showing `user_text` while the message waits.

A cancelled run is not over when you stop reading its events. Close the event
stream (`await events.aclose()`) and wait for that before you start the next
run: the session stops the agent, saves the run's messages and awaits the
`agent_settled` hooks there. A prompt submitted meanwhile should be held like a
refused one. Starting it at once runs two turns on one session. If the user
interrupts again while you wait, call `session.abandon_settle_hooks()` and do
not cancel the task that is closing the stream: the hooks are stopped or
skipped, and the save is left to finish.

## Slash commands

Slash commands belong to `tau_coding`. Before treating input as a prompt:

```python
result = session.handle_command(text)
```

If `result.handled`, apply the requested effect (`exit_requested`,
`clear_requested`, `new_session_requested`, `compact_summary`, `message`) and
show reference/status output *outside* the durable conversation. If
`result.compact_summary is not None`, call `await session.compact(result.compact_summary)`
(an empty string means "use the built-in prompt as-is").

`/skill:<name>` is intentionally **not** a command — pass it through to
`session.prompt(...)`, which expands it before the run.

## Restoring and switching sessions

Initialize the visible transcript from `session.messages` (the built-in
`TuiState.load_messages()` is a reference). `ToolResultMessage` preserves
structured metadata (e.g. edit patches), so you can render restored tool results
without reading JSONL directly.

For session switching, use `tau_coding.session_manager.SessionManager` —
`list_sessions(session.cwd)`, then `await session.resume(session_id)` (or load a
fresh `CodingSession` with `storage=jsonl_session_storage(record.path)`), then
rebuild the transcript from `session.messages`.

## Cancellation, pickers, keybindings

- Cancel with `session.cancel()` — keep consuming events until the stream ends.
- Read picker data directly from the session: `command_registry.list_commands()`,
  `skills`, `prompt_templates`, `available_model_choices`, `available_models`,
  `available_providers`, `thinking_level`, `available_thinking_levels`,
  `session_manager`. For model changes from another provider, call
  `set_provider(...)` then `set_model(...)`.
- Keybindings and themes are **frontend policy**. The built-in app reads
  `~/.tau/tui.json` via `tau_coding.tui.load_tui_settings()`, but your UI can
  ignore it.

## What not to depend on

Avoid coupling to private `CodingSession` attributes, provider-specific response
chunks, Textual internals, or the raw JSONL structure (use `SessionManager` /
`CodingSession`). Stick to the event, message, tool, harness, and session
primitives.

> [!NOTE]
> The full per-phase build journals for these systems live in the repo under
> `dev-notes/` (see [Contributing](/contributing/)).
