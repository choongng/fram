---
title: "Queued Steering And Follow-ups"
---

This slice adds Pi-style message queueing while an agent run is active.

## What Was Added

`tau_agent.AgentHarness` now owns two prompt queues:

- steering messages, injected after the current assistant turn and any tool batch
- follow-up messages, injected only when the run would otherwise stop

The harness exposes `steer()`, `follow_up()`, `clear_queues()`,
`queued_messages`, `pending_message_count`, and `is_running`. Direct overlapping
`prompt()` and `continue_()` calls are rejected so callers cannot mutate one
transcript from two active runs.

`is_running` does not say that a queued message will be delivered. A run reads
its queues for the last time before it emits `agent_end`, and not at all after a
cancel, but it runs until its event stream is closed. A message queued in that
stretch stays in its queue for the next run, whenever that is.
`delivers_queued_messages` is the narrower test: false once the current run will
not read the queues again. Callers that choose between queueing a message and
delivering it some other way use it.

True is not a promise of delivery. A run reads its queues between turns and
cannot know during a turn that it is the last one. A provider error, the turn
limit, a final report and a `finish_turn` hook that ends the run all stop it
without another read, so a message queued during that turn stays queued as well.
Nothing delivers it when the run ends, in any frontend: it shows as a queued
message and goes with the next prompt. Delivering it at the end of the run was
considered and left out. What is left in the queues then is a mix: messages the
run missed, and messages an extension sent with `trigger_turn=False` after the
last read, which are meant to wait. Nothing records which is which, and a turn
started for the first kind would also start for the second, after every run.

`run_agent_loop()` accepts provider-neutral queue-drain callbacks. When a queue
drains, the loop appends the queued user message to the same message list used
for normal prompts and emits ordinary user `MessageStartEvent` and
`MessageEndEvent` values before the next provider call.

`QueueUpdateEvent` reports pending steering and follow-up text for frontends.
The Textual TUI uses it for status-line queue counts.

## Coding Session Boundary

`CodingSession.prompt()` keeps ordinary non-running prompts unchanged. While the
harness is running, callers must pass one of:

```python
session.prompt(text, streaming_behavior="steer")
session.prompt(text, streaming_behavior="follow_up")
```

The session expands `/skill:<name>` prompt text before queueing. It does not
persist a queued message at queue time. Persistence still happens after the
active run finishes, when the harness has injected queued user messages into the
transcript.

`CodingSession.queue_prompt()` is the queue-only form. `prompt()` awaits the
`input` hooks before it looks at the harness, so a caller that saw the harness
running can still get a whole turn when the harness stops in between.
`queue_prompt()` runs the same hooks and expansion, queues only if the harness
still delivers queued messages afterwards, and refuses otherwise, so the caller
keeps the message. A frontend that queues from an input handler uses it to be
sure the handler never runs a turn.

`CodingSession.can_queue_messages` passes on the harness's
`delivers_queued_messages`. The extension runtime decides on it where a message
from `send_user_message` or `send_custom_message` goes: into the run's queue, or
to the frontend's turn-requested callback, which starts a turn for it. A message
sent from an `agent_end` handler therefore gets a turn of its own instead of
waiting in the queue of the run that just ended.

Its `QueuePromptResult` has a `status` of `queued`, `handled` or `refused`. A
refusal after the hooks ran carries `prepared_text`. The caller delivers that
text with `prepared=True`, on `queue_prompt()` or `prompt()`, which skips the
hooks and the expansion so that neither runs twice on one message.

## TUI Behavior

The built-in Textual frontend maps:

- `Enter` while running to steering queueing
- `Alt-Enter` while running to follow-up queueing
- `Up` on an empty prompt while running to edit the latest queued follow-up

Pending queues are visible above the prompt while the run continues. If several
follow-ups are queued, edit pulls the most recently queued follow-up back into
the prompt and removes it from the queue. Once a queued message is injected, it
appears in the transcript as a normal user message and the pending queue display
updates.

## Boundary

Queue ownership lives in `tau_agent`, prompt expansion and persistence stay in
`tau_coding`, and Textual only decides keybindings and presentation. Provider
adapters do not know about queued messages; they receive the updated transcript
on the next provider call.

## Tests

The behavior is covered by:

```text
tests/test_agent_loop.py
tests/test_agent_harness.py
tests/test_coding_session.py
tests/test_tui_adapter.py
tests/test_tui_app.py
```
