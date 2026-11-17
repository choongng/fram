# TUI: keep the layout still while the agent works

Rows that appear and vanish while a turn runs make the screen jump: the
transcript moves under the reader, or the prompt moves under the cursor. The
rule this note sets is that transient UI either has space reserved for it all
the time, or floats over other content. It never inserts rows into the layout.

## What was changed

While the model thinks with thinking tokens hidden, the TUI used to mount a
status row, "Thinking… Press Ctrl+T to show thinking tokens.", at the end of the
transcript, and remove it as soon as an answer or tool call followed. Each turn
therefore added a row and took it back.

The notice now has a row of its own, `#thinking-notice`, directly above the
prompt. The row is always one line tall and blank unless hidden thinking is in
progress, so the notice coming and going changes no layout. Its text starts in
the prompt's text column.

The notice first sat in the prompt's top border, as its `border_title`. Text on
the border read as part of the prompt rather than as a status, so it moved up a
row. The row costs the transcript one line.

In that row the notice sat directly under the transcript's last line, with no
gap. Then, when the answer arrived, its blank row and first line pushed the
history up two rows. Now, while thinking is hidden, the transcript ends with a
`ThinkingPlaceholder`: a blank row with the usual blank row above it, the same
shape as the first line of an answer or tool row (after a tool row it drops that
blank row; see below). The notice is lifted one row, over the placeholder, so it
has a blank row above it. The row that follows takes the placeholder's place,
and the history does not move back down. The scroll happens once, when thinking
starts, rather than when the answer lands.

- `TuiState.hidden_thinking_in_progress` decides from state: a run is going,
  thinking is hidden, no answer text is streaming, and the newest item is
  thinking.
- `_apply_activity_indicator` sets the notice text, along with the spinner and
  border colour it already drew. It runs on chrome refreshes, on each streamed
  message update and after Ctrl+T. It shows the notice only while there is
  something to say, so its lifted row never hides a transcript line. It lifts
  the notice only when the transcript's newest row is directly above it: the
  transcript follows the newest output, and no extension view, extension widget
  or queued-message list sits in between. Otherwise the notice keeps its own row
  rather than cover history or that other content.
- `TranscriptView.sync_thinking_placeholder` mounts the placeholder at the end
  of the transcript when hidden thinking starts, and again after any rebuild
  that left it elsewhere. It drops the placeholder when the run ends or thinking
  is shown.
- `TranscriptView.mount` retires the placeholder when the rows it mounts land
  after it. It hides the placeholder in the same call that inserts the row, so
  the next layout has the new row and not the placeholder, and no frame shows
  both or neither. This cannot be left to `pre_layout`: Textual drops hidden
  widgets from the arrangement before it calls `pre_layout`, so a change of
  `display` made there only takes effect on the following pass.
- When a tool row directly follows another, the two join with no blank row
  between them, and `TranscriptView.pre_layout` gives the placeholder the same
  treatment: after a tool row it drops its blank row too. The placeholder is
  then never taller than the row that replaces it. A tool row takes its place
  exactly, and an answer, which keeps its blank row, only pushes the history
  further up. Before this, each thinking step in a burst of tool calls pushed
  the history up two rows and let it fall back one, so the transcript bounced.

### Autocomplete floats above the prompt

Autocomplete suggestions used to sit in the layout under the prompt. Opening
them shrank the transcript and pushed the prompt up by as many as 18 rows, and
closing them moved it back.

`#autocomplete` is now `overlay: screen` with `position: absolute`, so it takes
no room in the layout. `_place_completions` gives it the prompt row's width,
sizes it to its rendered lines plus the border, and offsets it so its bottom
edge meets the prompt's top edge. It covers the bottom of the transcript and
leaves at least `COMPLETION_MIN_TRANSCRIPT_LINES` rows of it in view.

- `_completion_line_budget()` counts the rows above the prompt. It no longer
  depends on the list's own height, so the old "keep the largest measured
  height" logic and the initial terminal-fraction estimate are gone.
- The prompt moves when the terminal resizes or the prompt grows. The app
  subscribes to the screen's `screen_layout_refresh_signal` and places the list
  again when the prompt row's or main pane's region has changed. Comparing
  regions matters: placing the list updates its content, which triggers another
  layout pass.

## Prompts sent while a run is finishing

This one is about the app staying responsive rather than the layout staying
still. The TUI shows a run as working from the moment a prompt is submitted
until the session emits `agent_settled`. The session accepts a queued steering
or follow-up message for a shorter span: only while its agent is running. That
leaves gaps in which a run is live but cannot queue:

- before its first event, while input hooks and pre-prompt compaction run;
- between two agent runs of one turn, while the session compacts after a context
  overflow or fails over to another route before it retries;
- after the agent stops, while the session names itself and runs settle hooks.

`session.prompt` called in a gap does not queue. It runs a whole new turn. The
TUI used to call it from the Enter key handler whenever it showed a run as
working, so a prompt sent in a gap ran its turn inside the key handler: the app
handled no input and streamed nothing until the turn ended, and the two runs
overlapped in the session.

The rule now is that the key handler never runs a turn, and a live run is never
replaced. Three fields on the app carry it:

- `_active_prompt_run_id` is the run that owns the session, set in
  `_submit_prompt` before its first await and cleared by `_release_prompt_run`
  when the run's worker finishes. An interrupt clears it at once only if the
  worker has not reached the session yet; see "After an interrupt" below.
- `_held_prompts` is a queue of messages waiting for the session to take them.
- `_queueing_prompt` is the held message being offered to the session right now.
  It is out of the queue, so nothing else can take it, and still counts as held,
  so later submissions line up behind it.

The session side is `CodingSession.queue_prompt`. It runs the `input` hooks and
expands the text as `prompt` does, then queues the message only if the agent is
still running, and refuses it otherwise. It never starts a turn. `prompt` with a
`streaming_behavior` cannot give that promise: it awaits the hooks before it
looks at the agent, so the agent can stop in between, and then it runs a turn.
No check the TUI makes beforehand closes that window, which is why the entry
point lives in the session.

`queue_prompt` returns a `QueuePromptResult` whose `status` is `queued`,
`handled` (an `input` hook consumed the message, so nothing is left to deliver)
or `refused`. A refusal that comes after the hooks ran also carries
`prepared_text`, the text after the hooks and expansion. The message still has
to be delivered, and delivering the text the user typed would send it through
the hooks again: a hook that logs or counts would act twice on one message, and
a template would expand twice, which is not harmless when its body starts with
another `/` command. So `prompt` and `queue_prompt` both take `prepared=True`,
which says the text is a `prepared_text` and skips the hooks and the expansion.
Everything after that point is the ordinary code, including the `source` written
on the message's origin record.

A flag on `prompt` was chosen over a second method because the turn that follows
is the same in every respect; a second entry point would have to share all of
`prompt` but its first step. The text travels as a plain string, with the flag
next to it, so the held entry in the TUI can keep both the typed and the
prepared text.

On Enter, `_submit_prompt_from_editor` does one of two things:

1. Nothing owns the session, nothing is held, and the agent is not running:
   start a run and return.
2. Otherwise: append to `_held_prompts` and call `_queue_held_prompts`.

`_queue_held_prompts` offers held prompts to the session, oldest first, for as
long as the agent is running. If it is not, the call returns at once and the
prompt stays held. If it is, the handler awaits `queue_prompt`, which means it
awaits the session's `input` hooks, as queueing always has. What it can no
longer do is run a turn: a prompt the session refuses goes back to the front of
the queue. If the refusal carries a `prepared_text`, the entry that goes back
holds it next to the typed text, and every later delivery of that entry, queued
or as a run, sends the prepared text with `prepared=True`.

Held prompts are delivered at the first point the session can take them:

- The live run's own event loop, in `_run_prompt`, calls `_queue_held_prompts`
  after each event. When the agent is running again, which is from the run's
  first event and again when a retry starts, each held prompt is queued with the
  session as the steering or follow-up message the user asked for. So a prompt
  sent during compaction steers the retry instead of abandoning it. The agent's
  last event is skipped: the agent still reports running until that event is
  consumed, but it reads its queues no more, so a message queued then would be
  stranded. `CodingSession` says so itself, see "The end of a run" below; the
  skip remains for sessions that do not.
- When the run's worker finishes with prompts still held, as it does after a
  settle, `_release_prompt_run` schedules `_start_next_held_prompt` with
  `call_later`. It starts the oldest held prompt as the next run, whether it was
  sent as steering or as a follow-up. Any others stay held and are queued with
  that run when its agent starts, as if they had been sent during it. A run that
  emits no events, because an input hook handled its prompt, still releases the
  session, and the next held prompt starts.
- When the run ends while an offer is in flight, the offer comes back refused
  after the worker is done. `_queue_held_prompts` then schedules
  `_start_next_held_prompt` itself.

Because the earlier run is never replaced, its remaining events are applied as
usual: its settle clears `state.running`, an error it raises while settling gets
its error row, and that row comes before the held prompt's turn.

Extension messages take the same line. `_deliver_extension_message` starts a run
only under the conditions of path 1. Otherwise it holds the message as a
follow-up, so a message that arrives just as a run ends cannot start a run ahead
of the prompts already waiting.

Held prompts the user typed show in the queued-message list above the prompt as
"waiting" rows, and Up on an empty prompt pulls the latest one back for editing.
The prompt being offered stays listed but cannot be pulled back until the
session answers; taking it then would deliver it twice. Held extension messages
are neither listed nor editable.

A held prompt with a prepared text is still listed, and edited, as it was typed.
Up is allowed on it. Sending the edited text is a new submission and goes
through the hooks, as it does for a message pulled back from the session's own
queue, whose hooks have also run. The transcript row of a run started from a
prepared text shows the typed text until the session confirms the message, and
the prepared text from then on, the same as for any prompt a hook rewrites.

What drops held prompts, and what does not:

- Escape keeps its meaning: it interrupts the live run, whatever that run is
  doing. During a settle that cancels the rest of the settle, as it did before.
  The held prompts are dropped with the run they waited on, so none of them
  reaches the session afterwards; they are still in prompt history. A prompt
  sent after the Escape is a different matter, described below.
- `/new` interrupts the same way and drops them too. A run cancelled when the
  app exits also takes them with it.
- `/resume` and the session picker are refused while a run owns the session or
  prompts are held, with the message the session already gives while its agent
  runs. Refusing was chosen over dropping: the session refuses a resume in the
  middle of a run without disturbing it, and the gaps of a run should behave
  like the rest of it. A held prompt therefore never runs in a session it was
  not written for.
- A run cancelled any other way, for example by another exclusive worker, cannot
  deliver its held prompts. They are dropped, and a warning names them.
- A prompt the session fails to queue, for example because its template does not
  expand, is dropped alone, and the error names it. The others carry on.

One thing was tried and left out: a second worker that waits on the first
worker. Cancelling the waiter did not stop its prompt, because Textual's
`Worker.wait()` reports the waiter's own cancellation as a worker error, and
replacing the first run meant its settle and errors were never applied.

### After an interrupt

Escape cancels the run's worker. Textual reports that worker as cancelled at
once (`is_cancelled`), although it is not finished until the cancellation has
unwound it, and the app stops tracking it. The run is not finished either.
`CodingSession.prompt` still has its `finally` to run in that worker: it stops
the agent, saves the messages the run produced and calls the `agent_settled`
hooks, each of them awaited. The interrupt used to release the session
immediately, so a prompt sent in that window started a second run beside the
first one's cleanup: its provider request went out, and its own settle began,
while the first run's settle hooks were still running.

The interrupted run now keeps the session until its worker is done. Two more
fields carry that:

- `_session_prompt_run_id` is the run whose worker is inside `session.prompt`.
  `_run_prompt` sets it on entry and clears it on exit. A worker cancelled
  before that point has nothing to undo, and might never run at all, so an
  interrupt releases such a run at once, as before.
- `_interrupted_run_id` is the run an interrupt cancelled inside the session,
  kept until `_run_prompt` exits.

What the user sees:

- Prompts held before Escape are dropped by it, as before.
- A prompt sent after Escape, while the interrupted run finishes, is held and
  shows as a "waiting" row. When the worker is done it starts as the next run.
  It was written after the interrupt, so the interrupt does not apply to it.
  Dropping it would also make Escape followed by a correction, the usual reason
  to press Escape, lose the correction whenever the cleanup is slow.
- A second Escape during the cleanup drops the prompts held since the first one
  and calls `CodingSession.abandon_settle_hooks()`. Settle hooks that are
  running are cancelled and ones that have not started are skipped, which is the
  way out of a settle hook that never returns. Stopping the agent and saving the
  run's messages are not interrupted. Nothing in the key handler waits for the
  worker, so neither Escape nor quitting can hang on it.
- Quitting cancels the worker like any other, and the prompts held behind it go
  with the app.
- `/compact`, `/resume` and the session picker are refused during the cleanup,
  as they are during the rest of a run.

The second Escape does not cancel the worker again. A cancellation lands on
whatever the worker is awaiting, and after the first Escape that can be the
write of the "Tool call interrupted by user" result to the session file. An
earlier version of this change cancelled it there. The harness then skipped the
lines that mark the run as over, and the session reported a running agent for
good: every later prompt failed, and only a restart recovered. `AgentHarness`
now resets that state however its cleanup ends, so a second cancellation from
anywhere else, such as quitting, cannot do the same. The settle hooks run in a
task of their own inside the session, so that they can be stopped without
cancelling the worker.

Held messages are not offered to the session during the cleanup, even though its
agent still reports running until the session has closed it: an interrupted
agent reads its queues no more, and `can_queue_messages` says so.

`_run_prompt` also closes the session's event stream itself when it stops
reading early. Without that, a worker cancelled while the app was applying an
event, and not inside the session, left the session's run suspended until the
stream was garbage collected, at which point the cleanup ran in another task
with nothing owning the session.

`/new` during the cleanup counts as a second interrupt: it gives up the settle
hooks, as a second Escape does. It then replaces the session at once, without
waiting for the old run's worker to finish saving. Not waiting is older than
this change and is left alone; prompts sent afterwards are held until the old
run's worker is done.

### The end of a run

`AgentHarness.is_running` stays true until the run's event stream is closed. The
run reads its queues for the last time before it emits `agent_end`, and does not
read them after a cancel. A message queued in between is not lost, but no run is
left to deliver it: it waits in the queue until the user happens to send
something else.

That window was open to extensions. A handler of `agent_end` that sent a
message, which is where an extension that reacts to a finished run sends one,
had it queued with the run that had just ended, and no turn followed.

`AgentHarness.delivers_queued_messages` turns false when the current run will
not read its queues again, and `CodingSession.can_queue_messages` passes it on.
`queue_prompt` refuses on it, and the extension runtime chooses between queueing
and asking the frontend for a turn on it, where both used to look at
`is_running`. In the TUI the message then takes the ordinary path of an
extension message that arrives in a gap: it is held behind any waiting prompts
and starts the next run when the worker is done.

The property cannot see the other way a message is stranded. A run that ends on
a provider error, the turn limit, a final report or a `finish_turn` hook does
not read its queues after its last turn, and during that turn nothing says it is
the last. A message queued then, typed or sent by an extension, stays queued
when the run ends: the TUI is idle, the message shows as queued, `/compact` and
`/tree` are refused, and the next prompt delivers it. This is older than held
prompts and is not fixed here;
`dev-notes/architecture/queued-steering-follow-ups.md` says why.

## Next steps

- Audit the other rows that change height while a turn runs: `#queued-messages`,
  `#compact-session-info` and the extension slots.

## How it maps to Pi

This is a frontend change only. The agent still emits the same thinking events;
the TUI chooses where to show them.

## Testing

```bash
uv run pytest tests/test_tui_thinking_toggle.py
uv run pytest tests/test_tui_app.py -k "thinking or completion"
uv run pytest tests/test_tui_app.py -k "held or settl or between_agent_runs"
uv run pytest tests/test_coding_session.py -k queue_prompt
```

`test_hidden_thinking_moves_nothing_on_screen` records every transcript row's
place and the prompt's region once hidden thinking starts. It streams the rest
of the thinking and checks that nothing moves, then that the answer's first line
lands on the notice's row with the rows above it and the prompt where they were.
Other tests check that a row mounted after the placeholder retires it in the
same call, that the placeholder goes when a run ends mid-thinking, and that the
notice keeps its own row when the transcript is scrolled back or queued messages
or an extension widget sit above the prompt.

`test_completion_list_floats_above_prompt_and_moves_nothing` opens and closes
the list and checks that the transcript, prompt and session-info regions stay
put, and that the list's bottom edge meets the prompt.
`test_completion_list_follows_prompt_after_resize` checks that the list moves
with the prompt when the terminal shrinks.
