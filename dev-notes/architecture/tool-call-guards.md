# Tool call guards

## What was added

`src/tau_coding/guards.py` holds `ToolCallGuards`, which a `CodingSession` uses
to refuse tool calls that would make no progress. There are two checks so far:

- **Repeated calls.** A call that ran twice among the latest ten calls, with the
  same result each time, is refused. The refusal says how many times it ran,
  quotes the result once, and says what to do next.
- **A tool run as a command.** A `bash` line whose command is one of the
  session's tools, when no program by that name is on `PATH`.

Eight refusals in a row cancel the run.

## Why it exists

A model can lock onto one call, or two taken in turn, and repeat it until the
turn budget runs out. Each repeat costs a full request. "Command not found" also
tends to bring on another try, because it does not say that the name is a tool.

## How it works

The session binds two harness hooks:

- `before_tool_call` is `_refuse_unproductive_call`, which asks the guards.
- `after_tool_call` is `_after_tool_call`, which buffers the output and records
  what the call gave.

`ToolCallGuards` only records calls it let through. The agent loop answers some
calls without asking the guards, such as one cut off at the output limit, and a
refusal is not a result either. None of those count as having run.

`AgentTool.changes_workspace` marks tools such as `write` and `edit`. After one
runs, earlier results of other calls no longer count, because the same read may
now give something else. Changing calls stay on record, so a change repeated
with the same result is still caught.

A `bash` command may change anything, and nothing says whether it did. One whose
result differs from the last time that command ran counts as a change; one that
gives the same result again does not. A failing test run, a fix through `sed`,
and the same test run again is therefore not a repeat, while two commands taken
in turn with unchanging results are still caught, one round later than other
tools.

Both hooks must be set again when a session adopts a replacement harness after
`/new` or `/resume`, or the hook stays bound to the discarded session.

State resets at the start of each run.

## Design rules

Guards steer with refusals. They never edit the prompt prefix or the thinking
level, so cached prompts stay valid.

The numbers are heuristics, and the guards' module says what each one allows.

## How it maps to Pi

The loop and its hooks stay in the portable harness. The guards are coding-agent
policy, so they live in `tau_coding` and reach the harness through
`before_tool_call`.

## How to test it

`uv run pytest tests/test_repeat_breaker.py`

## Malformed calls

Some models end a tool call's last string with a chat template's control tokens,
running the reply's next call or reasoning into the argument. Others write a
tool call as text and end the turn, so nothing runs.

- `check` looks for template tokens in every string of a call's arguments. Where
  each leak sits in a top-level argument the schema declares and does not
  require, or in one that only labels the call, the tail is cut at the token and
  the call runs with a note in its result. Otherwise the call is refused, with
  the call to send instead.
- A reply that ended normally with template call markup in its text gets a
  follow-up asking for a real call, at most twice in a row and four times a run.
  A reply that only names `<tool_call>` in prose is left alone.

Refusing a leaked call counts toward the eight-refusal stop like any other.
