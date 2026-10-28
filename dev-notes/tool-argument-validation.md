# Tool-call arguments checked against the schema

## What

Before a tool runs, `tau_agent` checks the call's arguments against the JSON
schema the tool declares in `AgentTool.parameters`. If they do not fit, the
tool is not run. The model gets one tool error that lists every problem, each
with its path:

```text
Invalid arguments for read: missing required "path"; "limit" must be integer,
got string; "mode" must be one of "text", "bytes", got "lines"; unknown
argument "colour"; missing required "ranges[1].start". Send the call again
with these fixed.
```

(The error is one line. It is wrapped here.)

- The checker is `src/tau_agent/tool_arguments.py`: `argument_problems(schema,
  arguments)` returns the list of problems, and `invalid_arguments_message`
  builds the error.
- `_execute_tool_call` in `src/tau_agent/loop.py` runs it after the tool is
  found and before `_run_tool`. A refused call ends like any failed call: a
  `ToolExecutionEndEvent` and a `ToolResultMessage`, both with `is_error`. No
  new event or field was added.
- Arguments that pass reach the tool as sent. Nothing is coerced: `"10"` for
  an integer is refused, not turned into `10`.

## What the checker covers

It covers the JSON Schema keywords Tau's tools use:

- `type`: one name or a list of names. The names are `string`, `integer`,
  `number`, `boolean`, `array`, `object` and `null`. `true` is not an integer
  or a number. `10.0` is an integer, as in JSON Schema.
- `required`: each missing key is named.
- `properties`: each declared key is checked against its own schema.
- `additionalProperties`: `false` refuses each undeclared key. A schema there
  checks each undeclared key against it.
- `enum`: the value must equal one of the options. `true` does not equal `1`.
- `items`: one schema, checked against each element. The list form is ignored.

Any other keyword (`anyOf`, `oneOf`, `$ref`, `pattern`, `minimum`, `minItems`,
`format` and the rest) is ignored. A schema that uses one is checked no more
strictly than before, so tools whose schema is not strict JSON Schema keep
working. An unknown `type` name is ignored too.

A wrong type is reported once, and the value under it is not checked further.
A value outside an `enum` is reported the same way.

Paths are written as `limit`, `ranges[1].start` or `options.mode`.

## What is not checked

- A tool with no schema. An empty `parameters` mapping has no keywords, so
  nothing is checked. `{"type": "object"}` only checks that the arguments are
  an object, which they always are.
- A call whose arguments did not parse as JSON. A provider passes those on as
  `{"_raw_arguments": text}`, and the tool answers them itself
  (`tau_coding/tools.py` reports the cut-off). Checked against the schema,
  every required key would look missing, so the loop skips them.
- Arguments that an extension's `tool_call` hook returns. Extension hooks run
  inside the tool's executor (`tau_coding/extensions/runtime.py`), after the
  loop's check.

## Order

1. `before_tool_call` may block the call.
2. An unknown tool name is refused.
3. The arguments are checked against the schema. This is new.
4. The tool runs. Extension `tool_call` hooks run here.

## Required means the tool cannot do without it

Before this check, a schema's `required` list only advised the model. Now a
call that lacks a required key is refused and the tool never runs. A key the
tool can do without, or that it handles itself when it is missing, should
therefore not be listed as required.

Bash's `description` was such a key: it only labels the call in the UI, and
the tool already ran without it. Some models (Gemma 4 among them) leave it out,
so it is now optional rather than costing those models a refused turn.

## Why

Before this, Tau passed `parameters` to the provider and never looked at it
again. Each tool parsed its own arguments by hand, and stopped at the first
problem. A small model then learned one problem per call and spent a turn on
each. A single error that names every problem lets it fix the call in one
retry. The check is in `tau_agent`, so it holds for every tool and every
frontend, and it has no CLI or UI parts.

No dependency was added: the subset above is about 130 lines, where
`jsonschema` would be a new dependency of the core package.

## How it maps to Pi

As far as we know, Pi's agent loop also validates a call's arguments against the tool's schema
before the tool runs. Tau's version reports every problem in one message and
does not coerce. The loop/harness split is unchanged: the check is in the
harness, and `tau_coding` gets it for free.

## Tests

`tests/test_tool_arguments.py` uses a fake provider and a fake `read` tool:

- a missing required key is refused, and the tool does not run
- one error lists five problems: missing, wrong type, enum, unknown key, and a
  nested missing key
- a wrong type (a boolean for an integer, a string for an array) and a value
  outside an enum are each named
- valid arguments reach the tool unchanged (`10.0` stays a float)
- a tool with an empty schema is not checked
- `_raw_arguments` is left to the tool

Run with `uv run pytest tests/test_tool_arguments.py tests/test_agent_loop.py`.
