# Clearer answers to malformed and cut-off tool calls

## What was added

- A reply that ends at the output limit keeps `length` as its stop reason even
  when it holds a tool call. The agent loop answers that last call with a note
  and does not run it. Complete calls before it still run.
- A call whose arguments are not valid JSON now gets an error that says which
  arguments arrived, which string never closed and how it ends, and which
  required arguments never came. A call whose strings all closed but hold a bad
  escape is told how to escape them.
- An `edit` whose edit objects all name the same `path` is repaired. Any other
  misplaced `path` is refused with the correct form spelled out.

## Why it exists

Arguments cut off by the limit can still parse, so running them can write a
partial file. And a vague error ("path must be a string") reads as a type
problem, so a model tends to resend the same call.

## How it maps to Pi

The loop stays provider-neutral: providers report `length`, and the loop decides
what a cut call means. Tool-specific wording stays in `tau_coding.tools`.

## How to test

`uv run pytest tests/test_cut_tool_call.py tests/test_edit_path_forms.py tests/test_tool_cut_off_arguments.py`
