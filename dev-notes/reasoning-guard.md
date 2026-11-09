# Reasoning guard

## What was added

`tau_coding/reasoning_guard.py` wraps the provider for each run. It reads the
streamed reasoning and ends the reply when the reasoning loops or passes a
budget. `guards.py` then queues a follow-up asking the model to act.

- **Loop cut.** `ReasoningLoopDetector` looks for repeated lines and sentences.
  The looped passage is dropped from later requests, so the model doesn't
  continue it.
- **Budget cut.** A per-turn budget in characters. The reasoning is kept, so the
  next reply can build on it. A token budget converts to characters at the
  model's measured thinking ratio.
- **Switch.** `/reasoning-guard`, RPC `set_reasoning_guard`, `--reasoning-guard`
  and `TAU_REASONING_GUARD`.

## Why it exists

A model can spend its whole output limit reasoning and never act. The stream
looks healthy, so stall detection doesn't help, and the user pays for the time.

## Design

- Budgets are counted in characters or tokens, not time: a time budget would
  give slow models less thinking than fast ones.
- The guard reads the setting on every request. The harness takes its provider
  once per run, so adding or removing the wrapper would only take effect on the
  next run.
- With the guard off, the provider's own stream is returned untouched.
- The guard never edits the prompt prefix or the thinking level, so cached
  prompts stay valid. The only change to later requests is the looped passage
  left out of the cut turn.
- Thresholds are heuristics. A loop cut rarely loses anything, since a reply the
  detector catches would usually have run to the output limit anyway, but a
  budget can cut useful long reasoning, which is why the switch exists.

## How it maps to Pi

It sits in the coding-agent layer as a provider wrapper. `tau_agent` only gains
the diagnostics that mark a cut, and leaves looped reasoning out of the context
it sends.

## How to test

`uv run pytest tests/test_reasoning_guard_switch.py tests/test_reasoning_tokens.py tests/test_reasoning_char_budgets.py tests/test_session_output_limit.py tests/test_session_output_limit_nudge.py`
