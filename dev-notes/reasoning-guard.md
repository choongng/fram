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
- **Replay of a reasoning-only turn.** A budget cut usually leaves a turn that
  holds reasoning and nothing else. Once another message follows such a turn,
  `provider_context` in `tau_agent` sends it with one line of text after the
  reasoning (`BUDGET_CUT_TURN_TEXT`, or `REASONING_ONLY_TURN_TEXT` for a turn
  that ended that way for another reason, a failed or aborted one included).
  While the turn is the last message it is left out of the request. The session
  file keeps the turn as it was streamed.
- **Switch.** `/reasoning-guard`, RPC `set_reasoning_guard`, `--reasoning-guard`
  and `TAU_REASONING_GUARD`. The same switch governs the tool-argument loop cut
  in `tau_ai`, which ends a chat-completions reply whose tool-call arguments
  repeat one block of text. Only the chat-completions parser has that cut; the
  other providers have nothing for the switch to turn off.
- **A follow-up begun past its thinking.** Where the provider continues a final
  assistant message (`supportsAssistantPrefill`), the follow-up after a budget
  cut is sent with its reply begun, as an assistant message of one newline
  (`CLOSED_THINKING_PREFILL`), so its reply begins after a closed thinking
  block. Only while the guard is on, and only for chat-completions requests.

## Why it exists

A model can spend its whole output limit reasoning and never act. The stream
looks healthy, so stall detection doesn't help, and the user pays for the time.

## Design

- Budgets are counted in characters or tokens, not time: a time budget would
  give slow models less thinking than fast ones.
- The guard reads the setting on every request. The harness takes its provider
  once per run, so adding or removing the wrapper would only take effect on the
  next run.
- With the guard off, the provider's events are passed on as they come and the
  model's limits are not read.
- `tau_ai` does not read Tau's settings, and the choice is not kept on the
  provider: sessions can share one provider object and choose differently, and a
  request can wait for a slot between being made and being sent.
  `tau_ai.loop_cuts` holds the choice in a context variable, true by default.
  The guard sets it around each step of reading a response and resets it before
  handing the event on, so it is in force when the chat-completions parser
  starts on the response and does not reach the caller's other requests. A
  change of the switch reaches the next request; a response already being read
  keeps the choice its parser started with. The continuation-summary request
  carries the run's tools but goes to the provider without the guard, so the
  session marks it with the switch itself. Both cuts are heuristics with the
  same remedy, so one switch covers them: a user who turns the guard off to let
  a long reply through should not have to find a second control. Repeated tool
  calls stay refused either way: a refusal is an ordinary tool result and loses
  no output.
- The guard never edits the prompt prefix or the thinking level, so cached
  prompts stay valid. The only changes to later requests are the looped passage
  left out of a cut turn and the line of text on a reasoning-only one.
- A reasoning-only turn was not a well-formed turn in any provider's request.
  Chat completions (every thinking format, `qwen-chat-template` included) and
  Mistral sent `content: ""` beside `reasoning_content`, which some servers
  reject and some chat templates drop. The Responses and Codex builders sent
  nothing for the turn, and Anthropic's dropped it once its thinking was pruned,
  each leaving two user turns in a row. Google sent a turn of thought parts
  only. The text is added where the context is built rather than in each
  builder, so all of them get it, and it depends on the stored turn alone, so
  the turn has the same bytes in every later request.
- "No reply" means no tool call and no text beyond whitespace, which servers
  treat as no text. Blank text blocks are replaced by the line, not kept beside
  it. A turn that ended normally with no reasoning either gets
  `EMPTY_TURN_TEXT`, since it did not end "after its reasoning"; a failed,
  aborted or cut-off turn with no reasoning is left out, as before.
- A reasoning-only turn that is the last message is left out of the request. A
  failed turn keeps its partial reasoning, and `continue` or a retry sends the
  context with that turn last; with the line on it, the request would end on an
  assistant turn, which a server reads as the start of the reply to continue or
  refuses. Everything before the turn is sent the same either way, and the turn
  appears, already in its final form, in the first request that has a message
  after it, so the cached prefix is never rewritten. The continuation-summary
  request builds its context with its own message in place for the same reason.
  Sessions written before this change replay the same way; their first request
  afterwards misses the prompt cache from that turn on, once.
- Budgets have no floor. A replay of 28,477 stored reasoning turns found 47 over
  32,768 characters that ended normally, so the `medium` level budget cuts about
  one sound turn in 600, and lower budgets cut more. A floor on the level
  budgets would have to sit near that size to save those turns, which would make
  `minimal`, `low` and `medium` the same budget on the formats where the budget
  is the only thing a level changes. A floor on a budget the user set would
  override what they asked for. A wrong budget cut is also cheap: the reasoning
  is kept and the follow-up asks the model to act on it. Users who need longer
  thinking can raise the level (`xhigh` and `max` have no budget), set a larger
  budget, or switch the guard off.
- A model told in words to stop thinking does not always stop. One model, cut at
  its budget and nudged, thought each of four follow-ups to its budget again, on
  the same hand trace, and wrote nothing. Sending the follow-up with thinking
  off would stop it, but the Qwen and Gemma chat templates mark thinking in the
  system turn: with `enable_thinking=false`, Gemma's drops `<|think|>`, and
  Qwen's tool instructions change. The whole rendered prompt changes, and the
  server reads all of it again.
- The prefill keeps the prefix instead. llama.cpp continues a final assistant
  message as the reply, and these templates render a turn with text and no
  reasoning with its thinking block already closed: Qwen's
  `<think>\n\n</think>`, Gemma's `<|channel>thought\n<channel|>`. The newline is
  added to that one request, after `provider_context` has built it, and is never
  stored. Measured on llama.cpp from a warm cache:

  | Model | Follow-up, prefill | Follow-up, thinking off |
  |---|---|---|
  | Gemma 4 26B | 4,033 cached + 38 new, a `write` call | 0 cached + 4,068 new |
  | Qwen 3.6 35B | 4,525 cached + 62 new | 0 cached + 4,527 new |

  A second Qwen model matched the Qwen row. Each prefilled follow-up wrote a
  tool call with no reasoning, and the request after it hit the cache too.
- `supportsAssistantPrefill` is off by default. vLLM continues a final assistant
  message only when asked, most hosted APIs answer after it or refuse it, and
  Anthropic refuses one while thinking is on. A template that does not close the
  thinking block for such a turn gains nothing from it, so it is set per model.
  The measurements are one run per model, on one llama.cpp build; a build that
  refuses a final assistant message while thinking is on would fail the
  follow-up. Models added by the built-in llama.cpp extension carry no compat
  values, so they cannot turn it on.
- Thresholds are heuristics. A loop cut rarely loses anything, since a reply the
  detector catches would usually have run to the output limit anyway, but a
  budget can cut useful long reasoning, which is why the switch exists.

## How it maps to Pi

It sits in the coding-agent layer as a provider wrapper. `tau_agent` only gains
the diagnostics that mark a cut, and leaves looped reasoning out of the context
it sends.

## How to test

`uv run pytest tests/test_reasoning_guard_switch.py tests/test_reasoning_tokens.py tests/test_reasoning_char_budgets.py tests/test_session_output_limit.py tests/test_session_output_limit_nudge.py tests/test_budget_cut_prefill.py`
