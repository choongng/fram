# Model profiles

## What was added

Tau now keeps a measured profile for each served model: how fast it decodes, how
fast it reads an uncached prompt, how many characters it writes per token, and
what limits it was seen to honour. `tau probe` warm-starts a profile with three
short capped requests. `tau profile show [--json]` prints the stored profiles.

## Why it exists

Context estimates assumed four characters per token, but models differ, and
thinking text often runs well below four. Speed was not recorded anywhere, so a
caller planning around a server had nothing to read.

## How it maps to Pi's layers

| Layer | Piece | Job |
|---|---|---|
| `tau_ai` | `model_profile.py` | Pure: turn a response into samples, fold samples into a profile |
| `tau_agent` | `loop.timed_response` | One tool-less request, timed exactly as the loop times a turn |
| `tau_coding` | `model_profiles.py` | The JSON file under the home directory, and read helpers |
| `tau_coding` | `model_probe.py` | The probe requests |
| `tau_coding` | `session.py` | Records each response; reads the ratio for estimates |

`model_profile.py` imports nothing from `tau_coding` and does no I/O; a test
enforces that.

## What is measured

A profile is keyed by provider, endpoint (the base URL) and model, because speed
belongs to the server as much as to the model.

A response is skipped when it reports no output tokens, ended in an error or
abort, or carries a `usage_estimated` diagnostic (for a stream Tau cuts itself).
A response with both thinking and visible output, from a server that doesn't
report reasoning tokens, can't be split, so it gives only the combined ratio.

## The moving average

Each value keeps its sample count and `weight`, the tokens of evidence behind
it. A sample backed by `tokens` tokens moves the value by
`max(1 - 0.9 ** (tokens / 1000), tokens / (weight + tokens))`.

- Until about 10,000 tokens of evidence the second term wins and the value is
  the token-weighted mean, so a tiny first reply can't fix it.
- After that it is a moving average: a 1,000-token reply moves it 10%.
- From 500 tokens of evidence, a sample more than 3x off is clipped to 3x, so
  one stall can't swing the value.
- Speed and ratio samples need 32 tokens. Prefill samples need 128 uncached
  tokens and a prompt that wasn't mostly cached.

## Where the ratio is used

The session's fallback context estimates use the combined ratio once it rests on
`MIN_TRUSTED_RATIO_WEIGHT` (4,000) tokens. Provider-reported usage still wins
wherever it exists. That threshold is a heuristic.

Nothing else reads a profile, and a profile never changes a request.

## Failure behaviour

A profile is an estimate kept on the side. A failed read or write never reaches
the turn. Two processes writing at once can lose one sample (the later write
wins), which is acceptable for an estimate.

A failed probe request stores nothing: `tau probe` prints the error and exits 1.

## How to test or use it

```bash
uv run pytest tests/test_model_profiles.py
tau probe --provider local --model my-model
tau profile show --json
```

User-facing details are under "Model profiles" in
`website/content/reference/cli.md`.
