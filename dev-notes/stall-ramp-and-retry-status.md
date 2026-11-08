# Stall timer ramp and visible retries

A request sent after a long idle can reach a model server that is still loading
its weights. The server sends nothing, neither an error nor a first token, for
as long as the load takes. With a fixed stall window every attempt fails the
same way, and the session sits silent with no explanation. Two changes address
this: the stall timer ramps per attempt, and each retry is shown to the user.

## Stall timer ramp (`src/tau_ai/stall.py`)

The no-event wait doubles per attempt: 60, 120, 240, 480, then 600 s (the cap).
The first attempt still ends a stuck server quickly; the last retries outlast a
long cold load. `DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES` is 4, so a request gets
five attempts, enough to reach the cap.

`stream_stall_seconds(attempt)` computes the wait, and the retry loop in
`openai_compatible.py` reads it per attempt. `TAU_STREAM_STALL_SECONDS` replaces
the ramp with one fixed value for every attempt; `0` or `off` turns the timer
off.

## Retry status (`tau_agent` events)

`canonicalize_provider_stream` converts each `ProviderRetryEvent` a provider
adapter emits (`tau_ai`) to an `AssistantRetryEvent`
(`tau_agent.provider_events`), and the agent loop (`tau_agent/loop.py`) emits
that as the `ProviderRetryEvent` agent event in `tau_agent.events`. The event is
transient: it carries no assistant content and is not persisted. Consumers:

- The TUI adapter (`tau_coding/tui/adapter.py`) shows `… {message}` as a status
  line, like the auto-retry and compaction statuses.
- Print mode (`tau_coding/rendering/plain.py`) writes `… {message}` to stderr.
- JSON mode dumps it like every other event. Extensions can subscribe to the
  `provider_retry` event type (`tau_coding/extensions/api.py`).

Pi treats transport retries as internal to the provider; Tau forwards them as a
status event so a frontend can say why the agent is quiet.

## Testing

`tests/test_stream_stall.py` covers the ramp, the override and the attempt
count; `tests/test_provider_retry_surface.py` covers the conversion, the loop,
the renderer and the adapter. The exact-sequence retry tests in
`tests/test_tau_ai.py` include the `assistant_retry` entries.

To see it, point a provider at a model server that is still loading weights.
Each attempt reports
`… Retrying provider request 2/5 after the server sent nothing for 60 s in 0.25s.`
while it waits.
