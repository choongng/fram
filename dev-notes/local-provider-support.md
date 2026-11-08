# Local provider support

Three opt-in features make a `llama-server` behind an OpenAI-compatible provider
keep its prompt cache and avoid overlapping requests in long conversations.
User-facing setup is in `website/content/guides/providers-and-models.md`.

## What was added

- **`cache_prompt` and `id_slot_count`.** Provider preferences that add
  llama.cpp's request fields. `id_slot` is hashed from the session id, so a
  session asks for the same slot on every turn and after a resume.
- **Prompt-cache-miss reporting.** `PromptCacheMissDetector` watches the
  `cache_read` count on completed responses. Once a provider and model have
  shown a hit, a large prompt with no cache read emits `prompt_cache_miss`.
- **`max_concurrent_requests`.** `ConcurrencyLimitedProvider` wraps a provider
  so that everything in one process shares a queue per provider name.

## Why

A single-slot server keeps one cached prefix. When a session and its subagent
take turns on that slot, each evicts the other's prefix and every turn runs as a
cold start. The queue stops requests overlapping, slot pinning keeps sessions
apart on multi-slot servers, and the miss report makes an eviction visible.

## How it maps to Pi's design

The request fields live in the provider layer (`tau_ai`). The queue and the
detector live in `tau_coding`, because they depend on which sessions exist in
the process. The queue is a provider wrapper, so `tau_agent` does not know about
it, and the detector consumes the harness's existing message events.

## Behavior notes

- Observers of queued requests are process-wide and filter by session id. A
  discarded session must stop observing, which `_adopt_replacement` does for the
  replacement it absorbs.
- The detector is replaced on a session switch, and it skips the response after
  a compaction, since the summary request displaces the cached prefix.
- A gate is replaced when a request arrives with a different limit. Requests
  already holding the old gate finish under it.

## How to test

`uv run pytest tests/test_request_gate.py tests/test_prompt_cache.py`, and the
`cache_prompt` cases in `tests/test_tau_ai.py`.
