"""Detect prompt-cache misses on providers that normally report cache hits."""

from __future__ import annotations

from tau_agent.messages import AssistantMessage
from tau_coding.events import PromptCacheMissEvent

PROMPT_CACHE_MISS_MIN_PROMPT_TOKENS = 8_192
"""Smallest prompt worth flagging when none of it was served from cache.

Below this a miss costs little: short prompts reprocess quickly, and a first
turn or a tiny tool follow-up is routinely uncached.
"""


class PromptCacheMissDetector:
    """Flag a large, fully uncached prompt once a provider has shown it caches.

    A provider that has never reported ``cache_read`` may not cache at all or
    may not report it, so only a (provider, model) pair that has reported a hit
    earlier in the session is judged. A miss then means the whole prompt was
    reprocessed: on a hosted API it costs full input price; on a single-slot
    local server it usually means another request evicted this session's
    cached prefix, and the turn takes as long as a cold start.
    """

    def __init__(self, *, min_prompt_tokens: int = PROMPT_CACHE_MISS_MIN_PROMPT_TOKENS) -> None:
        self._min_prompt_tokens = min_prompt_tokens
        self._caching: set[tuple[str, str]] = set()
        self._skip_next = False

    def expect_miss(self) -> None:
        """Do not flag the next response, whose prefix was knowingly displaced."""
        self._skip_next = True

    def observe(self, message: AssistantMessage) -> PromptCacheMissEvent | None:
        """Record one completed response and return a miss event when it is one."""
        if message.stop_reason in {"error", "aborted"}:
            return None
        skip, self._skip_next = self._skip_next, False
        key = (message.provider, message.model)
        usage = message.usage
        if usage.cache_read > 0:
            self._caching.add(key)
            return None
        prompt_tokens = usage.input + usage.cache_write
        if skip or key not in self._caching or prompt_tokens < self._min_prompt_tokens:
            return None
        return PromptCacheMissEvent(
            provider=message.provider,
            model=message.model,
            prompt_tokens=prompt_tokens,
        )
