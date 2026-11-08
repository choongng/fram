"""httpx transport doubles shared by provider tests.

Held out of the ``test_`` modules so those contain only tests and helpers:
these are stand-ins for the Anthropic API itself, not test scaffolding.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import httpx

OK_SSE = (
    'data: {"type":"message_start","message":{"usage":{}}}\n\n'
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"text_delta","text":"ok"}}\n\n'
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    '"usage":{"output_tokens":1}}\n\n'
    'data: {"type":"message_stop"}\n\n'
)

SIGNATURE_400 = json.dumps(
    {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "Invalid signature in thinking block.",
        },
    }
)


def canonical(messages: list[dict[str, Any]]) -> str:
    return json.dumps(messages, sort_keys=True)


def blocks(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    content = message["content"]
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, Mapping)]


class SignatureBindingTransport(httpx.AsyncBaseTransport):
    """Mock transport enforcing Anthropic's signature-to-prefix binding.

    A thinking block is accepted only when the payload prefix before it is
    byte-identical to the prefix the signature was issued under; anything else
    answers with a 400 naming the signature. With every past thinking block
    replayed, a compacted or injected prefix would make every later request in
    the conversation fail.
    """

    def __init__(self, *, bound_prefixes: Mapping[str, str]) -> None:
        self._bound_prefixes = dict(bound_prefixes)
        self.captured: list[dict[str, Any]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        payload = json.loads(body.decode())
        self.captured.append(payload)
        messages = payload["messages"]
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                continue
            prefix = canonical(list(messages[:index]))
            for block in blocks(message):
                if block.get("type") not in {"thinking", "redacted_thinking"}:
                    continue
                signature = block.get("signature")
                if signature is None:
                    continue
                known = self._bound_prefixes.setdefault(str(signature), prefix)
                if known != prefix:
                    return httpx.Response(
                        400,
                        text=SIGNATURE_400,
                        headers={"content-type": "application/json"},
                    )
        return httpx.Response(
            200,
            text=OK_SSE,
            headers={"content-type": "text/event-stream"},
        )
