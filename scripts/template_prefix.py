"""Render Fram's requests through llama.cpp's chat template and compare prefixes.

A llama.cpp server reuses the cached prompt of a slot up to the first token
that differs, so a request whose rendered prompt does not start with the whole
of the last one reads the rest again. What Fram sends as JSON can stay a clean
extension while the rendered prompt does not: a chat template may mark thinking
in the system turn, or render an earlier turn differently once a later one
follows. This script asks the server itself, through `/apply-template`, which
renders a request without running it.

With no payload files, it builds a short coding session the way Fram does
(`_build_chat_payload`, with the `qwen-chat-template` thinking format) and
checks each step of it:

- a tool turn, then the next request;
- a cut at the thinking budget, then its follow-up begun with
  `CLOSED_THINKING_PREFILL`, then the request after the reply;
- the same follow-up with thinking off, which is what the prefill replaces.

Each check asks that the later request keep the conversation the earlier one
had. The assistant turn the earlier request leaves open for its reply is left
out of that, and how much of it was rendered otherwise is reported.

With two payload files (request bodies as JSON), it compares those two.

    uv run python scripts/template_prefix.py --model qwen3-30b-a3b
    uv run python scripts/template_prefix.py --model M before.json after.json
    uv run python scripts/template_prefix.py --url http://localhost:8081 ...

Behind llama-swap (the default, `http://localhost:8080`), the model's own server
is reached at `/upstream/MODEL`; `--url` names a llama-server directly.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

from tau_agent import AssistantMessage, UserMessage
from tau_agent.messages import TextContent, ThinkingContent, ToolCall, ToolResultMessage
from tau_agent.tools import AgentTool
from tau_ai.openai_compatible import _build_chat_payload
from tau_coding.guards import BUDGET_CUT_NUDGE
from tau_coding.reasoning_guard import CLOSED_THINKING_PREFILL

#: Fields of a chat request that `/apply-template` has no use for.
_NOT_RENDERED = ("model", "stream", "stream_options", "store", "cache_prompt", "id_slot")


def render(endpoint: str, payload: dict[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k not in _NOT_RENDERED}
    request = urllib.request.Request(
        f"{endpoint}/apply-template",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return str(json.load(response)["prompt"])


def settled(endpoint: str, payload: dict[str, Any], prompt: str) -> int:
    """How much of a request's rendered `prompt` is conversation already had.

    The rest is the assistant turn the request leaves open for the reply (with
    its prefill, if any). Once the reply is history the template may render that
    turn otherwise, gemma dropping an empty thought channel for one, so the
    server reads it again with the reply; that is a few tokens, not a miss.
    """
    messages = payload["messages"]
    if messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    closed = render(endpoint, {**payload, "messages": messages, "add_generation_prompt": False})
    return shared(closed, prompt)


def shared(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def compare(label: str, before: str, after: str, *, keep: int) -> bool:
    """Whether `after` starts with the first `keep` characters of `before`,
    printing where they part if not."""
    n = shared(before, after)
    ok = n >= keep
    open_turn = f", {len(before) - n} of the open turn read again" if ok and n < len(before) else ""
    print(
        f"{'ok  ' if ok else 'MISS'} {label}: {min(n, keep)} of {keep} characters kept{open_turn}"
    )
    if not ok:
        print(f"       before: {before[max(0, n - 60) : n + 60]!r}")
        print(f"       after:  {after[max(0, n - 60) : n + 60]!r}")
    return ok


def _tool() -> AgentTool:
    async def execute(*_args: object) -> object:
        raise NotImplementedError

    return AgentTool(
        name="write",
        label="write",
        description="Write a file.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        execute_fn=execute,  # type: ignore[arg-type]
    )


def _call(n: int, path: str) -> ToolCall:
    return ToolCall(id=f"call-{n}", name="write", arguments={"path": path, "content": "..."})


def _result(n: int, path: str) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=f"call-{n}", tool_name="write", content=[TextContent(text=f"wrote {path}")]
    )


def session_payloads(thinking: str) -> dict[str, dict[str, Any]]:
    """The requests of a short session with a budget cut, as Fram builds them."""
    tools = [_tool()]
    system = "You are a coding agent. Write the code the user asks for."

    def payload(messages: list[Any], effort: str = thinking) -> dict[str, Any]:
        return _build_chat_payload(
            model="m",
            system=system,
            messages=messages,
            tools=tools,
            reasoning_effort=effort,
            thinking_format="qwen-chat-template",
            include_reasoning_effort_none=True,
        )

    first = [UserMessage(content="Write rle.py, run-length encode and decode.")]
    turn = [
        AssistantMessage(
            content=[ThinkingContent(thinking="Notes first."), _call(1, "notes.txt")],
            stop_reason="toolUse",
        ),
        _result(1, "notes.txt"),
    ]
    cut = AssistantMessage(
        content=[ThinkingContent(thinking="Tracing decode by hand: count byte 0x03 ...")],
        stop_reason="length",
    )
    follow_up = [*first, *turn, cut, UserMessage(content=BUDGET_CUT_NUDGE)]
    reply = AssistantMessage(content=[_call(2, "rle.py")], stop_reason="toolUse")
    prefill = AssistantMessage(content=CLOSED_THINKING_PREFILL)
    return {
        "first": payload(first),
        "cut": payload([*first, *turn]),
        "follow-up, prefilled": payload([*follow_up, prefill]),
        "follow-up, thinking off": payload(follow_up, effort="none"),
        "after the reply": payload([*follow_up, reply, _result(2, "rle.py")]),
    }


def check_session(endpoint: str, thinking: str) -> bool:
    payloads = session_payloads(thinking)
    rendered = {name: render(endpoint, p) for name, p in payloads.items()}

    def step(label: str, before: str, after: str) -> bool:
        keep = settled(endpoint, payloads[before], rendered[before])
        return compare(label, rendered[before], rendered[after], keep=keep)

    results = [
        step("a tool turn, then the next request", "first", "cut"),
        step("the cut request, then its prefilled follow-up", "cut", "follow-up, prefilled"),
        step(
            "the prefilled follow-up, then the request after its reply",
            "follow-up, prefilled",
            "after the reply",
        ),
        step(
            "the cut request, then its follow-up with thinking off (what the prefill replaces)",
            "cut",
            "follow-up, thinking off",
        ),
    ]
    return all(results[:3])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("payloads", nargs="*", type=Path, help="two request bodies to compare")
    parser.add_argument("--model", required=True, help="the model llama-swap routes to")
    parser.add_argument("--url", help="a llama-server's own base URL, not behind llama-swap")
    parser.add_argument("--swap", default="http://localhost:8080", help="llama-swap's base URL")
    parser.add_argument("--thinking", default="medium", help="the reasoning effort Fram sends")
    args = parser.parse_args(argv)
    endpoint = args.url or f"{args.swap}/upstream/{args.model}"
    print(f"== {args.model} via {endpoint}")
    if args.payloads:
        if len(args.payloads) != 2:
            parser.error("give two payload files, or none")
        before, after = (json.loads(path.read_text()) for path in args.payloads)
        prompt = render(endpoint, before)
        keep = settled(endpoint, before, prompt)
        ok = compare("payload 1, then payload 2", prompt, render(endpoint, after), keep=keep)
    else:
        ok = check_session(endpoint, args.thinking)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
