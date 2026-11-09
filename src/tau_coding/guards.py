"""Guards on a coding session's tool calls.

A guard refuses a call that would make no progress and says what to do
instead. A refusal is an ordinary tool result; the prompt and the thinking
level are unchanged, so a cached prompt prefix stays valid.

The numbers here are heuristics. Each one says what it is meant to allow.
"""

import json
import re
import shutil
from collections import deque
from collections.abc import Callable, Hashable, Sequence

from tau_agent.messages import TextContent, ToolCall
from tau_agent.tools import AgentTool, AgentToolResult
from tau_coding.tools import NO_OUTPUT

#: Runs of one tool call, with the same result each time, after which the
#: next run of it is refused.
REPEATED_CALL_LIMIT = 2
#: The runs are counted over this many of the latest calls, so two calls
#: taken in turn repeat as surely as one call taken twice.
REPEATED_CALL_WINDOW = 10
#: Refusals in a row after which the run is stopped.
REPEATED_CALL_STOP = 8
#: Characters of a repeated call's earlier result its refusal quotes.
REPEATED_CALL_QUOTE = 300
#: What a repeat refusal tells the model to do after quoting the result.
REPEATED_CALL_STEP = "Do what that result asks, or take a different step."
#: The same, after a result that was empty and so asks nothing.
REPEATED_EMPTY_STEP = (
    "It gave no output each time, so it cannot show you what you are looking for. "
    "Take a different step, one whose output shows what you need to see."
)
#: The same, after a change that was already made.
REPEATED_CHANGE_STEP = (
    "That change is already made, so making it again changes nothing. Read or run "
    "what it changed to see whether it does what you meant, or take a different step."
)
#: How the note `buffer_tool_result` puts after a cut read begins; it names
#: the offset to continue from.
READ_CUT_NOTE = "[Read cut:"

#: Words that run in a shell without being a program on PATH.
_SHELL_BUILTINS = frozenset(
    [
        "alias",
        "bg",
        "bind",
        "break",
        "builtin",
        "caller",
        "cd",
        "command",
        "compgen",
        "complete",
        "compopt",
        "continue",
        "declare",
        "dirs",
        "disown",
        "echo",
        "enable",
        "eval",
        "exec",
        "exit",
        "export",
        "false",
        "fc",
        "fg",
        "getopts",
        "hash",
        "help",
        "history",
        "jobs",
        "kill",
        "let",
        "local",
        "logout",
        "mapfile",
        "popd",
        "printf",
        "pushd",
        "pwd",
        "read",
        "readarray",
        "readonly",
        "return",
        "set",
        "shift",
        "shopt",
        "source",
        "suspend",
        "test",
        "times",
        "trap",
        "true",
        "type",
        "typeset",
        "ulimit",
        "umask",
        "unalias",
        "unset",
        "wait",
    ]
)

_HEREDOC = re.compile(r"(?<!<)<<(-?)\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
_QUOTED = re.compile(r"'[^']*'|\"(?:\\.|[^\"\\])*\"", flags=re.DOTALL)


class ToolCallGuards:
    """Refuses tool calls that cannot help, for the length of one run.

    `problem` says what is wrong with a call, and `verdict` turns that into the
    answer for `before_tool_call`. `record` remembers what a call that ran gave.
    """

    def __init__(
        self, *, tools: Callable[[], Sequence[AgentTool]], stop_run: Callable[[], None]
    ) -> None:
        self._tools = tools
        self._stop_run = stop_run
        #: The latest tool calls that ran, each with a digest of its result.
        self._recent_calls: deque[tuple[str, int]] = deque(maxlen=REPEATED_CALL_WINDOW)
        #: The quoted line of each call's latest result, and what its repeat
        #: refusal tells the model to do.
        self._recent_heads: dict[str, tuple[str, str]] = {}
        #: The calls whose repeat refusal has already quoted their result.
        self._repeat_quoted: set[str] = set()
        #: The latest result digest of each `bash` command, kept across clears.
        self._bash_digests: dict[str, int] = {}
        self._refused_in_a_row = 0
        #: The calls this guard let through, so the only ones that can have run.
        self._cleared_call_ids: set[str] = set()

    def reset(self) -> None:
        """Forget everything; each run starts clean."""
        self._recent_calls.clear()
        self._recent_heads.clear()
        self._repeat_quoted.clear()
        self._bash_digests.clear()
        self._refused_in_a_row = 0
        self._cleared_call_ids.clear()

    def problem(self, call: ToolCall) -> str | None:
        """Why `call` is refused, or None to let it run."""
        return self._tool_run_as_command(call) or self._repeated_call(call)

    def verdict(self, call: ToolCall, problem: str | None) -> tuple[bool, str | None]:
        """Answer `call` given its `problem`, and stop a run that keeps at it.

        Each refusal says what to do instead. After `REPEATED_CALL_STOP`
        refusals in a row the run is stopped: a model that ignores the note
        that many times will not take it.
        """
        if problem is None:
            self._refused_in_a_row = 0
            self._cleared_call_ids.add(call.id)
            return False, None
        self._refused_in_a_row += 1
        if self._refused_in_a_row >= REPEATED_CALL_STOP:
            self._stop_run()
            return True, (
                f"Not run, and the run is stopped: {self._refused_in_a_row} tool calls in "
                "a row were refused."
            )
        return True, (
            f"Not run: {problem} {REPEATED_CALL_STOP - self._refused_in_a_row} more refused "
            "calls in a row stop the run."
        )

    def take_cleared(self, call: ToolCall) -> bool:
        """Whether this guard let `call` through, which also clears the mark.

        A call it refused did not run. Nor did one the agent loop answered
        without asking it, such as a call cut off at the output limit.
        """
        cleared = call.id in self._cleared_call_ids
        self._cleared_call_ids.discard(call.id)
        return cleared

    def record(
        self, call: ToolCall, result: AgentToolResult, shown: AgentToolResult, is_error: bool
    ) -> None:
        """Remember what `call` gave: `result` as run, `shown` as the model sees it."""
        # The text alone: details such as a duration differ between runs whose
        # output, which is all the model sees, is the same.
        text = _text(result)
        digest = hash((is_error, text))
        changing = {tool.name for tool in self._tools() if tool.changes_workspace}
        key = call_key(call)
        # A shell command may change anything, and nothing says whether it did.
        # One whose result differs from its last run is taken as a change; one
        # that gives the same as before is not, so a pair of commands taken in
        # turn with unchanging results is still caught.
        bash_changed = call.name == "bash" and self._bash_digests.get(key) != digest
        if call.name == "bash":
            self._bash_digests[key] = digest
        if call.name in changing or bash_changed:
            # A change can alter what any read gives, so earlier results no
            # longer count. The changing calls themselves are kept, so one
            # repeated with the same result each time is still caught.
            kept = [entry for entry in self._recent_calls if entry[0].split("\0", 1)[0] in changing]
            self._recent_calls.clear()
            self._recent_calls.extend(kept)
        self._recent_calls.append((key, digest))
        self._repeat_quoted.discard(key)  # it ran again: its next repeat quotes anew
        if text.strip() in ("", NO_OUTPUT):
            step = REPEATED_EMPTY_STEP
        elif call.name in changing and not is_error:
            step = REPEATED_CHANGE_STEP
        else:
            step = REPEATED_CALL_STEP
        self._recent_heads[key] = (self._quote(text, shown, result), step)
        live = {seen for seen, _ in self._recent_calls}
        for stale in [seen for seen in self._recent_heads if seen not in live]:
            del self._recent_heads[stale]
        self._repeat_quoted &= live

    def _quote(self, text: str, shown: AgentToolResult, result: AgentToolResult) -> str:
        """The part of a result worth repeating in its refusal."""
        first = next((line.strip() for line in text.splitlines() if line.strip()), "")
        head = first[:REPEATED_CALL_QUOTE]
        if len(first) > REPEATED_CALL_QUOTE:
            # A quote cut before the call a result asks for can read as done.
            tools = {tool.name for tool in self._tools()}
            if _named_call(head, tools) is None:
                head = _call_sentence(text, tools) or head
        if shown is not result:
            # A re-read of a cut file needs the offset to go on from, which the
            # cut note names and the file's first line does not.
            note = next(
                (
                    line.strip()
                    for line in reversed(_text(shown).splitlines())
                    if line.strip().startswith(READ_CUT_NOTE)
                ),
                "",
            )
            head = note[:REPEATED_CALL_QUOTE] or head
        return head

    def _repeated_call(self, call: ToolCall) -> str | None:
        """Why `call` repeats one that keeps giving the same result, or None.

        A model can lock onto one call, or a pair it takes in turn, and repeat
        it for as long as the run lasts. A call that ran `REPEATED_CALL_LIMIT`
        times among the latest `REPEATED_CALL_WINDOW`, with the same result each
        time, is a repeat. A call whose result changes is not.
        """
        key = call_key(call)
        results = [digest for seen, digest in self._recent_calls if seen == key]
        latest = results[-REPEATED_CALL_LIMIT:]
        if len(latest) < REPEATED_CALL_LIMIT or len(set(latest)) > 1:
            return None
        head, step = self._recent_heads.get(key, ("", REPEATED_CALL_STEP))
        if step != REPEATED_CALL_STEP:
            # An empty result or a made change asks nothing, so the step says
            # what it means and nothing is quoted.
            quoted = ""
        elif key in self._repeat_quoted:
            # One copy of the result in context is enough.
            quoted = " Its result is quoted in this call's first refusal as a repeat, above."
        else:
            quoted = f" It said: {head}" if head else ""
            self._repeat_quoted.add(key)
        return (
            f"this `{call.name}` call ran {len(results)} times in the last "
            f"{len(self._recent_calls)} calls with the same result each time, and would "
            f"give it again.{quoted} {step}"
        )

    def _tool_run_as_command(self, call: ToolCall) -> str | None:
        """Why a `bash` call runs one of this session's tools as a command, or None.

        "Command not found" does not tell a model that the name is a tool, so it
        tends to try again. A tool name that is also a program on PATH, or a
        shell builtin such as `read` or `history`, is left alone.
        """
        if call.name != "bash":
            return None
        command = call.arguments.get("command")
        if not isinstance(command, str):
            return None
        tools = {tool.name for tool in self._tools()}
        for head in command_heads(command):
            if head in tools and head not in _SHELL_BUILTINS and shutil.which(head) is None:
                return (
                    f"`{head}` is one of your tools, not a shell command. Call the "
                    f"`{head}` tool itself, with its arguments, instead of bash."
                )
        return None


def call_key(call: ToolCall) -> str:
    """What makes two tool calls the same call: the tool and its arguments.

    A list argument that differs only by repeated entries is the same argument.
    """
    arguments = _dedupe_lists(call.arguments)
    return f"{call.name}\0{json.dumps(arguments, sort_keys=True, default=str)}"


def command_heads(command: str) -> list[str]:
    """The first word of each command in a shell line, roughly.

    Heredoc bodies and quoted strings are taken out first: a commit message
    in a heredoc has lines that begin with any word, and none of them runs.
    """
    lines: list[str] = []
    ends: list[tuple[str, bool]] = []
    for line in command.split("\n"):
        if ends:
            end, tabs = ends[0]
            if (line.lstrip("\t") if tabs else line) == end:
                ends.pop(0)
            continue
        lines.append(line)
        ends.extend((match[3], match[1] == "-") for match in _HEREDOC.finditer(line))
    unquoted = _QUOTED.sub("''", "\n".join(lines))
    heads: list[str] = []
    for segment in re.split(r"&&|\|\||[;|\n]", unquoted):
        words = segment.split()
        if words:
            heads.append(words[0])
    return heads


def _text(result: AgentToolResult) -> str:
    return "\n".join(b.text for b in result.content if isinstance(b, TextContent))


def _dedupe_lists(value: object) -> object:
    """Drop repeated entries from lists of hashable items, keeping first-seen order."""
    if isinstance(value, dict):
        return {key: _dedupe_lists(item) for key, item in value.items()}
    if isinstance(value, list):
        items = [_dedupe_lists(item) for item in value]
        if not all(isinstance(item, Hashable) for item in items):
            return items
        seen: set[tuple[type, object]] = set()
        unique: list[object] = []
        for item in items:
            marker = (type(item), item)
            if marker not in seen:
                seen.add(marker)
                unique.append(item)
        return unique
    return value


def _named_call(text: str, tools: set[str]) -> re.Match[str] | None:
    """The first call in `text` to one of `tools`: its name, then arguments."""
    if not tools:
        return None
    names = "|".join(re.escape(name) for name in sorted(tools, key=len, reverse=True))
    return re.search(rf"(?<![\w`])`?(?:{names})`?(?:\(|\s+[A-Za-z_]\w*=)", text)


def _call_sentence(text: str, tools: set[str]) -> str | None:
    """The first sentence of `text` holding a call to one of `tools`, bounded.

    A sentence longer than `REPEATED_CALL_QUOTE` is quoted from the call on.
    """
    for line in text.splitlines():
        for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
            found = _named_call(sentence, tools)
            if found is None:
                continue
            if len(sentence) > REPEATED_CALL_QUOTE:
                sentence = sentence[found.start() :]
            return sentence[:REPEATED_CALL_QUOTE]
    return None
