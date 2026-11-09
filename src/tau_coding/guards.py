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
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass

from tau_agent.loop import RAW_ARGUMENTS
from tau_agent.messages import (
    REASONING_BUDGET_CUT,
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolCall,
    UserMessage,
    reasoning_loop_cut,
)
from tau_agent.tools import AgentTool, AgentToolResult
from tau_agent.types import JSONValue
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

#: A chat template's tool-call and channel control tokens, which are not ordinary
#: text. The pipe forms (`<|channel>`, `<channel|>`, `<|channel|>`, and the same
#: for `tool_call`) always count. A bare `<tool_call>` counts only with a function
#: or JSON call opening after it, so a file that merely names the tag is left
#: alone: it may be a parser of that markup, or its docs or tests.
TEMPLATE_ARGUMENT_TOKEN = re.compile(
    r"<\|(?:tool_call|channel)\|?>|<(?:tool_call|channel)\|>"
    r"|</?tool_call>\s*(?:<function=|\{\s*\"name\"\s*:)"
)
#: The same tokens in a reply's text. Without the tag, a call counts by its
#: function opening on a parameter, or its function close on the tag's close, so
#: a bare `<function=x>` does not.
TEMPLATE_TOOL_CALL_TOKEN = re.compile(
    TEMPLATE_ARGUMENT_TOKEN.pattern
    + r"|<function=[\w.-]+>\s*<parameter=|</function>\s*</tool_call>"
)
#: How the sent call closed just before a leaked template tail: a string's end
#: quote, bare, escaped or the template's own, and the call's closing brace.
TEMPLATE_LEAK_CALL_CLOSE = re.compile(r'(?:\\*"|<\|"\|>)\s*\}+\s*$')
#: Arguments that only label a call. A template tail leaked into one is cut like
#: one in an optional argument, even where the schema requires it, because what
#: the call does does not depend on it.
LABEL_ARGUMENTS = frozenset({"description"})
#: Characters a template-leak refusal quotes from each end of the clean value.
TEMPLATE_LEAK_QUOTE = 60
#: A reply that writes its tool call as text ends the run with nothing done, so
#: the session asks for the call again, a bounded number of times in a row (a
#: reply that makes a call resets it) and per run.
MAX_TEXT_TOOL_CALL_FOLLOW_UPS = 2
MAX_TEXT_TOOL_CALL_FOLLOW_UPS_PER_RUN = 4
TEXT_TOOL_CALL_NUDGE = (
    "Your last reply wrote a tool call as text, so it did not run. Make the call "
    "as a tool call, or answer without the markup."
)

#: A reply cut off at the output limit before it acted would end the run with
#: nothing done, so the session asks once more. Bounded in a row, so a model that
#: never acts stops, and per run over the asks whose reply made no tool call, so
#: one that acts just enough in between stops too.
MAX_OUTPUT_LIMIT_CONTINUATIONS = 2
MAX_OUTPUT_LIMIT_CONTINUATIONS_PER_RUN = 6
#: The follow-up to a turn the reasoning-loop guard cut, whose reasoning is not
#: replayed.
LOOP_CUT_NUDGE = (
    "Your last reply hit the output limit before you acted, and its reasoning was "
    "dropped. Do not work the whole problem out again in your head: take the next "
    "concrete step now with a tool call, or answer briefly."
)
#: The follow-up to a turn cut at its thinking budget.
BUDGET_CUT_NUDGE = (
    "Your last reply used up its thinking budget without acting and was stopped. "
    "Do not start your reasoning over. Take the next concrete step now with a tool "
    "call: run what you were working out, or write what you have, then check it."
)
#: The follow-up to each later budget cut in a row, when the first nudge's reply
#: also thought to its budget: this one asks for the write itself.
BUDGET_CUT_WRITE_NUDGE = (
    "You were stopped again: your thinking budget ran out. Do not think it "
    "through again: no more reasoning. Reply now with only a tool call that "
    "writes what you have so far, even partial, to the file you are working on; "
    "then run it."
)
#: Budget-cut follow-ups allowed in a row: their replies think on a share of the
#: budget, so several take no more than one full turn.
MAX_BUDGET_CUT_CONTINUATIONS = 4
#: The follow-up to a cut-off reply. It doesn't say the reasoning is still there:
#: a reply can be empty, a server can drop it on replay, and compaction can run.
OUTPUT_LIMIT_NUDGE = (
    "Your last reply hit the output limit before you acted. Do not start your "
    "reasoning over or work the rest out in your head. Take the next concrete "
    "step now with a tool call, or answer briefly."
)

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


@dataclass(frozen=True)
class TemplateLeak:
    """A chat template's tool-call token found in one string of a call's arguments."""

    #: Where the string sits: `description`, or `items[0].description`.
    path: str
    #: Whether the string is inside a list or object rather than an argument itself.
    nested: bool
    token: str
    #: The string up to the token, without the call close sent just before it.
    clean: str

    def problem(self, tool: str, missing: list[str]) -> str:
        """Why the call carrying this leak is refused."""
        if self.path == RAW_ARGUMENTS:
            return (
                f"its arguments are not valid JSON and carry the chat template's own "
                f"tool-call token (`{self.token}`), so the reply's next call ran into this "
                "one; send one call per reply, with the text alone."
            )
        lacks = ""
        if missing:
            names = " and ".join(f"`{name}`" for name in missing)
            lacks = f", and it has no {names}, which `{tool}` needs"
        return (
            f"its arguments carry the chat template's own tool-call tokens "
            f"(`{self.token}` in `{self.path}`), so the reply's next call ran into this "
            f"one{lacks}; send one call per reply, with the text alone. If the text "
            "must really hold that token, split it across strings in a script that "
            "joins them."
        )

    def resend(self, tool: str, missing: list[str]) -> str:
        """The call to send instead.

        A call short of required arguments is named, not quoted: its clean
        prefix can hold prose that ran on from the value, and quoting that as
        the call to send repeats both faults.
        """
        if self.path == RAW_ARGUMENTS:
            return (
                f"Send `{tool}` again with its arguments as JSON, each string ending before "
                f"`{self.token}` and before any reasoning that ran on into it. Reason before "
                "a call, not inside it."
            )
        if missing:
            names = " and ".join(f"`{name}`" for name in missing)
            return (
                f"Send `{tool}` again with {names} set, and `{self.path}` holding only "
                f"its own text: it ends before `{self.token}`, and before any reasoning "
                "that ran on into it. Reason before a call, not inside it."
            )
        if not self.clean:
            return f"The call to send is this same call with `{self.path}` empty."
        quote = self.clean
        if len(quote) > 2 * TEMPLATE_LEAK_QUOTE + 1:
            quote = f"{quote[:TEMPLATE_LEAK_QUOTE]}…{quote[-TEMPLATE_LEAK_QUOTE:]}"
        return (
            f"The call to send is this same call with `{self.path}` ending before "
            f"`{self.token}`, as `{quote}`."
        )


class OutputLimitNudges:
    """Decides when a reply cut off at the output limit gets a follow-up.

    The session passes `queued`, its pending follow-ups. A steering message
    queued beside a nudge goes first and takes the request, so the nudge's reply
    is the turn after the loop sends it (`sent`), not the next to end; and a
    cut-off while a nudge still waits queues no second one.
    """

    def __init__(self, queued: Callable[[], Sequence[AgentMessage]]) -> None:
        self._queued = queued
        self._in_a_row = 0
        self._budget_cuts_in_a_row = 0
        self._unanswered_this_run = 0
        self._nudge: UserMessage | None = None
        self._awaiting_reply = False

    def reset(self) -> None:
        """Forget the last run's counts."""
        self._in_a_row = self._budget_cuts_in_a_row = self._unanswered_this_run = 0
        self._nudge = None
        self._awaiting_reply = False

    def sent(self, message: AgentMessage) -> None:
        """Note that the loop delivered `message`."""
        if message is self._nudge:
            self._nudge = None
            self._awaiting_reply = True

    def after_turn(self, message: AssistantMessage) -> UserMessage | None:
        """The follow-up to queue after `message`, if it needs one."""
        if self._awaiting_reply:
            self._awaiting_reply = False
            if not message.tool_calls:
                self._unanswered_this_run += 1
        if message.stop_reason != "length" or message.tool_calls:
            self._in_a_row = self._budget_cuts_in_a_row = 0
            return None
        text = self._follow_up(message)
        budget_cut = text in (BUDGET_CUT_NUDGE, BUDGET_CUT_WRITE_NUDGE)
        if (
            self._waiting()
            or self._in_a_row
            >= (MAX_BUDGET_CUT_CONTINUATIONS if budget_cut else MAX_OUTPUT_LIMIT_CONTINUATIONS)
            or self._unanswered_this_run >= MAX_OUTPUT_LIMIT_CONTINUATIONS_PER_RUN
        ):
            return None
        self._in_a_row += 1
        self._budget_cuts_in_a_row = self._budget_cuts_in_a_row + 1 if budget_cut else 0
        self._nudge = UserMessage(content=text)
        return self._nudge

    def _follow_up(self, message: AssistantMessage) -> str:
        """What to ask after `message`, by how it was cut off."""
        if reasoning_loop_cut(message):
            return LOOP_CUT_NUDGE
        if not any(d.type == REASONING_BUDGET_CUT for d in message.diagnostics or []):
            return OUTPUT_LIMIT_NUDGE
        return BUDGET_CUT_WRITE_NUDGE if self._budget_cuts_in_a_row else BUDGET_CUT_NUDGE

    def undelivered(self) -> UserMessage | None:
        """Take the nudge still waiting, if any, for the session to withdraw."""
        nudge, self._nudge = self._nudge, None
        return nudge

    def _waiting(self) -> bool:
        return self._nudge is not None and any(message is self._nudge for message in self._queued())


class ToolCallGuards:
    """Refuses tool calls that cannot help, for the length of one run.

    `check` is the answer for `before_tool_call`. `record` remembers what a call
    that ran gave.
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
        #: The note each call whose leaked template tail was cut adds to its result.
        self._template_cut_notes: dict[str, str] = {}
        #: Follow-ups in a row, and this run, to replies that wrote a call as text.
        self._text_call_follow_ups = 0
        self._run_text_call_follow_ups = 0

    def reset(self) -> None:
        """Forget everything; each run starts clean."""
        self._recent_calls.clear()
        self._recent_heads.clear()
        self._repeat_quoted.clear()
        self._bash_digests.clear()
        self._refused_in_a_row = 0
        self._cleared_call_ids.clear()
        self._template_cut_notes.clear()
        self._text_call_follow_ups = self._run_text_call_follow_ups = 0

    def check(self, call: ToolCall) -> tuple[bool, str | None]:
        """Whether to refuse `call`, and the note to answer it with."""
        leak = self._cut_template_tail(call)
        # Arguments that did not parse sit in one raw string, so every required
        # argument would look missing.
        raw = set(call.arguments) == {RAW_ARGUMENTS}
        missing = self._missing_required(call) if leak is not None and not raw else []
        problem = (leak.problem(call.name, missing) if leak is not None else None) or (
            self.problem(call)
        )
        refused, note = self.verdict(call, problem)
        if refused and leak is not None and note is not None and note.startswith("Not run: "):
            note = f"{note} {leak.resend(call.name, missing)}"
        return refused, note

    def take_cut_note(self, call: ToolCall) -> str | None:
        """The note saying what was cut from `call`'s arguments, if anything was."""
        return self._template_cut_notes.pop(call.id, None)

    def wants_text_call_follow_up(self, message: AssistantMessage) -> bool:
        """Whether `message` wrote a tool call as text and should be asked again.

        Only a reply that ended on "stop" counts: a length stop has its own
        handling, and an error or abort ends the run. The text must hold the
        template's own call markup, so a reply that names `<tool_call>` in prose
        is left to end the run.
        """
        if (
            message.tool_calls
            or message.stop_reason != "stop"
            or not TEMPLATE_TOOL_CALL_TOKEN.search(message.text)
        ):
            self._text_call_follow_ups = 0
            return False
        if (
            self._text_call_follow_ups >= MAX_TEXT_TOOL_CALL_FOLLOW_UPS
            or self._run_text_call_follow_ups >= MAX_TEXT_TOOL_CALL_FOLLOW_UPS_PER_RUN
        ):
            self._text_call_follow_ups = 0
            return False
        self._text_call_follow_ups += 1
        self._run_text_call_follow_ups += 1
        return True

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

    def _cut_template_tail(self, call: ToolCall) -> TemplateLeak | None:
        """Cut leaked template tails from arguments the call can run without, or
        return the leak that refuses it.

        A model can run the reply's next call into this one's last string.
        Refusing such a call whole tends to bring the same call back, so where
        every leak is a top-level argument the tool's schema declares and does
        not require (or one that only labels the call), each is cut at its
        token and the call runs; the result says what was dropped. A leak in a
        required or nested value, or in a call whose schema does not declare
        it, is refused, since its clean prefix may not be what the model meant.
        """
        leaks = _template_tokens_in_arguments(call)
        if not leaks:
            return None
        cuttable = self._cuttable_arguments(call.name)
        for leak in leaks:
            if leak.nested or leak.path not in cuttable:
                return leak
        for leak in leaks:
            call.arguments[leak.path] = leak.clean
        dropped = ", ".join(f"`{leak.token}` in `{leak.path}`" for leak in leaks)
        self._template_cut_notes[call.id] = (
            f"Note: the text after the chat template's own tool-call token ({dropped}) was "
            "the reply's next call run into this one. It was dropped, and the call ran "
            "without it. Send each call in its own reply."
        )
        return None

    def _tool_named(self, name: str) -> AgentTool | None:
        return next((tool for tool in self._tools() if tool.name == name), None)

    def _missing_required(self, call: ToolCall) -> list[str]:
        """The arguments `call`'s tool requires that the call does not carry."""
        tool = self._tool_named(call.name)
        required = tool.parameters.get("required") if tool is not None else None
        if not isinstance(required, list):
            return []
        return [name for name in required if isinstance(name, str) and name not in call.arguments]

    def _cuttable_arguments(self, tool_name: str) -> frozenset[str]:
        """The arguments of `tool_name` its schema declares and a call can run without."""
        tool = self._tool_named(tool_name)
        if tool is None:
            return frozenset()
        properties = tool.parameters.get("properties")
        if not isinstance(properties, Mapping):
            return frozenset()
        required = tool.parameters.get("required")
        needed = (
            {name for name in required if isinstance(name, str)}
            if isinstance(required, list)
            else set()
        )
        return frozenset(
            name for name in properties if name not in needed or name in LABEL_ARGUMENTS
        )

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


def _template_tokens_in_arguments(call: ToolCall) -> list[TemplateLeak]:
    """Each string in `call`'s arguments that carries a chat template's tool-call tokens."""
    leaks: list[TemplateLeak] = []
    pending: list[tuple[str, JSONValue, bool]] = [
        (name, value, False) for name, value in call.arguments.items()
    ]
    while pending:
        name, value, nested = pending.pop(0)
        if isinstance(value, str):
            match = TEMPLATE_ARGUMENT_TOKEN.search(value)
            if match is not None:
                token = match.group(0).split(">", 1)[0] + ">"
                clean = TEMPLATE_LEAK_CALL_CLOSE.sub("", value[: match.start()]).rstrip()
                leaks.append(TemplateLeak(path=name, nested=nested, token=token, clean=clean))
        elif isinstance(value, dict):
            pending.extend((f"{name}.{key}", item, True) for key, item in value.items())
        elif isinstance(value, list):
            pending.extend((f"{name}[{i}]", item, True) for i, item in enumerate(value))
    return leaks


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
