"""Subagents extension for Tau, ported from tintinweb/pi-subagents.

Registers an `agent` tool that spawns autonomous subagents in-process (a
scoped `CodingSession` with its own tools and system prompt), a
`get_subagent_result` tool for background runs, a `steer_subagent` tool for
redirecting live runs, a `stop_subagent` tool for ending them, and an
`/agents` command.

Foreground agents block and return their final assistant text. Background
agents return an id immediately and deliver a `<task-notification>` back into
the parent conversation when they finish. Only background agents count toward
the `maxConcurrent` limit; excess background spawns are queued FIFO and started
as slots free up.

Install by copying this directory into `~/.tau/extensions/subagents/`, or run:

    tau -e examples/extensions/subagents
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from tau_agent.session import SessionEntry
from tau_agent.tools import AgentTool, AgentToolResult
from tau_coding import (
    CodingSession,
    CodingSessionConfig,
    FileCredentialStore,
    auto_compaction_threshold_for_context_window,
    load_provider_settings,
    provider_has_usable_credentials,
    resolve_provider_selection,
)
from tau_coding.extensions import (
    ExtensionAPI,
    SessionShutdownEvent,
    SessionStartEvent,
)
from tau_coding.provider_runtime import create_model_provider
from tau_coding.thinking import DEFAULT_THINKING_LEVEL, THINKING_LEVELS

from .agents import AgentDefinition, format_agent_type_list, load_agent_definitions
from .agents_menu import (
    show_agents_menu,
    stop_run,
    supports_menu,
)
from .fork import ForkCapture, build_fork_entries, capture_fork, wrap_fork_task
from .group_join import DEFAULT_TIMEOUT, STRAGGLER_TIMEOUT, GroupJoinManager
from .jobs import Job, JobStore
from .jobs_tool import JobActions, build_jobs_tool, jobs_command_text
from .jobs_watch import JobWatcher
from .memory import prepare_memory
from .nesting import SPAWN_TOOLS, Nesting, current_nesting, nested
from .notification_render import (
    SUCCESS_STATUSES,
    render_agent_result,
    render_notification,
    render_stall_notice,
    stat_parts,
)
from .output_file import (
    OutputFileWriter,
    full_report_path,
    output_file_path,
    sweep_transcripts,
)
from .phase import (
    PHASE_AWAITING_MODEL,
    PHASE_IN_TOOL,
    PHASE_QUEUED,
    PHASE_STREAMING,
    format_phase,
    format_seconds,
    phase_elapsed,
    streamed_tokens,
)
from .prompts import (
    build_child_system_prompt,
    build_parent_context,
    detect_environment,
    inherited_resource_paths,
    resolve_skill_blocks,
    sub_agent_role,
)
from .schedule import SubagentScheduler
from .schedule_store import ScheduleStore, resolve_store_path
from .settings import SubagentSettings, load_subagent_settings
from .tool_log import (
    RECENT_TOOLS_KEPT,
    ToolCallRecord,
    format_tool_log,
    summarize_args,
    tool_log_details,
)
from .worktree import (
    WORKTREE_ERROR_MESSAGE,
    Worktree,
    cleanup_worktree,
    create_worktree,
    prune_worktrees,
)

if TYPE_CHECKING:
    from tau_agent.events import AgentEvent

# Fit-or-fetch caps for the model-visible completion notifications: a result
# that fits is included whole; a longer one is omitted entirely (with a
# get_subagent_result pointer) rather than truncated — a partial preview
# tempts the parent to act on incomplete information instead of fetching.
INDIVIDUAL_RESULT_CHARS = 500
GROUP_RESULT_CHARS = 300
# Display-only budget for the foreground card's expanded view (Ctrl+O). More
# generous than the notification previews above, which also enter the model's
# context: the old generic result block showed the full text, so a tight cap
# here would be a regression.
FOREGROUND_RESULT_CHARS = 6_000
RECORD_RESULT_CHARS = 4_000
BATCH_DEBOUNCE_SECONDS = 0.1
NUDGE_HOLD_SECONDS = 0.2
#: How send_custom_message queues a notice while the parent is running.
DeliverAs = Literal["steer", "follow_up"]
#: How often the stall check looks at running background agents (at most;
#: a shorter stallNoticeSeconds checks that often instead).
STALL_CHECK_SECONDS = 30.0
#: How long stop_subagent waits for a cancelled run to wind down before it
#: reports the stop as requested but not yet taken.
STOP_WAIT_SECONDS = 5.0
GROUP_TIMEOUT_SECONDS = DEFAULT_TIMEOUT
STRAGGLER_TIMEOUT_SECONDS = STRAGGLER_TIMEOUT
STALE_AFTER_SECONDS = 600.0
#: How often the job watcher reads the registry. Two seconds is a finished
#: job reported well after the fact, and a readdir nobody notices.
JOBS_POLL_SECONDS = 2.0
TERMINAL_STATUSES = (
    "completed",
    "completed_no_report",
    "turn_limit",
    "aborted",
    "error",
    "cancelled",
)
#: Steering message sent once when a child ends its run with an empty message.
REPORT_REQUEST_MESSAGE = (
    "Your last message was empty, so the agent that started you has no report."
    " Reply now with your final report: what you found or did, what is not"
    " done, and where the work is. Do not start new work."
)
#: A final report longer than this is sent back to its author to condense. Tau
#: cuts an oversized tool result, and a report with its middle cut out misleads
#: the agent that asked for it.
#: Under the size a parent on a 32K-token model is shown whole (16,000
#: characters), with room for the header the result carries.
CONDENSE_OVER_CHARS = 12_000
CONDENSE_TARGET_CHARS = 8_000
#: A shorter answer to the request is a remark ("Done."), not a report.
CONDENSE_MIN_CHARS = 300
#: Sent once to a child whose final report is over `CONDENSE_OVER_CHARS`.
CONDENSE_REQUEST_MESSAGE = (
    f"Condense your report to under {CONDENSE_TARGET_CHARS:,} characters; keep"
    " every finding, drop narration. Reply with the condensed report itself;"
    " do not call tools."
)


def report_tool_options(tool_class: type = AgentTool) -> dict[str, str]:
    """`AgentTool` arguments for a tool whose result is a child's report.

    They ask Tau to show the result whole rather than cut it to a window.
    Empty on a Tau whose `AgentTool` has no `output_window`; that Tau cuts
    the result as it always did.
    """
    if "output_window" in getattr(tool_class, "__dataclass_fields__", {}):
        return {"output_window": "whole"}
    return {}


# Event types that change what a viewer would show (session.messages / status).
# Streaming (message_update) is skipped: it never touches session.messages, so
# pushing on it would redraw identical content on every token. Errors need no
# entry of their own: they arrive as assistant messages with stop_reason
# "error"/"aborted", i.e. through message_end.
RUN_PUSH_EVENTS = frozenset(
    {"message_end", "tool_execution_start", "tool_execution_end", "turn_end"}
)


def format_job_notice(job: Job) -> str:
    """One line telling the parent a job is over, in the row's vocabulary.

    A job with no `done.json` and no live pid is `lost`, which is the only
    answer there is to "how did it end": the process is gone and nothing
    recorded why, so the notice says `lost` where a finished job names its
    state, and shows no exit code the registry never got. The wording lives
    in one function so a future `/jobs`, the strip, and this notice cannot
    drift apart.
    """
    elapsed = int(job.elapsed())
    if job.rc is None and job.status() == "lost":
        return f"Job {job.id} ({job.label}) lost after {elapsed}s."
    rc = "rc=?" if job.rc is None else f"rc={job.rc}"
    return f"Job {job.id} ({job.label}) {job.status()} after {elapsed}s ({rc})."


AGENT_SNIPPET = "Spawn an autonomous subagent for delegated tasks."
AGENT_SNIPPET_DELEGATE = (
    "Spawn a subagent for a separate piece of work, never to pass on your own task."
)


def agent_tool_description(type_list: str, *, delegate: bool) -> str:
    """The `agent` tool's description for one session.

    Says when not to spawn as well as how: a model reads this far more often
    than it reads any prompt about delegation. `delegate` adds the line for
    a session that is itself a sub-agent. A fork passes its parent's value,
    so that its tool descriptions match the parent's (see `Nesting`).
    """
    delegate_line = (
        " You are a sub-agent yourself: spawn only for a separate subtask,"
        " never to pass on the task you were given."
        if delegate
        else ""
    )
    return (
        "Spawn an autonomous subagent to handle a task. The subagent works"
        " in its own context with its own tools and returns its final"
        " report.\n\nWhen not to use: do the task yourself when a few tool"
        " calls would do it. Use agents for independent pieces of work that"
        " can run in parallel, or to keep heavy reading out of your context."
        " Never forward your whole task to a single agent."
        f"{delegate_line}\n\nSet run_in_background=true for long tasks; a"
        " completion notification will arrive when it finishes. Use"
        " steer_subagent to redirect a running agent, and resume=<id> with a"
        " new prompt to continue a finished agent's session. Two ways to"
        " share this conversation: inherit_context=true prepends a text"
        " digest of the user and assistant messages only (no tool calls or"
        " results) to a fresh agent of any type/model; subagent_type fork"
        " runs on this conversation's actual messages, including every tool"
        " call and result, with the same system prompt, tools, and model, so"
        " the prompt cache is shared.\n\nAvailable agent types:"
        f"\n{type_list}"
    )


def depth_limit_message(depth: int) -> str:
    """What a sub-agent at the nesting limit is told when it tries to spawn."""
    return (
        f"No agent was started: this session is a sub-agent {depth} levels"
        " deep, which is the limit (the maxDepth setting). Do the task"
        " yourself with your own tools, then report the result."
    )


def soft_limit_message(max_turns: int, grace_turns: int) -> str:
    """The steering message sent when a run reaches its turn limit.

    States the turns left before the hard stop, so the child can plan its
    wrap-up instead of losing uncommitted work to the cancel.
    """
    limit_noun = "turn" if max_turns == 1 else "turns"
    left_noun = "turn" if grace_turns == 1 else "turns"
    return (
        f"You have reached your turn limit ({max_turns} {limit_noun}). You have"
        f" {grace_turns} more {left_noun} before this run is stopped. Do not"
        " start new work. Finish the step in progress; if you changed files"
        " under version control, commit or stage the work that is done. Then"
        " give your final report: what is done, what is not done, and where"
        " the work is."
    )


class _MemoryStorage:
    """Append-only in-memory storage for subagent sessions."""

    def __init__(self) -> None:
        self.entries: list[SessionEntry] = []

    async def append(self, entry: SessionEntry) -> None:
        self.entries.append(entry)

    async def read_all(self) -> list[SessionEntry]:
        return list(self.entries)


def _child_compact_threshold(selection: object) -> int | None:
    """Where a child compacts, from its model's configured context window.

    A child session carries no provider settings, so Tau would size it by its
    128k default; for a smaller local model that point lies past what the
    server accepts, and the child would only compact after a refused request.
    """
    provider = getattr(selection, "provider", None)
    windows = getattr(provider, "context_windows", None) or {}
    window = windows.get(getattr(selection, "model", None))
    if not isinstance(window, int) or window <= 0:
        return None
    return auto_compaction_threshold_for_context_window(window)


@dataclass(slots=True)
class AgentRun:
    """State of one spawned subagent."""

    agent_id: str
    agent_type: str
    description: str
    prompt: str
    background: bool
    status: str = "running"
    result_text: str = ""
    error: str | None = None
    turns: int = 0
    tool_calls: int = 0
    revision: int = 0
    task: asyncio.Task[None] | None = None
    session: CodingSession | None = None
    provider: object | None = None
    result_consumed: bool = False
    started_at: float | None = None
    completed_at: float | None = None
    context_tokens: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_cache_write: int = 0
    # Liveness, which tells a busy run from a wedged one: when the child last
    # emitted any event, and the tool it is inside right now. A tool that never
    # returns emits nothing, and turns only count once a turn ends, so without
    # these a hung run looks exactly like a slow one.
    last_event_at: float | None = None
    current_tool: str | None = None
    current_tool_started_at: float | None = None
    # What the run is doing now (see phase.py) and since when. streamed_chars
    # counts the deltas of the response streaming now, for a token estimate
    # before its usage arrives. tools_in_flight maps a tool call id to its
    # (name, start time); current_tool is the latest one still running.
    phase: str = PHASE_QUEUED
    phase_since: float | None = None
    streamed_chars: int = 0
    tools_in_flight: dict[str, tuple[str, float]] = field(default_factory=dict)
    # The last few tool calls (see tool_log.py), for get_subagent_result's peek.
    # quiet_since of the stretch the stall notice last fired for, so it fires
    # once per stretch and again after the agent speaks and goes quiet anew.
    stall_noticed_for: float | None = None
    recent_tools: deque[ToolCallRecord] = field(
        default_factory=lambda: deque(maxlen=RECENT_TOOLS_KEPT)
    )
    # Terminal error carried by the last assistant message seen so far. Only a
    # tentative signal while the run is live: tau (like pi) can recover from a
    # mid-run overflow error via compaction + auto-retry, so a later successful
    # assistant message clears it and the final verdict is taken once the event
    # stream is exhausted (never mid-stream).
    pending_error: str | None = None
    # Resolved model id actually running this agent (None while queued).
    # requested_model below is only the tool-param request, which frontmatter
    # may override — the viewer header shows this resolved value.
    model: str | None = None
    requested_provider: str | None = None
    requested_model: str | None = None
    requested_thinking: str | None = None
    requested_max_turns: int | None = None
    max_turns: int | None = None
    grace_turns: int = 5
    soft_limit_reached: bool = False
    # The last assistant message had no text and no tool calls; the child is
    # asked once (report_requested) to give its report before the run ends.
    report_missing: bool = False
    report_requested: bool = False
    #: Finished, and answering the request to condense its report.
    condensing: bool = False
    # The latest non-empty assistant text, kept for a partial result when the
    # run is stopped or cancelled before it reports.
    last_text: str = ""
    aborted: bool = False
    pending_steers: list[str] = field(default_factory=list)
    # Live stats ticker (foreground runs only): the executor's on_update
    # callback, set for the duration of the blocking tool call. Takes a partial
    # AgentToolResult (pi's AgentToolUpdateCallback). `last_progress` dedups so
    # the line only repaints when a stat actually changed.
    on_update: Callable[[AgentToolResult], None] | None = None
    last_progress: str = ""
    join_mode: str | None = None
    requested_isolation: str | None = None
    # `isolated: true` (pi's param): spawn the child without extensions, so
    # it gets core tools only and cannot spawn subagents of its own.
    requested_isolated: bool = False
    # Parent-session snapshot for fork runs, captured at tool-call time and
    # cleared once the seed entries are built (it pins a full transcript copy).
    fork_capture: ForkCapture | None = None
    # Seeded parent message count for fork runs; None for every other agent
    # type. Sizes the output-file skip so the seeded prefix stays out of it.
    fork_inherited: int | None = None
    worktree: Worktree | None = None
    used_worktree: bool = False
    output_writer: OutputFileWriter | None = None
    output_file: str | None = None
    # Provider selection checked by the agent tool before it reported the
    # spawn. None for forks (resolved from their capture) and for scheduled
    # runs; _execute resolves those itself.
    selection: object | None = None
    # Per-run push listeners (component seam): fired when this run's content or
    # status changes so an open conversation viewer can re-render. The direct
    # analog of pi's ``session.subscribe(() => tui.requestRender())``. Safe to
    # call widget methods from here because runs are asyncio tasks on the TUI
    # event loop (see the design's push-refresh invariant).
    listeners: list[Callable[[], None]] = field(default_factory=list)

    @property
    def has_usage(self) -> bool:
        """Whether the provider reported any billed usage (pi gates on total > 0).

        Usage is always present on new-protocol assistant messages (zeros when
        the provider reports nothing), so presence can't be the signal anymore;
        a nonzero lifetime total is. Zero-usage runs fall back to the chars/4
        context estimate everywhere this gate is checked.
        """
        return lifetime_tokens(self) > 0


class SubagentManager:
    """Spawns and tracks subagent runs for one Tau session."""

    def __init__(self, api: ExtensionAPI) -> None:
        self._api = api
        # Read here, not at spawn time: a manager is built in `setup()`,
        # which runs inside the scope its session's parent set, whereas a
        # queued or scheduled spawn can start from some other task's context.
        self.nesting = current_nesting()
        self._runs: dict[str, AgentRun] = {}
        self._counter = 0
        self._settings: SubagentSettings | None = None
        self._running_background = 0
        self._queue: list[tuple[AgentRun, AgentDefinition]] = []
        self._shutting_down = False
        self._batch: list[AgentRun] = []
        self._batch_timer: asyncio.TimerHandle | None = None
        self._batch_counter = 0
        self._group_join: GroupJoinManager | None = None
        self._worktree_repos: set[str] = set()
        self._nudge_timers: dict[str, asyncio.TimerHandle] = {}
        # Terminal notices held while the parent is mid-run, keyed like the
        # nudge timers; flushed as steering at the parent's next turn_end.
        self._outbox: dict[str, Callable[[DeliverAs], None]] = {}
        self._stall_timer: asyncio.TimerHandle | None = None
        # Roster change signal: fired when the run list or a run's status
        # changes. On the component seam it is wired (in setup()) to the UI
        # controller's on_change, which refreshes the extension's own strip
        # and any open viewer. (The name predates the seam migration: it once
        # pointed at tau core's removed transcript-sources callback.)
        self.sources_changed: Callable[[], None] | None = None

    def _notify_sources(self) -> None:
        callback = self.sources_changed
        if callback is not None:
            with contextlib.suppress(Exception):
                callback()

    def _inherited_trust_config(self) -> dict[str, object]:
        """Hand the parent's project-trust decision to a child session.

        Tau 0.4+ gates project resources (`.tau/skills`, project extensions)
        on trust, and a headless child would otherwise resolve as untrusted
        and lose them. Earlier Tau has no trust gate and no such field.
        """
        try:
            trusted = self._api.context.project_trusted
        except AttributeError:
            return {}
        return {"trust_override": "approve" if trusted else "decline"}

    def _notify_run(self, run: AgentRun) -> None:
        """Fire a run's per-run listeners (viewer push); guarded per listener."""
        for listener in tuple(run.listeners):
            with contextlib.suppress(Exception):
                listener()

    @property
    def runs(self) -> dict[str, AgentRun]:
        return self._runs

    @property
    def max_concurrent(self) -> int:
        return self._get_settings().max_concurrent

    def definitions(self) -> dict[str, AgentDefinition]:
        return load_agent_definitions(self._api.context.cwd)

    @property
    def max_depth(self) -> int:
        """The nesting limit: the top session's, for every session below it."""
        inherited = self.nesting.max_depth
        return self._get_settings().max_depth if inherited is None else inherited

    def _child_nesting(self, definition: AgentDefinition) -> Nesting:
        """The nesting a child of this session runs at.

        A fork is its parent continued, so it is a delegate only if its
        parent is one: its tool descriptions must match the parent's.
        """
        return Nesting(
            depth=self.nesting.depth + 1,
            delegate=self.nesting.delegate if definition.fork else True,
            max_depth=self.max_depth,
        )

    def _get_settings(self) -> SubagentSettings:
        if self._settings is None:
            self._settings = load_subagent_settings(self._api.context.cwd)
        return self._settings

    def spawn(
        self,
        *,
        agent_type: AgentDefinition,
        prompt: str,
        description: str,
        background: bool,
        max_turns: int | None = None,
        isolation: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        thinking: str | None = None,
        bypass_queue: bool = False,
        fork_capture: ForkCapture | None = None,
        isolated: bool = False,
        selection: object | None = None,
    ) -> AgentRun:
        self._counter += 1
        run = AgentRun(
            agent_id=f"agent-{self._counter}",
            agent_type=agent_type.name,
            description=description,
            prompt=prompt,
            background=background,
            requested_provider=provider,
            requested_model=model,
            requested_thinking=thinking,
            requested_max_turns=max_turns,
            requested_isolation=isolation,
            requested_isolated=isolated,
            fork_capture=fork_capture,
            fork_inherited=len(fork_capture.messages) if fork_capture else None,
            selection=selection,
        )
        run.output_writer = OutputFileWriter(
            output_file_path(
                self._api.context.cwd,
                self._api.context.session_id,
                run.agent_id,
                # transcriptRetentionDays: 0 opts out of durable storage;
                # an unset value keeps durable transcripts indefinitely.
                durable=self._get_settings().transcript_retention_days != 0,
            ),
            run.agent_id,
            self._api.context.cwd,
            # Seeded parent messages stay out of the durable transcript.
            inherited=run.fork_inherited or 0,
        )
        run.output_file = str(run.output_writer.path)
        self._runs[run.agent_id] = run
        if background:
            run.join_mode = self._get_settings().default_join_mode
            if run.join_mode in ("smart", "group"):
                self._batch.append(run)
                self._schedule_batch_finalize()
        if (
            background
            and not bypass_queue
            and self._running_background >= self.max_concurrent
        ):
            run.status = "queued"
            self._queue.append((run, agent_type))
            self._notify_sources()
            return run
        self._start(run, agent_type)
        self._notify_sources()
        return run

    def resolve_selection(
        self,
        definition: AgentDefinition,
        *,
        provider: str | None,
        model: str | None,
    ) -> object:
        """Resolve the provider and model a non-fork run will use.

        Frontmatter wins over the tool param, per pi precedence. When neither
        names a provider or model, the child inherits the parent session's
        active selection, not the durable Tau settings default. The parent's
        model is only inherited on the parent's provider: another provider
        gets its own default model, since the parent's is rarely configured
        there. Raises ProviderConfigError for an unknown provider or model.
        """
        parent_provider = self._api.context.provider_name
        provider_name = provider or parent_provider
        inherited_model = (
            self._api.context.model if provider_name == parent_provider else None
        )
        return resolve_provider_selection(
            load_provider_settings(),
            provider_name=provider_name,
            model=definition.model or model or inherited_model,
        )

    def check_spawn_target(
        self,
        definition: AgentDefinition,
        *,
        provider: str | None,
        model: str | None,
    ) -> tuple[object | None, str | None]:
        """Resolve and check a run's provider before the spawn is reported.

        Returns (selection, None) when the run can start, or (None, error)
        with an error text that names the providers that have credentials.
        The credential check is skipped for the parent's own provider: the
        parent session is using it, so it works.
        """
        try:
            selection = self.resolve_selection(
                definition, provider=provider, model=model
            )
        except Exception as exc:  # noqa: BLE001 - reported as a tool error
            return None, (
                f"Cannot start the subagent: {exc}."
                f" {_authenticated_providers_text()}"
                " Omit provider and model to use the calling session's."
            )
        chosen = getattr(selection, "provider", None)
        name = getattr(chosen, "name", None)
        if name != self._api.context.provider_name and not _provider_is_usable(
            chosen
        ):
            return None, (
                f"Cannot start the subagent: provider {name} has no credentials."
                f" {_authenticated_providers_text()}"
                " Omit provider and model to use the calling session's."
            )
        return selection, None

    def _get_group_join(self) -> GroupJoinManager:
        if self._group_join is None:
            self._group_join = GroupJoinManager(
                self._deliver_group_notification,
                group_timeout=GROUP_TIMEOUT_SECONDS,
                straggler_timeout=STRAGGLER_TIMEOUT_SECONDS,
            )
        return self._group_join

    def _schedule_batch_finalize(self) -> None:
        if self._batch_timer is not None:
            self._batch_timer.cancel()
        self._batch_timer = asyncio.get_running_loop().call_later(
            BATCH_DEBOUNCE_SECONDS, self._finalize_batch
        )

    def _finalize_batch(self) -> None:
        self._batch_timer = None
        batch = self._batch
        self._batch = []
        if len(batch) >= 2:
            self._batch_counter += 1
            self._get_group_join().register_group(
                f"batch-{self._batch_counter}", [run.agent_id for run in batch]
            )
            for run in batch:
                if run.status in TERMINAL_STATUSES and run.status != "cancelled":
                    self._get_group_join().on_agent_complete(run)
            return
        for run in batch:
            if run.status in TERMINAL_STATUSES and run.status != "cancelled":
                self._deliver_background_result(run)

    def _start(self, run: AgentRun, definition: AgentDefinition) -> None:
        run.status = "running"
        run.started_at = time.monotonic()
        _set_phase(run, PHASE_AWAITING_MODEL, run.started_at)
        if run.background:
            self._running_background += 1
            self._watch_for_stalls()
        run.task = asyncio.get_running_loop().create_task(
            self._run_agent(run, definition)
        )

    def _watch_for_stalls(self) -> None:
        """Arm the stall check unless it is armed, off, or shutting down."""
        threshold = self._get_settings().stall_notice_seconds
        if self._stall_timer is not None or threshold <= 0 or self._shutting_down:
            return
        self._stall_timer = asyncio.get_running_loop().call_later(
            min(STALL_CHECK_SECONDS, threshold), self._check_stalls
        )

    def _check_stalls(self) -> None:
        """Tell the parent about background agents quiet past the threshold."""
        self._stall_timer = None
        threshold = self._get_settings().stall_notice_seconds
        now = time.monotonic()
        watching = False
        for run in list(self._runs.values()):
            if not run.background or run.status != "running":
                continue
            watching = True
            quiet_since = (
                run.last_event_at if run.last_event_at is not None else run.started_at
            )
            if quiet_since is None or now - quiet_since < threshold:
                continue
            if run.stall_noticed_for == quiet_since:
                continue
            run.stall_noticed_for = quiet_since
            try:
                self._send_stall_notice(run, now - quiet_since, now)
            except Exception:  # noqa: BLE001 - timer callbacks must never crash the loop
                pass
        if watching:
            self._watch_for_stalls()

    def _send_stall_notice(self, run: AgentRun, quiet: float, now: float) -> None:
        quiet_text = format_seconds(quiet)
        self._api.notify(
            f"Subagent {run.agent_id} ({run.description}) has been quiet for"
            f" {quiet_text}.",
            "warning",
        )
        content = (
            f"Background subagent {run.agent_id} ({run.description}) has sent no"
            f" events for {quiet_text} and may be stalled."
            f" {format_liveness(run, now)}\n"
            f"Check it with get_subagent_result (tools: 5), steer it with"
            " steer_subagent, or end it with stop_subagent. This notice is sent"
            " once per quiet stretch; a slow tool may simply still be working."
        )
        self._api.send_custom_message(
            content,
            custom_type="subagent-stall",
            details={
                "agent_id": run.agent_id,
                "description": run.description,
                "quiet": quiet_text,
                "phase": format_phase(run, now),
                "output_file": run.output_file,
            },
            deliver_as="follow_up",
        )

    def _drain_queue(self) -> None:
        if self._shutting_down:
            return
        while self._queue and self._running_background < self.max_concurrent:
            run, definition = self._queue.pop(0)
            if run.status != "queued":
                continue
            self._start(run, definition)

    async def resume(self, run: AgentRun, prompt: str) -> None:
        """Resume a finished run's live session with a follow-up prompt."""
        session = run.session
        assert session is not None
        run.status = "running"
        run.error = None
        run.pending_error = None
        run.result_text = ""
        run.aborted = False
        run.soft_limit_reached = False
        run.report_missing = False
        run.report_requested = False
        run.started_at = time.monotonic()
        run.completed_at = None
        # A stopped run can leave calls that never ended; they are gone now.
        run.tools_in_flight.clear()
        run.current_tool = None
        run.current_tool_started_at = None
        _set_phase(run, PHASE_AWAITING_MODEL, run.started_at)
        self._notify_sources()
        self._notify_run(run)
        final_text: list[str] = []
        try:
            async for event in session.prompt(prompt):
                self._observe(run, event, final_text)
                self._request_report_if_empty(run, event)
                await self._flush_output(run, session, event)
            run.result_text = final_text[-1] if final_text else ""
            await self._condense_report(run, session)
            run.context_tokens = session.context_token_estimate
            self._finalize_status(run)
        except asyncio.CancelledError:
            # The resume runs inline in the tool coroutine, so a parent Esc
            # (hard-cancel) lands here; settle the record before re-raising.
            run.status = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001 - report subagent failures as results
            run.status = "error"
            run.error = str(exc)
        finally:
            if run.status in PARTIAL_STATUSES:
                run.result_text = format_partial_result(run)
            await self._persist_record(run)
            run.completed_at = time.monotonic()
            self._notify_sources()
            self._notify_run(run)

    async def shutdown(self) -> None:
        self._shutting_down = True
        try:
            if self._batch_timer is not None:
                self._batch_timer.cancel()
                self._batch_timer = None
            self._batch.clear()
            if self._group_join is not None:
                self._group_join.cancel_all()
            for handle in self._nudge_timers.values():
                handle.cancel()
            self._nudge_timers.clear()
            self._outbox.clear()
            if self._stall_timer is not None:
                self._stall_timer.cancel()
                self._stall_timer = None
            for run, _definition in self._queue:
                if run.status == "queued":
                    run.status = "cancelled"
                    run.completed_at = time.monotonic()
            self._queue.clear()
            pending: list[asyncio.Task[None]] = []
            for run in self._runs.values():
                if run.task is not None and not run.task.done():
                    if run.session is not None:
                        run.session.cancel()
                    run.task.cancel()
                    pending.append(run.task)
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            for run in self._runs.values():
                await self.close_run(run)
            for repo in self._worktree_repos:
                with contextlib.suppress(Exception):
                    await prune_worktrees(Path(repo))
            self._worktree_repos.clear()
        finally:
            self._shutting_down = False
            self._notify_sources()

    async def evict_stale(self) -> None:
        """Close sessions of consumed terminal runs older than the staleness cap."""
        now = time.monotonic()
        for run in self._runs.values():
            if run.session is None and run.provider is None:
                continue
            if run.status not in TERMINAL_STATUSES or not run.result_consumed:
                continue
            if run.completed_at is None or now - run.completed_at < STALE_AFTER_SECONDS:
                continue
            await self.close_run(run)

    async def close_run(self, run: AgentRun) -> None:
        session = run.session
        if session is not None:
            try:
                await session.aclose()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        closer = getattr(run.provider, "aclose", None)
        if closer is not None:
            try:
                await closer()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
        run.session = None
        run.provider = None

    async def _run_agent(self, run: AgentRun, definition: AgentDefinition) -> None:
        try:
            # Around the child's load, where the child's own manager reads
            # the scope. A later resume=<id> runs outside it, which is why
            # the manager keeps what it read.
            with nested(self._child_nesting(definition)):
                await self._execute(run, definition)
        except asyncio.CancelledError:
            run.status = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001 - report subagent failures as results
            run.status = "error"
            run.error = str(exc)
        finally:
            await self._finalize_run(run)
            run.completed_at = time.monotonic()
            self._notify_sources()
            self._notify_run(run)
            if run.background:
                self._running_background -= 1
                self._drain_queue()
                self._handle_background_completion(run)

    async def _finalize_run(self, run: AgentRun) -> None:
        if run.status in PARTIAL_STATUSES:
            run.result_text = format_partial_result(run)
        writer = run.output_writer
        if writer is not None and not writer.started:
            # The run failed before its transcript existed (provider setup,
            # worktree creation). The notification names the output file, so
            # write a stub that records the prompt and the failure.
            with contextlib.suppress(Exception):
                await writer.write_initial(run.prompt)
                if run.status == "error":
                    await writer.write_error(run.error or "subagent failed")
        elif writer is not None and run.session is not None:
            with contextlib.suppress(Exception):
                await writer.flush(run.session.messages)
        if run.worktree is not None:
            await self._cleanup_worktree(run)
        await self._persist_record(run)

    async def _cleanup_worktree(self, run: AgentRun) -> None:
        worktree = run.worktree
        run.worktree = None
        assert worktree is not None
        try:
            result = await cleanup_worktree(worktree, run.description)
        except Exception:  # noqa: BLE001 - worktree cleanup is best-effort
            return
        if result.has_changes and result.branch:
            annotation = (
                f"\n\n---\nChanges saved to branch `{result.branch}`."
                f" Merge with: `git merge {result.branch}`"
            )
            run.result_text += annotation
            # Error terminals surface run.error, not result_text — annotate
            # both so committed work is never invisible.
            if run.status == "error":
                run.error = (run.error or "subagent failed") + annotation

    async def _persist_record(self, run: AgentRun) -> None:
        try:
            await self._api.append_entry(
                "subagents:record",
                {
                    "id": run.agent_id,
                    "type": run.agent_type,
                    "description": run.description,
                    "status": run.status,
                    "result": run.result_text[:RECORD_RESULT_CHARS],
                    "error": run.error,
                    "turns": run.turns,
                    "tool_calls": run.tool_calls,
                    "total_tokens": lifetime_tokens(run) if run.has_usage else None,
                },
            )
        except Exception:  # noqa: BLE001 - record persistence is best-effort
            pass

    def _handle_background_completion(self, run: AgentRun) -> None:
        if run.status == "cancelled":
            return
        if run.join_mode in ("smart", "group"):
            if any(pending is run for pending in self._batch):
                return  # the batch finalizer will route this completion
            if self._get_group_join().on_agent_complete(run) != "pass":
                return
        self._deliver_background_result(run)

    async def _execute(self, run: AgentRun, definition: AgentDefinition) -> None:
        settings = self._get_settings()
        cwd = self._api.context.cwd
        # Create the provider before the first await so concurrent spawns
        # claim providers in spawn order (tests script provider sequences).
        # See resolve_selection for the precedence rules.
        if definition.fork:
            capture = run.fork_capture
            assert capture is not None
            # A fork resolves against the snapshot taken at tool-call time,
            # not the live parent: a queued fork must run the model the
            # parent had when it forked (a /model switch in between would
            # otherwise break the shared prompt cache).
            selection = resolve_provider_selection(
                load_provider_settings(),
                provider_name=capture.provider_name,
                model=capture.model,
            )
        elif run.selection is not None:
            # Checked by the agent tool before it reported the spawn.
            selection = run.selection
        else:
            selection = self.resolve_selection(
                definition, provider=run.requested_provider, model=run.requested_model
            )
        provider = create_model_provider(
            selection.provider,
            model=selection.model,
            # Forks run the parent's live thinking level (captured at
            # tool-call time). Any other level would put a different thinking
            # config in the fork's request and cost (part of) the shared
            # prompt cache. When Tau does not expose the level (pre-0.4.0)
            # the capture is None and the provider's persisted per-model
            # level applies instead.
            thinking_level=(
                capture.thinking_level
                if definition.fork
                else definition.thinking
                or run.requested_thinking
                or DEFAULT_THINKING_LEVEL
            ),
        )
        run.provider = provider
        run.model = selection.model
        # A viewer opened on the queued run painted its header without a
        # model; push so it picks the resolved one up immediately.
        self._notify_run(run)
        child_cwd = cwd
        if (definition.isolation or run.requested_isolation) == "worktree":
            worktree = await create_worktree(cwd, run.agent_id)
            if worktree is None:
                raise RuntimeError(WORKTREE_ERROR_MESSAGE)
            run.worktree = worktree
            run.used_worktree = True
            self._worktree_repos.add(str(worktree.repo))
            child_cwd = worktree.work_path
        if run.output_writer is not None:
            await run.output_writer.write_initial(run.prompt)
        storage = _MemoryStorage()
        extra_config: dict[str, object] = self._inherited_trust_config()
        memory_block: str | None = None
        memory_rw = False
        role: str | None = None
        if definition.fork:
            # Fork: seed the child storage with the parent transcript and keep
            # the parent system prompt byte-identical (shared prompt cache).
            # No memory, skill preloads, or tool gating — the fork is the
            # parent session continued. Its instruction to do the task itself
            # is in the task message (wrap_fork_task).
            assert capture is not None
            storage.entries = build_fork_entries(capture, child_cwd)
            prompt_text: str | None = capture.system_prompt
            append_active = True
            # The snapshot pins a full transcript copy; the seed has it now.
            run.fork_capture = None
        else:
            skill_blocks = resolve_skill_blocks(
                definition.skills if isinstance(definition.skills, tuple) else None,
                cwd,
            )
            if definition.memory is not None:
                memory_rw = definition.tools is None or bool(
                    {"write", "edit"} & set(definition.tools)
                )
                memory_block = await prepare_memory(
                    definition.name, definition.memory, cwd, read_write=memory_rw
                )
            parent_prompt: str | None = None
            if definition.prompt_mode == "append":
                try:
                    parent_prompt = self._api.context.system_prompt
                except Exception:  # noqa: BLE001 - fall back to replace-mode assembly
                    parent_prompt = None
            environment = ""
            if definition.prompt_mode == "append" and parent_prompt:
                environment = await detect_environment(child_cwd)
            can_spawn = (
                not run.requested_isolated
                and (definition.tools is None or "agent" in definition.tools)
                and self.nesting.depth + 1 < self.max_depth
            )
            # A child that loads this extension has `agent` in its prompt's
            # tool list. Where the tool is then taken away, later Tau
            # rebuilds the prompt (below), except an append-mode one, which
            # is the parent's.
            prompt_rebuilt = hasattr(
                CodingSession, "refresh_system_prompt"
            ) and not (definition.prompt_mode == "append" and parent_prompt)
            role = sub_agent_role(
                can_spawn=can_spawn,
                agent_listed=not run.requested_isolated and not prompt_rebuilt,
            )
            prompt_text = build_child_system_prompt(
                definition,
                parent_prompt=parent_prompt,
                environment=environment,
                skill_blocks=skill_blocks,
                memory_block=memory_block,
                role=role,
            )
            append_active = definition.prompt_mode == "append" and bool(parent_prompt)
            # skills: none/false disables skill discovery; a named CSV also
            # disables it so preloaded bodies aren't double-listed in the index
            # (pi sets noSkills for both).
            if definition.skills is False or isinstance(definition.skills, tuple):
                extra_config["skills_enabled"] = False
        session = await CodingSession.load(
            CodingSessionConfig(
                provider=provider,
                model=selection.model,
                cwd=child_cwd,
                storage=storage,
                system=prompt_text if append_active else None,
                custom_system_prompt=None if append_active else prompt_text,
                # Tau adds this after the agent's own prompt, or after its
                # default coding prompt when the agent type has none. An
                # append-mode prompt is a full override with the role in it.
                append_system_prompt=None if append_active else role,
                # Children always discover skills natively (Tau defaults
                # resource paths from the session cwd). skills: true pins
                # discovery to the PARENT cwd, which matters under worktree
                # isolation where the default would use the worktree copy.
                resource_paths=(
                    inherited_resource_paths(cwd)
                    if definition.skills is True
                    else None
                ),
                provider_name=selection.provider.name,
                # Long unattended tasks outgrow any window, so children compact
                # like the parent: between turns once past the threshold.
                auto_compact_token_threshold=_child_compact_threshold(selection),
                # Children load extensions natively (pi parity), so subagents
                # can spawn subagents and a fork's tool pool matches the
                # parent's (shared prompt cache). `isolated: true` opts a
                # child out, restoring a core-tools-only session.
                extensions_enabled=not run.requested_isolated,
                **extra_config,
            )
        )
        # The child needs its provider config to build the thinking-off twin
        # tau sends an output-limit nudge with; without it every nudge went
        # out with thinking on. Set after load, not passed as
        # CodingSessionConfig.runtime_provider_config: tau rebuilds a provider
        # it does not own from that config at load, at the session's thinking
        # level, which would replace the one created above at the agent's.
        session._runtime_provider_config = selection.provider  # noqa: SLF001
        run.session = session
        for message in run.pending_steers:
            session.queue_steering_message(message)
        run.pending_steers.clear()
        run.max_turns = _effective_max_turns(
            definition.max_turns, run.requested_max_turns, settings.default_max_turns
        )
        run.grace_turns = settings.grace_turns
        if definition.tools is not None:
            allowed = set(definition.tools)
            if memory_block is not None:
                allowed |= {"read", "write", "edit"} if memory_rw else {"read"}
            session._harness.config.tools = [  # noqa: SLF001 - scoped tool gating
                tool for tool in session._harness.config.tools if tool.name in allowed
            ]
        if not definition.fork and self.nesting.depth + 1 >= self.max_depth:
            # The backstop: a child at the nesting limit cannot spawn. A fork
            # keeps its tools, which must match the parent's, and has the
            # call refused instead (see run_agent_tool).
            session._harness.config.tools = [  # noqa: SLF001 - scoped tool gating
                tool
                for tool in session._harness.config.tools
                if tool.name not in SPAWN_TOOLS
            ]
        # Tau listed the tools in the system prompt during the load, before
        # any were removed above. Later Tau can rebuild the prompt.
        refresh = getattr(session, "refresh_system_prompt", None)
        if refresh is not None and not definition.fork:
            refresh()
        final_text: list[str] = []
        async for event in session.prompt(run.prompt):
            self._observe(run, event, final_text)
            if event.type == "turn_end":
                self._enforce_turn_limit(run)
                self._request_report_if_empty(run, event)
                run.context_tokens = session.context_token_estimate
            await self._flush_output(run, session, event)
        run.result_text = final_text[-1] if final_text else ""
        await self._condense_report(run, session)
        run.context_tokens = session.context_token_estimate
        self._finalize_status(run)

    async def _condense_report(self, run: AgentRun, session: CodingSession) -> None:
        """Give a child whose final report is very long one turn to shorten it.

        Called when the run's event stream has ended and before its status is
        decided. Only a run that would be plain "completed" is asked: a
        stopped, failed or out-of-turns run has no finished report to
        condense. The turn is for one answer and nothing else: it is
        cancelled if the child calls a tool or goes on to a second turn, it
        is not counted in the run's turns, and the run takes no steering
        while it lasts. An answer of at least `CONDENSE_MIN_CHARS` and under
        `CONDENSE_OVER_CHARS` becomes the result, the original is saved as
        text beside the output file, and the result names that copy.
        Otherwise the run keeps the original, as if it had not been asked.
        """
        original = run.result_text
        if (
            len(original) <= CONDENSE_OVER_CHARS
            or run.aborted
            or run.pending_error is not None
            or run.report_missing
            or run.soft_limit_reached
        ):
            return
        turns = run.turns
        answers: list[str] = []
        strayed = False
        run.condensing = True
        try:
            async for event in session.prompt(CONDENSE_REQUEST_MESSAGE):
                self._observe(run, event, answers)
                await self._flush_output(run, session, event)
                if not strayed and (
                    event.type == "tool_execution_start" or run.turns > turns + 1
                ):
                    # Not an answer but more work: "Saved to report.md" would
                    # otherwise be taken for the report.
                    strayed = True
                    session.cancel()
        except Exception:  # noqa: BLE001 - the original report still stands
            strayed = True
        finally:
            run.condensing = False
            run.turns = turns
        condensed = answers[-1] if answers else ""
        usable = (
            not strayed
            and CONDENSE_MIN_CHARS <= len(condensed) < CONDENSE_OVER_CHARS
            and not run.aborted
            and run.pending_error is None
            and not run.report_missing
        )
        # Whatever the extra turn did, the run finished when its report did.
        run.aborted = False
        run.pending_error = None
        run.report_missing = False
        run.tools_in_flight.clear()
        run.current_tool = None
        run.current_tool_started_at = None
        if not usable:
            run.last_text = original
            return
        where = ""
        if run.output_file:
            # Plain text, which `read` can page; in the transcript the report
            # is one long JSON-escaped line.
            copy = full_report_path(Path(run.output_file))
            try:
                await asyncio.to_thread(copy.write_text, original, encoding="utf-8")
                where = f" The full report is in {copy}."
            except OSError:
                where = f" The full report is in the transcript, {run.output_file}."
        run.result_text = (
            f"[The agent condensed this report from {len(original):,} to"
            f" {len(condensed):,} characters.{where}]\n\n{condensed}"
        )

    async def _flush_output(
        self, run: AgentRun, session: CodingSession, event: AgentEvent
    ) -> None:
        """Append new messages to the output file at a tool start or turn end.

        Tau adds an assistant message to the session only after its
        message_end, and runs its tool calls after that, so a tool start is
        the first point at which the call it makes can be written.
        """
        if run.output_writer is None:
            return
        if event.type in ("tool_execution_start", "turn_end"):
            await run.output_writer.flush(session.messages)
        elif event.type == "compaction_end" and not getattr(event, "aborted", False):
            await run.output_writer.compacted(session.messages)

    def _finalize_status(self, run: AgentRun) -> None:
        """Decide a run's terminal status once its event stream is exhausted.

        Never decided mid-stream (pi semantics): a message_end with
        stop_reason "error" may be recovered by overflow compaction +
        auto-retry, so only the error still pending at stream end counts.
        """
        if run.aborted:
            run.status = "aborted"
        elif run.pending_error is not None:
            run.status = "error"
            run.error = run.pending_error
        elif run.status == "running":
            if run.report_missing:
                # Asked once and still empty: result_text is already the
                # last non-empty message, the best report there is.
                run.status = "completed_no_report"
            elif run.soft_limit_reached:
                run.status = "turn_limit"
            else:
                run.status = "completed"

    def _request_report_if_empty(self, run: AgentRun, event: AgentEvent) -> None:
        """At a turn that ends the run with an empty message, ask once for a report.

        The steering message is drained right after turn_end, so the child
        gets one more turn instead of the run ending with nothing to return.
        """
        if (
            event.type != "turn_end"
            or not run.report_missing
            or run.report_requested
            or run.aborted
            or run.pending_error is not None
            or run.session is None
        ):
            return
        if getattr(event, "tool_results", None):
            return
        run.report_requested = True
        run.session.queue_steering_message(REPORT_REQUEST_MESSAGE)

    def _enforce_turn_limit(self, run: AgentRun) -> None:
        if run.max_turns is None or run.session is None:
            return
        if not run.soft_limit_reached and run.turns >= run.max_turns:
            run.soft_limit_reached = True
            run.session.queue_steering_message(
                soft_limit_message(run.max_turns, run.grace_turns)
            )
        elif (
            run.soft_limit_reached
            and not run.aborted
            and run.turns >= run.max_turns + run.grace_turns
        ):
            run.aborted = True
            run.session.cancel()

    def _observe(self, run: AgentRun, event: AgentEvent, final_text: list[str]) -> None:
        run.revision += 1
        now = time.monotonic()
        run.last_event_at = now
        if event.type == "turn_end":
            run.turns += 1
        elif event.type == "tool_execution_start":
            run.tool_calls += 1
            run.tools_in_flight[event.tool_call_id] = (event.tool_name, now)
            run.recent_tools.append(
                ToolCallRecord(
                    call_id=event.tool_call_id,
                    name=event.tool_name,
                    args=summarize_args(event.args),
                    started_at=now,
                )
            )
            run.current_tool = event.tool_name
            run.current_tool_started_at = now
            _set_phase(run, PHASE_IN_TOOL, now)
        elif event.type == "tool_execution_end":
            run.tools_in_flight.pop(event.tool_call_id, None)
            for record in reversed(run.recent_tools):
                if record.call_id == event.tool_call_id:
                    record.ended_at = now
                    record.is_error = bool(event.is_error)
                    break
            if run.tools_in_flight:
                # Parallel calls: report the latest one still running.
                name, started = list(run.tools_in_flight.values())[-1]
                run.current_tool = name
                run.current_tool_started_at = started
            else:
                run.current_tool = None
                run.current_tool_started_at = None
                _set_phase(run, PHASE_AWAITING_MODEL, now)
        elif event.type == "message_start":
            if getattr(event.message, "role", None) == "assistant":
                _set_phase(run, PHASE_STREAMING, now)
        elif event.type == "message_update":
            delta = getattr(event.assistant_message_event, "delta", None)
            if isinstance(delta, str):
                run.streamed_chars += len(delta)
        elif event.type == "message_end":
            message = event.message
            if getattr(message, "role", None) == "assistant":
                if not run.tools_in_flight:
                    _set_phase(run, PHASE_AWAITING_MODEL, now)
                # Real billed usage. Lifetime sum of input + output +
                # cache_write per response; cache reads are excluded because
                # each turn re-reads the whole cached prefix, so summing them
                # counts the prefix N times (pi issue #38).
                usage = message.usage
                run.tokens_input += usage.input
                run.tokens_output += usage.output
                run.tokens_cache_write += usage.cache_write
                # Terminal errors ride the assistant message now (pi protocol:
                # there is no separate error event). Tentative until the stream
                # ends: an overflow error can be compacted away and retried, in
                # which case the next assistant message clears it.
                if message.stop_reason == "error":
                    run.pending_error = (
                        message.error_message or "subagent provider error"
                    )
                else:
                    run.pending_error = None
                text = message.text.strip()
                if text:
                    final_text.append(text)
                    run.last_text = text
                run.report_missing = (
                    not text
                    and not message.tool_calls
                    and message.stop_reason not in ("error", "aborted")
                )
        self._emit_progress(run, event)
        if event.type in RUN_PUSH_EVENTS:
            self._notify_run(run)

    def _emit_progress(self, run: AgentRun, event: AgentEvent) -> None:
        """Update the live stats ticker under the tool row (foreground runs).

        One stable, cumulative line in the completion card's stats vocabulary
        (turns · tool uses · tokens), so the running row morphs into the
        finished card — unlike the per-event activity echo this replaces
        (dropped as noise in d0d5ac8), the line only changes when a stat does.
        """
        if run.on_update is None:
            return
        if event.type not in ("turn_end", "tool_execution_start", "message_end"):
            return
        message = format_live_stats(run)
        if not message or message == run.last_progress:
            return
        run.last_progress = message
        data: dict[str, object] = {
            "agent_id": run.agent_id,
            "turns": run.turns,
            "tool_uses": run.tool_calls,
        }
        if run.has_usage:
            data["total_tokens"] = lifetime_tokens(run)
        with contextlib.suppress(Exception):
            run.on_update(AgentToolResult(content=message, details=data))

    def _deliver_background_result(self, run: AgentRun) -> None:
        if run.result_consumed:
            return
        self._schedule_nudge(
            run.agent_id,
            lambda deliver_as: self._send_individual_nudge(run, deliver_as),
            announce=lambda: self._announce_result(run),
        )

    def _deliver_group_notification(self, records: list[AgentRun], partial: bool) -> None:
        key = "group:" + ",".join(run.agent_id for run in records)
        self._schedule_nudge(
            key,
            lambda deliver_as: self._send_group_notification(
                records, partial, deliver_as
            ),
        )

    def _schedule_nudge(
        self,
        key: str,
        send: Callable[[DeliverAs], None],
        *,
        announce: Callable[[], None] | None = None,
    ) -> None:
        """Hold a notification briefly so a prompt result read can cancel it.

        When the hold ends the user gets the toast at once. The parent model
        gets the notice at once if it is idle (starting a turn); if it is mid-run
        the notice waits in the outbox for the parent's next turn boundary and
        goes in as steering, so the parent sees it before its run ends instead
        of after (which a follow-up would mean).
        """
        existing = self._nudge_timers.pop(key, None)
        if existing is not None:
            existing.cancel()

        def fire() -> None:
            self._nudge_timers.pop(key, None)
            if announce is not None:
                announce()
            if self._parent_is_running():
                self._outbox[key] = send
            else:
                send("follow_up")

        self._nudge_timers[key] = asyncio.get_running_loop().call_later(
            NUDGE_HOLD_SECONDS, fire
        )

    def _parent_is_running(self) -> bool:
        try:
            return bool(self._api.context.is_running)
        except Exception:  # noqa: BLE001 - no bound session reads as idle
            return False

    def flush_outbox(self, deliver_as: DeliverAs) -> None:
        """Send held notices; each skips runs whose result was read meanwhile.

        Called from the parent's turn_end (as steering, drained at that same
        boundary) and agent_settled (the parent is idle, so the send starts a
        turn).
        """
        held = self._outbox
        self._outbox = {}
        for send in held.values():
            send(deliver_as)

    def cancel_nudge(self, agent_id: str) -> None:
        """Drop a pending individual nudge for a consumed run."""
        handle = self._nudge_timers.pop(agent_id, None)
        if handle is not None:
            handle.cancel()
        self._outbox.pop(agent_id, None)

    def notify_parent(self, key: str, text: str) -> None:
        """Tell the parent session about something that is not a run notice.

        The one notice path, for jobs today and for anything else tomorrow:
        the toast goes to the terminal at once, and `text` reaches the model
        as steering at its next turn boundary if it is mid-run, or as a
        follow-up turn if it is idle. `key` dedups a notice the way a run id
        does, so a second one for the same key replaces the first.
        """
        self._schedule_nudge(
            key,
            lambda deliver_as: self._send_plain_notice(text, deliver_as),
            announce=lambda: self._announce_plain(text),
        )

    def _announce_plain(self, text: str) -> None:
        """Show a notice's text in the terminal; a dead UI is not our problem."""
        with contextlib.suppress(Exception):
            self._api.notify(text, "info")

    def _send_plain_notice(self, text: str, deliver_as: DeliverAs) -> None:
        """Put a notice's text into the parent's context as a plain message.

        No `custom_type`, so no renderer claims it and there is nothing to
        register: the text is what the model reads, which is all a job notice
        is for. A run notice keeps its card; this one keeps its sentence.
        """
        try:
            self._api.send_user_message(text, deliver_as=deliver_as)
        except Exception:  # noqa: BLE001 - timer callbacks must never crash the loop
            pass

    def _announce_result(self, run: AgentRun) -> None:
        if run.result_consumed:
            return
        with contextlib.suppress(Exception):
            self._api.notify(
                f"Subagent {run.agent_id} ({run.description}) {run.status}"
                f" (turns={format_turns(run)}).",
                "info" if run.status in SUCCESS_STATUSES else "warning",
            )

    def _send_individual_nudge(self, run: AgentRun, deliver_as: DeliverAs) -> None:
        if run.result_consumed:
            return
        try:
            self._deliver_notification(
                f"{format_task_notification(run)}\n{COMPLETION_NOTICE}",
                build_notification_details(run),
                deliver_as,
            )
        except Exception:  # noqa: BLE001 - timer callbacks must never crash the loop
            pass

    def _send_group_notification(
        self, records: list[AgentRun], partial: bool, deliver_as: DeliverAs
    ) -> None:
        unconsumed = [run for run in records if not run.result_consumed]
        if not unconsumed:
            return
        try:
            details = build_notification_details(unconsumed[0], GROUP_RESULT_CHARS)
            details["others"] = [
                build_notification_details(run, GROUP_RESULT_CHARS)
                for run in unconsumed[1:]
            ]
            self._deliver_notification(
                format_group_notification(unconsumed, partial=partial),
                details,
                deliver_as,
            )
        except Exception:  # noqa: BLE001 - timer callbacks must never crash the loop
            pass

    def _deliver_notification(
        self, content: str, details: dict[str, object], deliver_as: DeliverAs
    ) -> None:
        """Send as a custom message so the registered renderer draws the card."""
        self._api.send_custom_message(
            content,
            custom_type="subagent-notification",
            details=details,
            deliver_as=deliver_as,
        )


def _effective_max_turns(
    frontmatter: int | None, param: int | None, default: int | None
) -> int | None:
    """Resolve the turn limit: frontmatter wins over param wins over default."""
    if frontmatter is not None:
        raw = frontmatter
    elif param is not None:
        raw = param
    else:
        raw = default
    if raw is None or raw <= 0:
        return None
    return max(1, raw)


def _provider_is_usable(provider: object) -> bool:
    """Whether Tau has credentials for a provider (stored, OAuth, or env).

    Fails open: when the check itself breaks, the run starts and a missing
    credential surfaces as a run error with a stub transcript instead.
    """
    try:
        return provider_has_usable_credentials(
            provider,  # type: ignore[arg-type]
            credential_reader=FileCredentialStore(),
        )
    except Exception:  # noqa: BLE001 - never block a spawn on the check itself
        return True


def authenticated_provider_names() -> list[str]:
    """Names of the configured providers that have usable credentials."""
    try:
        providers = load_provider_settings().providers
    except Exception:  # noqa: BLE001 - best-effort listing
        return []
    return [
        provider.name for provider in providers if _provider_is_usable(provider)
    ]


def _authenticated_providers_text() -> str:
    names = authenticated_provider_names()
    if not names:
        return "No provider has credentials configured."
    return f"Providers with credentials: {', '.join(names)}."


def format_background_spawn(run: AgentRun, max_concurrent: int) -> str:
    """Format the tool result for a background spawn (started or queued)."""
    started = run.status != "queued"
    lines = [
        "Agent started in background." if started else "Agent queued in background.",
        f"Agent ID: {run.agent_id}",
        f"Type: {run.agent_type}",
        f"Description: {run.description}",
    ]
    if run.output_file:
        lines.append(f"Output file: {run.output_file}")
    if not started:
        lines.append(f"Position: queued (max {max_concurrent} concurrent)")
    lines.extend(
        [
            "",
            "You will be notified when this agent completes.",
            "Use get_subagent_result to retrieve full results, steer_subagent"
            " to send it messages, or stop_subagent to end it.",
            "Do not duplicate this agent's work.",
        ]
    )
    return "\n".join(lines)


COMPLETION_NOTICE = (
    "This is an automated subagent completion notice, not a user message. "
    "Use the result to continue the original task."
)


def format_task_notification(
    run: AgentRun, max_result_chars: int = INDIVIDUAL_RESULT_CHARS
) -> str:
    """Format one run's completion as a minimal <task-notification> block.

    Deliberately minimal: usage stats are omitted (the get_subagent_result
    header repeats them, and the TUI card renders them from `details`,
    outside the model's context). Turns are included as used/max, so the
    parent can tell a run that hit its limit from one that finished. The
    result is fit-or-fetch:
    included whole when it fits `max_result_chars`, otherwise replaced by a
    pointer to get_subagent_result — never truncated.
    """
    body = run.error if run.status == "error" else run.result_text
    body = body or "(no output)"
    if len(body) <= max_result_chars:
        result_line = f"<result>{body}</result>\n"
    else:
        result_line = (
            f"<result-pending>Result is {len(body)} chars. Call "
            f'get_subagent_result("{run.agent_id}") to read it before acting.'
            "</result-pending>\n"
        )
    output_line = (
        f"<output-file>{run.output_file}</output-file>\n" if run.output_file else ""
    )
    if run.status == "completed_no_report":
        result_line = f"<note>{no_report_note(run)}</note>\n{result_line}"
    return (
        "<task-notification>\n"
        f"<agent-id>{run.agent_id}</agent-id>\n"
        f"<description>{run.description}</description>\n"
        f"{output_line}"
        f"<status>{run.status}</status>\n"
        f"<turns>{format_turns(run)}</turns>\n"
        f"{result_line}"
        "</task-notification>"
    )


def no_report_note(run: AgentRun) -> str:
    """Why a completed_no_report result may not be the answer, and where to look."""
    where = f" at {run.output_file}" if run.output_file else ""
    return (
        "The agent ended without a final report, even after being asked once."
        " The result is its last non-empty message, which may be a"
        f" mid-task remark rather than an answer; read its transcript{where}"
        " before relying on it."
    )


def format_turns(run: AgentRun) -> str:
    """Turns used, with the limit when there is one: "5/5" or "3"."""
    if run.max_turns is None:
        return str(run.turns)
    return f"{run.turns}/{run.max_turns}"


def build_notification_details(
    run: AgentRun, result_max_chars: int = INDIVIDUAL_RESULT_CHARS
) -> dict[str, object]:
    """Structured details for the custom renderer (pi's buildNotificationDetails)."""
    body = run.error if run.status == "error" else run.result_text
    preview = body or "No output."
    if len(preview) > result_max_chars:
        preview = preview[:result_max_chars] + "…"
    return {
        "description": run.description,
        "status": run.status,
        "turn_count": run.turns,
        "max_turns": run.max_turns,
        "tool_uses": run.tool_calls,
        "total_tokens": lifetime_tokens(run) if run.has_usage else 0,
        "duration_ms": _duration_ms(run) or 0,
        "output_file": run.output_file,
        "error": run.error,
        "result_preview": preview,
    }


def format_group_notification(records: list[AgentRun], *, partial: bool) -> str:
    """Format a consolidated completion notice for a group of runs."""
    label = f"{len(records)} agent(s) finished"
    if partial:
        label += " (partial — others still running)"
    blocks = "\n\n".join(
        format_task_notification(run, GROUP_RESULT_CHARS) for run in records
    )
    return (
        f"Background agent group completed: {label}\n\n"
        f"{blocks}\n\n"
        "Use get_subagent_result for full output."
    )


#: Statuses of a run cut off before its report; their result is a partial.
PARTIAL_STATUSES = ("aborted", "cancelled")
#: Tool calls listed in a partial result.
PARTIAL_TOOL_TAIL = 5


def format_partial_result(run: AgentRun) -> str:
    """What a stopped or cancelled run got done: its last text and tool tail."""
    ended = run.completed_at if run.completed_at is not None else time.monotonic()
    lines = [
        f"Partial result: the run was {run.status} before its final report.",
        "",
        "Last message:",
        run.last_text or "(none)",
    ]
    if run.recent_tools:
        lines += [
            "",
            format_tool_log(
                run.recent_tools, PARTIAL_TOOL_TAIL, run.tool_calls, ended, live=False
            ),
        ]
    if run.output_file:
        lines += ["", f"Transcript: {run.output_file}"]
    return "\n".join(lines)


def format_run_summary(run: AgentRun) -> str:
    """Format one run for /agents and get_subagent_result output."""
    return (
        f"{run.agent_id} [{run.status}] type={run.agent_type}"
        f" turns={run.turns} tools={run.tool_calls} — {run.description}"
    )


#: Seconds without an event after which get_subagent_result says a running
#: agent may be wedged and names stop_subagent. Long enough that a test suite
#: or a slow build does not trip it on its own.
QUIET_WARNING_SECONDS = 300


def format_liveness(run: AgentRun, now: float | None = None) -> str:
    """How long a live run has gone, what it is doing, and its last sign of life."""
    now = time.monotonic() if now is None else now
    if run.started_at is None:
        return "Not started yet."
    parts = [f"Running for {format_seconds(now - run.started_at)}"]
    elapsed = phase_elapsed(run, now)
    elapsed_text = format_seconds(elapsed) if elapsed is not None else "?"
    if run.phase == PHASE_IN_TOOL and run.current_tool is not None:
        parts.append(f"inside `{run.current_tool}` for {elapsed_text}")
    elif run.phase == PHASE_STREAMING:
        parts.append(f"streaming a response for {elapsed_text}")
    else:
        parts.append(f"awaiting the model for {elapsed_text}")
    if run.last_event_at is None:
        parts.append("no events yet")
    else:
        parts.append(f"last event {format_seconds(now - run.last_event_at)} ago")
    tokens = tokens_so_far(run)
    if tokens:
        parts.append(f"~{tokens} tokens so far")
    line = "; ".join(parts) + "."
    quiet_since = run.last_event_at if run.last_event_at is not None else run.started_at
    if now - quiet_since >= QUIET_WARNING_SECONDS:
        line += (
            " It may be wedged: a steering message waits for the current tool to"
            " return, and stop_subagent ends the run."
        )
    return line


def tokens_so_far(run: AgentRun) -> int:
    """Billed tokens so far plus an estimate of the response streaming now."""
    return lifetime_tokens(run) + streamed_tokens(run)


def build_liveness_details(run: AgentRun, now: float | None = None) -> dict[str, object]:
    """Structured liveness for get_subagent_result on a live run."""
    now = time.monotonic() if now is None else now
    elapsed = phase_elapsed(run, now) if run.status == "running" else None
    return {
        "agent_id": run.agent_id,
        "status": run.status,
        "phase": run.phase if run.status == "running" else run.status,
        "phase_elapsed_s": round(elapsed, 1) if elapsed is not None else None,
        "current_tool": run.current_tool,
        "running_s": (
            round(now - run.started_at, 1) if run.started_at is not None else None
        ),
        "last_event_s": (
            round(now - run.last_event_at, 1) if run.last_event_at is not None else None
        ),
        "tokens_so_far": tokens_so_far(run),
        "turns": run.turns,
        "max_turns": run.max_turns,
        "tool_uses": run.tool_calls,
        "output_file": run.output_file,
    }


def _set_phase(run: AgentRun, phase: str, now: float) -> None:
    run.phase = phase
    run.phase_since = now
    if phase == PHASE_STREAMING:
        run.streamed_chars = 0


def lifetime_tokens(run: AgentRun) -> int:
    """Lifetime billed tokens: input + output + cache_write (pi semantics)."""
    return run.tokens_input + run.tokens_output + run.tokens_cache_write


def format_live_stats(run: AgentRun) -> str:
    """Cumulative running-stats line in the completion card's vocabulary."""
    return " · ".join(
        stat_parts(
            {
                "turn_count": run.turns,
                "max_turns": run.max_turns,
                "tool_uses": run.tool_calls,
                "total_tokens": lifetime_tokens(run) if run.has_usage else 0,
            }
        )
    )


def format_usage_parts(run: AgentRun) -> str:
    """Format pi-style usage parts joined by middle dots."""
    parts = []
    if run.has_usage:
        parts.append(f"{lifetime_tokens(run)} tokens")
    parts.append(f"{run.tool_calls} tool uses")
    if run.context_tokens:
        parts.append(f"~{run.context_tokens} context tokens")
    duration = _duration_ms(run)
    if duration is not None:
        parts.append(f"{duration / 1000:.1f}s")
    return " · ".join(parts)


def _duration_ms(run: AgentRun) -> int | None:
    if run.started_at is None or run.completed_at is None:
        return None
    return max(0, int((run.completed_at - run.started_at) * 1000))


STEER_CALL_PREVIEW_CHARS = 60


def render_agent_call(arguments: Mapping[str, object]) -> str:
    """Friendly invocation line for the agent tool (pi's renderCall port)."""
    agent_type = str(arguments.get("subagent_type") or "general")
    description = str(arguments.get("description") or "").strip()
    line = f"▸ {agent_type} agent"
    schedule = arguments.get("schedule")
    if schedule:
        line += f" (scheduled {schedule})"
    return f"{line} · {description}" if description else line


def render_get_result_call(arguments: Mapping[str, object]) -> str:
    """Friendly invocation line for get_subagent_result."""
    line = f"▸ get result · {arguments.get('agent_id') or '?'}"
    if arguments.get("wait"):
        line += " (wait)"
    return line


def render_stop_call(arguments: Mapping[str, object]) -> str:
    """Friendly invocation line for stop_subagent."""
    return f"▸ stop {arguments.get('agent_id') or '?'}"


def render_steer_call(arguments: Mapping[str, object]) -> str:
    """Friendly invocation line for steer_subagent."""
    message = " ".join(str(arguments.get("message") or "").split())
    if len(message) > STEER_CALL_PREVIEW_CHARS:
        message = message[:STEER_CALL_PREVIEW_CHARS].rstrip() + "…"
    line = f"▸ steer {arguments.get('agent_id') or '?'}"
    return f"{line} · {message}" if message else line


def setup(tau: ExtensionAPI) -> None:
    """Register subagent tools, the /agents command, and shutdown cleanup."""
    manager = SubagentManager(tau)
    # Notifications render as pi-style cards instead of raw
    # <task-notification> XML bubbles.
    tau.register_message_renderer("subagent-notification", render_notification)
    tau.register_message_renderer("subagent-stall", render_stall_notice)
    scheduler = SubagentScheduler(manager)
    # The registry and its one poller, built at `session_start` rather than
    # here: the store's root comes from `TAU_JOBS_DIR`, which is only what it
    # is by then, and a watcher that outlived its session would notify a
    # parent that had gone. One of each, held by closure like the scheduler.
    # Named for what it holds and not `jobs`: the tool, the command and this
    # slot all speak about jobs, and only one of them is the running watcher.
    job_registry: dict[str, tuple[JobStore, JobWatcher]] = {}

    def on_job_change(started: list[Job], finished: list[Job]) -> None:
        # A job that appeared is a row to draw, not news: the parent never
        # asked to be told about starts, and the strip shows them anyway.
        del started
        for job in finished:
            manager.notify_parent(f"job:{job.id}", format_job_notice(job))

    async def start_watcher(retention_days: int) -> None:
        # Called from `session_start`, so a start that arrives without a
        # shutdown (a host force-clear) rebuilds rather than doubling up.
        stop_watcher()
        # `JobStore()` takes `~/.tau/jobs`, or `TAU_JOBS_DIR` if set, so the
        # watcher reads the registry the job command writes. The sweep and the
        # seed each read every job directory, so both go off the loop; the
        # sweep is its own guard, leaving a directory it cannot read alone.
        store = JobStore()
        if retention_days > 0:
            await asyncio.to_thread(store.sweep, retention_days=retention_days)
        watcher = JobWatcher(store, interval=JOBS_POLL_SECONDS)
        watcher.subscribe(on_job_change)
        job_registry["active"] = (store, watcher)
        # Seeded first, armed last: a tick before the seed would report every
        # already-running job as one that had just appeared.
        with contextlib.suppress(Exception):
            await asyncio.to_thread(watcher.poll_once)
        watcher.start()  # a no-op with no loop: print mode stays quiet

    def stop_watcher() -> None:
        entry = job_registry.pop("active", None)
        if entry is None:
            return
        # `stop()` cancels and returns; only `stop_async` awaits, which a sync
        # handler cannot. Dropping the entry first is what leaves a tick
        # already in flight with nothing of ours to notify.
        entry[1].stop()

    def start_scheduler() -> None:
        # Session-scoped: the store is built once the session id is available
        # (mirrors pi, which constructs it in session_start). Scheduling is
        # non-essential — swallow failures so an unwritable .tau/ can't break
        # the rest of the extension.
        try:
            session_id = tau.context.session_id
            if not session_id:
                return  # id not yet available — retry on the next session_start
            store = ScheduleStore(resolve_store_path(tau.context.cwd, session_id))
            scheduler.start(store)
        except Exception:  # noqa: BLE001 - scheduling must never break the session
            pass

    # Component seam: the extension owns the agents strip and the
    # conversation viewer as Textual widgets mounted through the component
    # bridge. Installed lazily on session_start — NOT in setup() — because the
    # host attaches its UI bridge (making supports_components True) only after
    # CodingSession.load has already run setup() with the print-mode NullUiBridge.
    ui_controller: list[object] = []  # 0 or 1 element (a nonlocal-friendly cell)

    def _install_ui_components() -> None:
        # supports_components is a runtime gate, not feature detection: the
        # print-mode NullUiBridge reports False, and the extension then runs
        # dialog-only, no strip.
        components = tau.context.ui.components
        if not components.supports_components:
            return
        # Defensive reinstall (bug fix 2): a controller can survive into the next
        # bind if a session_start arrives without a matching shutdown (or the host
        # force-cleared its slots on rebind). Tear the stale one down and mount
        # fresh widgets against the host's current slots. The host sequences the
        # same-tick teardown+reinstall, so the strip's slot never collides on its
        # id (which used to raise DuplicateIds and drop the strip).
        existing = _current_controller()
        if existing is not None:
            existing.teardown()
            ui_controller.clear()
        from .ui import SubagentUiController

        controller = SubagentUiController(manager, components)
        controller.install()
        manager.sources_changed = controller.on_change
        ui_controller.append(controller)

    def _current_controller():  # noqa: ANN202 - SubagentUiController | None
        return ui_controller[0] if ui_controller else None

    async def on_session_start(event: SessionStartEvent, context) -> None:  # noqa: ANN001
        del event, context
        if not scheduler.is_active():
            start_scheduler()
        # The same age-out for finished jobs the transcripts get below: what
        # goes out is the retention setting, and the sweep is part of opening
        # the registry rather than a second pass over it.
        await start_watcher(manager._get_settings().jobs_retention_days)
        _install_ui_components()
        # ADR 0003: age out durable transcripts. Best-effort and off-loop so
        # an enormous or slow ~/.tau/subagents/ tree cannot delay startup.
        retention = manager._get_settings().transcript_retention_days
        if retention is not None and retention > 0:
            try:
                await asyncio.to_thread(sweep_transcripts, retention)
            except OSError:
                pass

    async def run_agent_tool(tool_call_id, arguments, signal=None, on_update=None):  # noqa: ANN001, ANN202
        del tool_call_id
        if manager.nesting.depth >= manager.max_depth:
            return _tool_result(content=depth_limit_message(manager.nesting.depth))
        await manager.evict_stale()
        if arguments.get("schedule"):
            return await run_schedule(arguments)
        resume_id = arguments.get("resume")
        if resume_id:
            return await run_resume(str(resume_id), arguments, on_update)

        definitions = manager.definitions()
        agent_type = str(arguments.get("subagent_type", "general"))
        definition = definitions.get(agent_type)
        if definition is None:
            available = ", ".join(sorted(definitions))
            return _tool_result(
                content=f"Unknown subagent_type: {agent_type}. Available: {available}",
            )
        prompt = str(arguments.get("prompt", "")).strip()
        if not prompt:
            return _tool_result(content="prompt is required")
        description = str(arguments.get("description", "")) or f"{agent_type} agent"
        background = bool(arguments.get("run_in_background", False))
        max_turns = _coerce_max_turns(arguments.get("max_turns"))
        isolation = (
            "worktree" if arguments.get("isolation") == "worktree" else None
        )
        isolated = bool(arguments.get("isolated", False))
        fork_capture = None
        if definition.fork:
            # Reject, never silently ignore: a fork always runs the parent's
            # model, thinking config, and tool pool (anything else breaks the
            # shared prompt cache). Dropping a `model` param quietly would let
            # the parent believe it spawned a cheaper model while the fork
            # actually runs the parent's — a silent cost surprise.
            rejected = [
                name
                for name in ("provider", "model", "thinking", "isolated")
                if arguments.get(name)
            ]
            if rejected:
                return _tool_result(
                    content=f"A fork cannot take {', '.join(rejected)} — it"
                    " always runs the parent's provider, model, thinking"
                    " config, and tool pool (a different one would break the"
                    " shared prompt cache). Omit the parameter(s), or spawn a"
                    " different agent type with inherit_context=true to use"
                    " another provider/model with the conversation context.",
                )
            # Captured at tool-call time (like inherit_context): queued forks
            # see the conversation as of this call. inherit_context stays
            # accepted — the fork fulfills exactly that request.
            fork_capture = capture_fork(tau.context)
            prompt = wrap_fork_task(prompt, worktree=isolation == "worktree")
        inherit = arguments.get("inherit_context")
        if inherit is None:
            inherit = definition.inherit_context
        if inherit and not definition.fork:
            # Captured at spawn time (pi semantics): the digest reflects the
            # parent conversation as of this tool call, even for queued runs.
            parent_context = build_parent_context(tau.context.transcript)
            if parent_context:
                prompt = parent_context + prompt
        provider = (
            str(arguments.get("provider")) if arguments.get("provider") else None
        )
        model = str(arguments.get("model")) if arguments.get("model") else None
        thinking = arguments.get("thinking")
        if thinking is not None:
            thinking = str(thinking)
            if thinking not in THINKING_LEVELS:
                return _tool_result(
                    content=f"Invalid thinking level: {thinking}."
                    f" Valid options: {', '.join(THINKING_LEVELS)}",
                )

        selection = None
        if not definition.fork:
            # Check before reporting "started": a bad provider, model, or
            # missing credential must fail this call, not a background run.
            selection, error = manager.check_spawn_target(
                definition, provider=provider, model=model
            )
            if error is not None:
                return _tool_result(content=error)

        run = manager.spawn(
            agent_type=definition,
            prompt=prompt,
            description=description,
            background=background,
            max_turns=max_turns,
            isolation=isolation,
            provider=provider,
            model=model,
            thinking=thinking,
            fork_capture=fork_capture,
            isolated=isolated,
            selection=selection,
        )
        if background:
            return _tool_result(
                content=format_background_spawn(run, manager.max_concurrent),
                # A "background" card: the row confirms the spawn in one dim
                # line; completion arrives later as a notification card.
                details={
                    "status": "background",
                    "agent_id": run.agent_id,
                    "queued": run.status == "queued",
                    "output_file": run.output_file,
                },
            )

        async def cancel_child() -> None:
            """Stop the child and settle its record (pi's parent-abort cascade)."""
            if run.session is not None:
                run.session.cancel()
            assert run.task is not None
            run.task.cancel()
            # asyncio.wait (not `await run.task`) so the child's CancelledError
            # is never raised here — a plain suppress would also swallow a
            # cancellation aimed at THIS coroutine landing at that await.
            await asyncio.wait({run.task})
            # Settle before the async close: a cancel that lands before the
            # task first runs never reaches _run_agent's CancelledError
            # handler, and a re-cancel during close_run must not skip this.
            if run.status not in TERMINAL_STATUSES:
                run.status = "cancelled"
            if run.completed_at is None:
                run.completed_at = time.monotonic()
            await manager.close_run(run)

        # Live stats ticker: only valid while this tool call is executing, so
        # foreground-only. The try/finally clears it even when tau hard-cancels
        # this coroutine mid-wait.
        run.on_update = on_update
        assert run.task is not None
        try:
            while not run.task.done():
                if signal is not None and signal.is_cancelled():
                    await cancel_child()
                    if run.status in SUCCESS_STATUSES:
                        # The run beat the cancel to the finish line — report
                        # the real result, not a phantom cancellation.
                        return _foreground_result(run)
                    return _tool_result(
                        content="Subagent cancelled."
                        + (f"\n\n{run.result_text}" if run.result_text else ""),
                        # Cancelled runs stay in the card family (∅ cancelled).
                        details=build_notification_details(
                            run, FOREGROUND_RESULT_CHARS
                        ),
                    )
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            # Esc in the TUI: tau hard-cancels this tool coroutine (the event
            # stream is torn down before the 50ms poll can see the signal), so
            # the cooperative branch above never runs. Take the child down
            # with us — otherwise it keeps running as a zombie and its result
            # is silently dropped. Background runs are untouched by design:
            # they never pass through this wait loop (pi wires the parent
            # abort signal on the foreground path only).
            await cancel_child()
            raise
        finally:
            run.on_update = None

        return _foreground_result(run)

    async def run_resume(agent_id: str, arguments, on_update=None) -> AgentToolResult:  # noqa: ANN001
        run = manager.runs.get(agent_id)
        if run is None:
            return _tool_result(
                content=f'Agent not found: "{agent_id}". It may have been cleaned up.',
            )
        if run.task is not None and not run.task.done():
            return _tool_result(
                content=f'Agent "{agent_id}" is still running.'
                " Use steer_subagent to redirect it.",
            )
        if run.used_worktree:
            return _tool_result(
                content=f'Agent "{agent_id}" ran in an isolated worktree that has'
                " been cleaned up; resume is not supported for worktree agents.",
            )
        if run.session is None:
            return _tool_result(
                content=f'Agent "{agent_id}" has no active session to resume.',
            )
        prompt = str(arguments.get("prompt", "")).strip()
        if not prompt:
            return _tool_result(content="prompt is required")
        run.on_update = on_update
        run.last_progress = ""
        try:
            await manager.resume(run, prompt)
        finally:
            run.on_update = None
        return _foreground_result(run)

    async def run_schedule(arguments) -> AgentToolResult:  # noqa: ANN001
        # Guards mirror pi: a scheduled job creates a fresh background agent at
        # fire time, so it is incompatible with resume, inherit_context, and a
        # foreground run.
        if arguments.get("resume"):
            return _tool_result(
                content="Cannot combine `schedule` with `resume` —"
                " schedules create fresh agents.",
            )
        if arguments.get("inherit_context"):
            return _tool_result(
                content="Cannot combine `schedule` with `inherit_context` —"
                " there is no parent conversation at fire time.",
            )
        if arguments.get("run_in_background") is False:
            return _tool_result(
                content="Cannot combine `schedule` with `run_in_background: false`"
                " — scheduled jobs always run in background.",
            )
        definitions = manager.definitions()
        agent_type = str(arguments.get("subagent_type", "general"))
        definition = definitions.get(agent_type)
        if definition is None:
            available = ", ".join(sorted(definitions))
            return _tool_result(
                content=f"Unknown subagent_type: {agent_type}. Available: {available}",
            )
        if definition.fork:
            return _tool_result(
                content="Cannot schedule a fork — there is no parent"
                " conversation to inherit at fire time.",
            )
        if not scheduler.is_active():
            return _tool_result(
                content="Scheduler is not active in this session yet."
                " Try again after the session has fully started.",
            )
        prompt = str(arguments.get("prompt", "")).strip()
        if not prompt:
            return _tool_result(content="prompt is required")
        description = str(arguments.get("description", "")) or f"{agent_type} agent"
        thinking = arguments.get("thinking")
        if thinking is not None:
            thinking = str(thinking)
            if thinking not in THINKING_LEVELS:
                return _tool_result(
                    content=f"Invalid thinking level: {thinking}."
                    f" Valid options: {', '.join(THINKING_LEVELS)}",
                )
        isolation = "worktree" if arguments.get("isolation") == "worktree" else None
        _selection, error = manager.check_spawn_target(
            definition,
            provider=str(arguments["provider"]) if arguments.get("provider") else None,
            model=str(arguments["model"]) if arguments.get("model") else None,
        )
        if error is not None:
            return _tool_result(content=error)
        try:
            job = scheduler.add_job(
                name=description,
                description=description,
                schedule=str(arguments.get("schedule")),
                subagent_type=agent_type,
                prompt=prompt,
                provider=(
                    str(arguments.get("provider"))
                    if arguments.get("provider")
                    else None
                ),
                model=str(arguments.get("model")) if arguments.get("model") else None,
                thinking=thinking,
                max_turns=_coerce_max_turns(arguments.get("max_turns")),
                isolation=isolation,
            )
        except (ValueError, RuntimeError) as exc:
            return _tool_result(content=str(exc))
        next_run = scheduler.get_next_run(job.id) or "(unknown)"
        return _tool_result(
            content=(
                f'Scheduled "{job.name}" (id: {job.id}, type: {job.schedule_type}).'
                f" Next run: {next_run}."
                " Manage via /agents -> Scheduled jobs."
            ),
        )

    async def run_get_result_tool(tool_call_id, arguments, signal=None, on_update=None):  # noqa: ANN001, ANN202
        del tool_call_id, signal, on_update
        await manager.evict_stale()
        agent_id = str(arguments.get("agent_id", ""))
        run = manager.runs.get(agent_id)
        if run is None:
            known = ", ".join(sorted(manager.runs)) or "none"
            return _tool_result(
                content=f"Unknown agent_id: {agent_id}. Known agents: {known}",
            )
        if bool(arguments.get("wait", False)) and run.status in ("running", "queued"):
            # Claim the result before waiting so the background completion
            # path skips its redundant follow-up notification. Poll on status
            # rather than the task so a queued run is followed through its
            # queued -> started -> finished transitions.
            run.result_consumed = True
            manager.cancel_nudge(agent_id)
            while run.status in ("running", "queued"):
                await asyncio.sleep(0.05)
        header = format_run_summary(run)
        header += f"\nUsage: {format_usage_parts(run)}"
        if run.output_file:
            header += f"\nOutput file: {run.output_file}"
        tool_count = _requested_tool_count(arguments.get("tools"))
        now = time.monotonic()
        peek = ""
        peek_details: dict[str, object] = {}
        if tool_count:
            live = run.status in ("running", "queued")
            ended = run.completed_at if run.completed_at is not None else now
            peek = "\n\n" + format_tool_log(
                run.recent_tools,
                tool_count,
                run.tool_calls,
                now if live else ended,
                live=live,
            )
            peek_details["recent_tools"] = tool_log_details(
                run.recent_tools, tool_count, now if live else ended, live=live
            )
        if run.status == "queued":
            return _tool_result(
                content=f"{header}\n\n"
                f"Still queued (max {manager.max_concurrent} concurrent).",
                details=build_liveness_details(run, now) | peek_details,
            )
        if run.status == "running":
            return _tool_result(
                content=f"{header}\n{format_liveness(run, now)}{peek}"
                "\n\nStill running.",
                details=build_liveness_details(run, now) | peek_details,
            )
        run.result_consumed = True
        manager.cancel_nudge(agent_id)
        body = run.error if run.status == "error" else run.result_text
        return _tool_result(
            content=f"{header}\n\n{body or '(no output)'}{peek}",
            details=peek_details or None,
        )

    async def run_steer_tool(tool_call_id, arguments, signal=None, on_update=None):  # noqa: ANN001, ANN202
        del tool_call_id, signal, on_update
        agent_id = str(arguments.get("agent_id", ""))
        message = str(arguments.get("message", ""))
        run = manager.runs.get(agent_id)
        if run is None:
            return _tool_result(
                content=f'Agent not found: "{agent_id}". It may have been cleaned up.',
            )
        if run.status not in ("running", "queued"):
            return _tool_result(
                content=f'Agent "{agent_id}" is not running (status: {run.status}).'
                " Cannot steer a non-running agent.",
            )
        if run.condensing:
            # Its work is done; a steer now would be answered in place of
            # the condensed report.
            return _tool_result(
                content=f'Agent "{agent_id}" has finished and is condensing its'
                " report. Cannot steer it; resume it once it has reported.",
            )
        if run.session is None:
            run.pending_steers.append(message)
            return _tool_result(
                content=f"Steering message queued for agent {agent_id}."
                " It will be delivered once the session initializes.",
            )
        run.session.queue_steering_message(message)
        state_parts = []
        if run.has_usage:
            state_parts.append(f"{lifetime_tokens(run)} tokens")
        state_parts.append(f"{run.tool_calls} tool uses")
        with contextlib.suppress(Exception):
            state_parts.append(
                f"~{run.session.context_token_estimate} context tokens"
            )
        return _tool_result(
            content=f"Steering message sent to agent {agent_id}."
            " The agent will process it after its current tool execution.\n"
            f"Current state: {' · '.join(state_parts)}",
        )

    async def run_stop_tool(tool_call_id, arguments, signal=None, on_update=None):  # noqa: ANN001, ANN202
        del tool_call_id, signal, on_update
        agent_id = str(arguments.get("agent_id", ""))
        run = manager.runs.get(agent_id)
        if run is None:
            return _tool_result(
                content=f'Agent not found: "{agent_id}". It may have been cleaned up.',
            )
        if run.status not in ("running", "queued"):
            return _tool_result(
                content=f'Agent "{agent_id}" is not running (status: {run.status}).'
                " Nothing to stop.",
            )
        liveness = format_liveness(run) if run.status == "running" else ""
        stop_run(run)
        # Cancelling the session kills the tool in flight (bash takes its whole
        # process group down), so the run normally winds down at once. Wait a
        # little to report the outcome here rather than in a later nudge.
        deadline = time.monotonic() + STOP_WAIT_SECONDS
        while run.status in ("running", "queued") and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        header = format_run_summary(run)
        if liveness:
            header += f"\nBefore the stop: {liveness}"
        if run.status in ("running", "queued"):
            return _tool_result(
                content=f"{header}\n\nStop requested, but the agent has not wound"
                f" down after {STOP_WAIT_SECONDS:g}s: its current tool is not"
                " responding to cancellation. You will be notified when it ends.",
            )
        run.result_consumed = True
        manager.cancel_nudge(agent_id)
        return _tool_result(
            content=f"{header}\n\nStopped.\n\n{run.result_text or '(no output)'}",
        )

    # The job verbs, bound to a store built at call time rather than to the
    # watcher's closure: `JobStore()` is a root and no state, so the tool and
    # /jobs read the registry in print mode and before any session_start has
    # armed a watcher — which is exactly when a job started from a shell needs
    # to be visible.
    job_actions = JobActions(
        JobStore(),
        cwd=lambda: tau.context.cwd,
        session_id=lambda: tau.context.session_id,
        log_max_bytes=lambda: manager._get_settings().jobs_log_max_bytes,
    )

    async def run_jobs_tool(tool_call_id, arguments, signal=None, on_update=None):  # noqa: ANN001, ANN202
        del tool_call_id, signal, on_update
        return await job_actions.call(arguments or {})

    def jobs_command(args: str, context) -> str | None:  # noqa: ANN001
        # Plain text, so print mode gets the same table as the TUI; the store
        # is built here and not captured, for the same reason as above.
        del context
        return jobs_command_text(JobStore(), args)

    def agents_command(args: str, context) -> str | None:  # noqa: ANN001
        del args, context
        ui = tau.context.ui
        if supports_menu(ui):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                # Sync handlers can't await dialogs; drive the menu from a
                # loop task (the documented ui-dialogs pattern). Return no
                # message: any text would open a modal the user must dismiss
                # before the menu appears.
                task = loop.create_task(
                    show_agents_menu(
                        manager, ui, scheduler, controller=_current_controller()
                    )
                )
                menu_tasks.add(task)
                task.add_done_callback(menu_tasks.discard)
                return None
        definitions = manager.definitions()
        lines = ["Agent types:", format_agent_type_list(definitions)]
        if manager.runs:
            lines.append("")
            lines.append("Runs:")
            lines.extend(format_run_summary(run) for run in manager.runs.values())
        else:
            lines.append("")
            lines.append("No agents have been spawned in this session.")
        return "\n".join(lines)

    async def on_shutdown(event: SessionShutdownEvent, context) -> None:  # noqa: ANN001
        # Tear down on every shutdown reason (new/resume/branch/quit): runs
        # belong to the outgoing transcript and would otherwise leak sessions.
        del context
        running = [
            run for run in manager.runs.values() if run.status in ("running", "queued")
        ]
        if running:
            # Tau has no hook that fires before a quit is committed, so this
            # can only report the loss, not prevent it. On quit the UI may be
            # gone already; on new/resume/branch the toast is seen.
            names = ", ".join(f"{run.agent_id} ({run.description})" for run in running)
            with contextlib.suppress(Exception):
                tau.notify(
                    f"Session {event.reason}: stopping {len(running)} running"
                    f" subagent(s): {names}. Their partial results are kept in"
                    " the run records and transcripts.",
                    "warning",
                )
        scheduler.stop()
        stop_watcher()
        controller = _current_controller()
        if controller is not None:
            # pi parity: the extension clears its own widgets on shutdown (the
            # host also force-clears as a safety net on the next bridge install).
            controller.teardown()
            ui_controller.clear()
        await manager.shutdown()
        manager.runs.clear()
        manager._notify_sources()  # noqa: SLF001 - same-module teardown signal

    type_list = format_agent_type_list(load_agent_definitions(Path.cwd()))
    authenticated = authenticated_provider_names()
    provider_hint = (
        f" (at session start: {', '.join(authenticated)})" if authenticated else ""
    )
    tau.register_tool(
        AgentTool(
            name="agent",
            label="Agent",
            description=agent_tool_description(
                type_list, delegate=manager.nesting.delegate
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task for the subagent to perform.",
                    },
                    "description": {
                        "type": "string",
                        "description": "A short (3-5 word) description of the task.",
                    },
                    "subagent_type": {
                        "type": "string",
                        "description": "The agent type to spawn (default: general).",
                    },
                    "run_in_background": {
                        "type": "boolean",
                        "description": "Return immediately and notify on completion.",
                    },
                    "provider": {
                        "type": "string",
                        "description": "Omit this unless the user named a"
                        " provider; never guess one. Default: the calling"
                        " session's provider. When set, it must be an exact"
                        " configured Tau provider ID that has credentials"
                        f"{provider_hint}; an unknown or unauthenticated"
                        " provider fails the call.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Omit this unless the user named a"
                        " model; never guess one. Default: the agent type's"
                        " model, else the calling session's model. When set,"
                        " it must be an exact model ID configured for the"
                        " provider; an unknown model fails the call and the"
                        " error lists the configured models.",
                    },
                    "thinking": {
                        "type": "string",
                        "enum": list(THINKING_LEVELS),
                        "description": "Reasoning effort for the subagent"
                        " (default: medium).",
                    },
                    "max_turns": {
                        "type": "number",
                        "minimum": 1,
                        "description": "Soft turn limit; the agent is asked to wrap"
                        " up, then hard-cancelled after a grace period.",
                    },
                    "resume": {
                        "type": "string",
                        "description": "Id of a finished agent to resume with this"
                        " prompt (always runs foreground).",
                    },
                    "isolation": {
                        "type": "string",
                        "enum": ["worktree"],
                        "description": "Set to \"worktree\" to run the agent in an"
                        " isolated git worktree; changes are saved to a"
                        " tau-agent-<id> branch.",
                    },
                    "isolated": {
                        "type": "boolean",
                        "description": "If true, spawn the agent without"
                        " extension tools (core tools only); it cannot spawn"
                        " subagents of its own. Default: false.",
                    },
                    "inherit_context": {
                        "type": "boolean",
                        "description": "If true, prepend a text digest of the"
                        " parent conversation (user and assistant messages;"
                        " tool calls and results omitted) to the agent's"
                        " prompt. Default: false (fresh context). No effect on"
                        " forks, which already have the full history.",
                    },
                    "schedule": {
                        "type": "string",
                        "description": "Opt-in only — fire later instead of now."
                        " Omit to run immediately (the default, almost always"
                        ' correct). Formats: 5-field cron ("0 9 * * 1" = 9am Mon),'
                        ' interval ("5m"/"1h"), one-shot ("+10m" or ISO). Forces'
                        " run_in_background; incompatible with inherit_context and"
                        " resume. Returns a job ID.",
                    },
                },
                "required": ["prompt", "description"],
            },
            execute_fn=run_agent_tool,
            prompt_snippet=(
                AGENT_SNIPPET_DELEGATE if manager.nesting.delegate else AGENT_SNIPPET
            ),
            render_call=render_agent_call,
            render_result=render_agent_result,
            **report_tool_options(),  # type: ignore[arg-type]
        )
    )
    tau.register_tool(
        AgentTool(
            name="get_subagent_result",
            label="Get subagent result",
            description="Get the status or final result of a spawned subagent.",
            parameters={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The id returned when the agent was spawned.",
                    },
                    "wait": {
                        "type": "boolean",
                        "description": "Block until the agent finishes.",
                    },
                    "tools": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": RECENT_TOOLS_KEPT,
                        "description": (
                            "Also list the agent's last N tool calls (tool,"
                            " main argument, running or how long it took),"
                            f" up to {RECENT_TOOLS_KEPT}. Use it to see what a"
                            " long-running agent is doing."
                        ),
                    },
                },
                "required": ["agent_id"],
            },
            execute_fn=run_get_result_tool,
            render_call=render_get_result_call,
            **report_tool_options(),  # type: ignore[arg-type]
        )
    )
    tau.register_tool(
        AgentTool(
            name="steer_subagent",
            label="Steer subagent",
            description=(
                "Send a steering message to a running subagent. It will be"
                " delivered after the agent's current tool execution and appear as"
                " a user message in the agent's conversation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The id of the running agent to steer.",
                    },
                    "message": {
                        "type": "string",
                        "description": "The message to send. This will appear as a"
                        " user message in the agent's conversation.",
                    },
                },
                "required": ["agent_id", "message"],
            },
            execute_fn=run_steer_tool,
            render_call=render_steer_call,
        )
    )
    tau.register_tool(
        AgentTool(
            name="stop_subagent",
            label="Stop subagent",
            description=(
                "Stop a running or queued subagent. Cancels the tool it is"
                " executing (a hung shell command is killed with its children) and"
                " ends the run as aborted; work it already wrote to disk stays."
                " Use it when get_subagent_result shows no events for far longer"
                " than the current step should take: steering cannot reach an agent"
                " whose tool never returns."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The id of the agent to stop.",
                    },
                },
                "required": ["agent_id"],
            },
            execute_fn=run_stop_tool,
            render_call=render_stop_call,
        )
    )
    tau.register_tool(build_jobs_tool(run_jobs_tool))
    tau.register_command(
        "agents",
        agents_command,
        description="List subagent types and runs.",
    )
    tau.register_command(
        "jobs",
        jobs_command,
        description="List background jobs, or tail one by id.",
        usage="/jobs [id]",
    )

    async def on_parent_turn_end(event: object, context) -> None:  # noqa: ANN001
        # The parent's loop drains steering right after its turn_end handlers
        # return, so notices queued here reach the model at this boundary.
        del event, context
        manager.flush_outbox("steer")

    async def on_parent_settled(event: object, context) -> None:  # noqa: ANN001
        # Anything that fired after the parent's last turn_end: the parent is
        # idle now, so sending starts a turn to deliver it.
        del event, context
        manager.flush_outbox("follow_up")

    tau.on("session_start", on_session_start)
    tau.on("session_shutdown", on_shutdown)
    tau.on("turn_end", on_parent_turn_end)
    tau.on("agent_settled", on_parent_settled)


def _foreground_result(run: AgentRun) -> AgentToolResult:
    details = build_notification_details(run, FOREGROUND_RESULT_CHARS)
    if run.status in SUCCESS_STATUSES:
        content = run.result_text or "(subagent produced no output)"
        duration = _duration_ms(run)
        seconds = f"{duration / 1000:.1f}" if duration is not None else "?"
        if run.has_usage:
            tokens_note = f", {lifetime_tokens(run)} tokens"
        elif run.context_tokens:
            tokens_note = f", ~{run.context_tokens} context tokens"
        else:
            tokens_note = ""
        completed_line = (
            f"Agent completed in {seconds}s"
            f" ({run.tool_calls} tool uses{tokens_note})."
        )
        if run.status == "completed_no_report":
            completed_line += f"\n{no_report_note(run)}"
        return _tool_result(
            content=f"{format_run_summary(run)}\n{completed_line}\n\n{content}",
            details=details,
        )
    return _tool_result(
        content=f"{format_run_summary(run)}\n\n{run.error or 'subagent failed'}",
        details=details,
    )


def _coerce_max_turns(value) -> int | None:  # noqa: ANN001
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _requested_tool_count(value: object) -> int:
    """The ``tools`` argument as a count in [0, RECENT_TOOLS_KEPT]."""
    if isinstance(value, bool):
        return 0
    try:
        count = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return 0
    return max(0, min(count, RECENT_TOOLS_KEPT))


def _tool_result(
    *,
    content: str,
    details: dict | None = None,
) -> AgentToolResult:
    """Build a plain text tool result (pi-subagents' textResult).

    Failures are returned as ordinary results whose text says what went wrong,
    never raised: pi-subagents does the same on every path, and raising would
    drop `details` (the completion card) — the loop's error handler rebuilds
    results with empty details.
    """
    return AgentToolResult(content=content, details=details)
