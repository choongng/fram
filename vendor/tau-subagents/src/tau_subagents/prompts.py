"""Child system prompt assembly for subagents, ported from pi-subagents.

Two modes, selected by agent frontmatter `prompt_mode`:

- `replace` (default): the agent body becomes the child's base prompt
  (`custom_system_prompt`), with any preloaded skill sections appended.
- `append`: the parent session's full system prompt is kept verbatim as a
  byte-identical prefix, followed by pi's sub-agent bridge, the sub-agent
  role, an `<active_agent/>` tag, an environment block, the agent body
  wrapped in `<agent_instructions>`, and skill sections. The result is a
  full system prompt override.

Every child is told it is a sub-agent (`sub_agent_role`), in either mode and
also when it has no prompt of its own and runs on Tau's default one. Without
that, a child reads its task as a request from a person and may hand it to
another agent, as its parent did.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from tau_coding import TauResourcePaths
from tau_coding.skills import Skill, format_skill_invocation, load_skills

from .paths import tau_home

if TYPE_CHECKING:
    from .agents import AgentDefinition

GIT_TIMEOUT_SECONDS = 5

SUB_AGENT_BRIDGE = """<sub_agent_context>
You are operating as a sub-agent invoked to handle a specific task.
- Use the read tool instead of cat/head/tail
- Use the edit tool instead of sed/awk
- Use the write tool instead of echo/heredoc
- Use the find tool instead of bash find/ls for file search
- Use the grep tool instead of bash grep/rg for content search
- Make independent tool calls in parallel
- Use absolute file paths
- Do not use emojis
- Be concise but complete
</sub_agent_context>"""

# Kept short and first-person-plain: small models follow it, and it is paid
# for on every child request.
_ROLE_DO_THE_WORK = (
    "You are a sub-agent. Another agent gave you this task to do, not to pass on.\n"
    "Do the work yourself with your own tools, then report what you found or did."
)
_ROLE_WHEN_TO_SPAWN = (
    "Do not hand your task, or a reworded copy of it, to another agent. Spawn an\n"
    "agent only for a clearly separate piece of work that can run alongside yours,\n"
    "and never when you could do it with a few tool calls."
)
_ROLE_NO_AGENT_TOOL = "You have no agent tool; ignore any mention of one."
_ROLE_SECTION = re.compile(
    r"(?:\n\n)?<sub_agent_role>\n.*?\n</sub_agent_role>", re.DOTALL
)


def sub_agent_role(*, can_spawn: bool, agent_listed: bool = False) -> str:
    """The section telling a child it is a delegate and must do its own task.

    A child that has no `agent` tool is not told when to use one. Its prompt
    can still name the tool, where the prompt was written before the tool
    was taken away; `agent_listed` says so, and the section then says the
    tool is not there.
    """
    lines = [_ROLE_DO_THE_WORK]
    if can_spawn:
        lines.append(_ROLE_WHEN_TO_SPAWN)
    elif agent_listed:
        lines.append(_ROLE_NO_AGENT_TOOL)
    body = "\n".join(lines)
    return f"<sub_agent_role>\n{body}\n</sub_agent_role>"


PARENT_CONTEXT_HEADER = """# Parent Conversation Context
The following is the conversation history from the parent session that spawned you.
Use this context to understand what has been discussed and decided so far.

"""

PARENT_CONTEXT_FOOTER = """

---
# Your Task (below)
"""


def build_parent_context(transcript: object) -> str:
    """Text digest of the parent conversation (pi's buildParentContext).

    Keeps user and assistant text turns; drops tool results (too verbose for
    context, matching pi). Tau folds compaction summaries into user messages
    during replay, so they appear as [User] turns rather than pi's [Summary]
    framing. Returns "" when there is nothing to inherit.
    """
    parts: list[str] = []
    for message in transcript or ():  # type: ignore[union-attr]
        role = getattr(message, "role", None)
        if role not in ("user", "assistant"):
            continue
        # Pi-protocol messages carry block content; `.text` joins the text
        # blocks (and is a plain-str passthrough on user messages).
        text = getattr(message, "text", None)
        if not isinstance(text, str):
            continue
        text = text.strip()
        if text:
            label = "User" if role == "user" else "Assistant"
            parts.append(f"[{label}]: {text}")
    if not parts:
        return ""
    return PARENT_CONTEXT_HEADER + "\n\n".join(parts) + PARENT_CONTEXT_FOOTER


def load_available_skills(cwd: Path, home: Path | None = None) -> list[Skill]:
    """Load skills visible from cwd (project `.tau/skills` overrides user)."""
    return load_skills(inherited_resource_paths(cwd, home))


def inherited_resource_paths(cwd: Path, home: Path | None = None) -> TauResourcePaths:
    """Resource paths for a child that inherits skills natively (skills: true)."""
    return TauResourcePaths(root=tau_home(home), cwd=cwd)


def resolve_skill_blocks(
    names: tuple[str, ...] | None, cwd: Path, home: Path | None = None
) -> list[tuple[str, str]]:
    """Resolve named skills to (name, block) pairs; missing → placeholder."""
    if not names:
        return []
    try:
        available = {skill.name: skill for skill in load_available_skills(cwd, home)}
    except Exception:  # noqa: BLE001 - unreadable skill dirs degrade to placeholders
        available = {}
    blocks: list[tuple[str, str]] = []
    for name in names:
        skill = available.get(name)
        if skill is None:
            blocks.append((name, f'(Skill "{name}" not found)'))
        else:
            blocks.append((name, format_skill_invocation(skill)))
    return blocks


def build_child_system_prompt(
    definition: AgentDefinition,
    *,
    parent_prompt: str | None,
    environment: str,
    skill_blocks: list[tuple[str, str]],
    memory_block: str | None = None,
    role: str | None = None,
) -> str | None:
    """Assemble the child system prompt for one agent definition.

    Append mode (with a parent prompt available) returns a full system prompt
    override incorporating `environment` (from `detect_environment`) and
    `role` (from `sub_agent_role`). The parent can be a sub-agent too, whose
    prompt has a bridge and a role of its own; those are taken out, so the
    child has one of each, and the role is the one for its own depth.
    Otherwise returns a base prompt for
    `custom_system_prompt`, or None to keep the default coding prompt; `role`
    is not in it, because there may be no base prompt to put it in. The
    caller passes it as `append_system_prompt`, which Tau adds after either
    base. Extras follow pi's order: memory block first, then skill sections.
    """
    if definition.prompt_mode == "append" and parent_prompt:
        custom_section = ""
        if definition.system_prompt:
            custom_section = (
                "\n\n<agent_instructions>\n"
                f"{definition.system_prompt}\n"
                "</agent_instructions>"
            )
        extras: list[str] = []
        if memory_block:
            extras.append(memory_block)
        extras.extend(
            f"\n# Preloaded Skill: {name}\n{block}" for name, block in skill_blocks
        )
        extras_suffix = "\n\n" + "\n".join(extras) if extras else ""
        role_section = f"{role}\n\n" if role else ""
        prefix = _ROLE_SECTION.sub("", parent_prompt)
        prefix = prefix.replace(f"\n\n{SUB_AGENT_BRIDGE}", "")
        return (
            f"{prefix}\n\n{SUB_AGENT_BRIDGE}\n\n{role_section}"
            f'<active_agent name="{definition.name}"/>\n\n'
            f"{environment}{custom_section}{extras_suffix}"
        )
    parts: list[str] = []
    if definition.system_prompt:
        parts.append(definition.system_prompt)
    if memory_block:
        parts.append(memory_block)
    parts.extend(f"# Preloaded Skill: {name}\n{block}" for name, block in skill_blocks)
    return "\n\n".join(parts) if parts else None


async def detect_environment(cwd: Path) -> str:
    """Build the append-mode environment block for cwd without blocking the loop."""
    return await asyncio.to_thread(_detect_environment_blocking, cwd)


def _detect_environment_blocking(cwd: Path) -> str:
    return (
        "# Environment\n"
        f"Working directory: {cwd}\n"
        f"{_git_status_line(cwd)}\n"
        f"Platform: {sys.platform}"
    )


def _git_status_line(cwd: Path) -> str:
    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return "Not a git repository"
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return "Not a git repository"
    name = branch.stdout.strip() if branch.returncode == 0 else ""
    if name:
        return f"Git repository: yes\nBranch: {name}"
    return "Git repository: yes"
