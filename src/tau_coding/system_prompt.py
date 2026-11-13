"""System prompt assembly for Tau coding sessions."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal
from xml.sax.saxutils import escape

from tau_agent.tools import AgentTool
from tau_coding.branding import BRANDING
from tau_coding.self_docs import tau_docs_path, tau_examples_path, tau_readme_path
from tau_coding.skills import Skill

PromptSourceKind = Literal[
    "default", "system", "append", "extension", "context", "skill", "runtime"
]


@dataclass(frozen=True, slots=True)
class ProjectContextFile:
    """A project instruction file included in the system prompt."""

    path: str
    content: str


@dataclass(frozen=True, slots=True)
class PromptSection:
    """A free-form section appended to the system prompt."""

    title: str | None
    body: str
    source: str | None = None


@dataclass(frozen=True, slots=True)
class SystemPromptSource:
    """One contiguous, attributed section of an effective system prompt."""

    kind: PromptSourceKind
    label: str
    source: str
    content: str


@dataclass(frozen=True, slots=True)
class SystemPromptInspection:
    """An effective system prompt paired with its ordered provenance."""

    text: str
    sources: tuple[SystemPromptSource, ...]


@dataclass(frozen=True, slots=True)
class BuildSystemPromptOptions:
    """Options used to build Tau's system prompt."""

    cwd: Path
    tools: Sequence[AgentTool] = ()
    skills: Sequence[Skill] = ()
    custom_prompt: str | None = None
    append_system_prompt: str | None = None
    context_files: Sequence[ProjectContextFile] = ()
    current_date: date | None = None
    extra_guidelines: Sequence[str] = field(default_factory=tuple)
    append_sections: Sequence[PromptSection] = field(default_factory=tuple)
    extra_sections: Sequence[PromptSection] = field(default_factory=tuple)
    custom_prompt_source: str | None = None


def build_system_prompt(options: BuildSystemPromptOptions) -> str:
    """Build a deterministic Pi-style system prompt for Tau."""
    return build_system_prompt_inspection(options).text


def build_system_prompt_inspection(options: BuildSystemPromptOptions) -> SystemPromptInspection:
    """Build Tau's system prompt and attribute every contiguous section."""
    current_date = options.current_date or date.today()
    cwd = _format_path(options.cwd)
    sources: list[SystemPromptSource] = []

    if options.custom_prompt is not None:
        sources.append(
            SystemPromptSource(
                kind="system",
                label="System prompt override",
                source=options.custom_prompt_source or "runtime configuration",
                content=options.custom_prompt,
            )
        )
    else:
        sources.append(
            SystemPromptSource(
                kind="default",
                label=f"{BRANDING.product_name} default prompt",
                source="tau_coding.system_prompt",
                content=(
                    "You are an expert coding assistant operating "
                    f"inside {BRANDING.product_name}, a coding agent "
                    "harness. You help users by reading files, executing commands, editing code, "
                    "and writing new files."
                    f"\n\nAvailable tools:\n{format_available_tools(options.tools)}"
                    "\n\nIn addition to the tools above, you may have access to other custom tools "
                    "depending on the project."
                    f"\n\nGuidelines:\n{format_guidelines(options.tools, options.extra_guidelines)}"
                    f"\n\n{format_tau_documentation()}"
                ),
            )
        )

    append_sections = list(options.append_sections)
    if options.append_system_prompt:
        append_sections.insert(
            0,
            PromptSection(
                title=None,
                body=options.append_system_prompt,
                source="runtime append configuration",
            ),
        )

    def add_sections(sections: Sequence[PromptSection], kind: PromptSourceKind) -> None:
        for section in sections:
            sources.append(
                SystemPromptSource(
                    kind=kind,
                    label=section.title or "Appended system prompt",
                    source=section.source or "runtime configuration",
                    content=f"\n\n{format_prompt_section(section)}",
                )
            )

    add_sections(append_sections, "append")
    add_sections(options.extra_sections, "extension")

    sources.extend(_project_context_sources(options.context_files))
    if _has_tool(options.tools, "read"):
        sources.extend(_skill_sources(options.skills))
    sources.extend(
        (
            SystemPromptSource(
                kind="runtime",
                label="Current date",
                source=f"{BRANDING.product_name} runtime",
                content=f"\nCurrent date: {current_date.isoformat()}",
            ),
            SystemPromptSource(
                kind="runtime",
                label="Working directory",
                source=cwd,
                content=f"\nCurrent working directory: {cwd}",
            ),
        )
    )
    text = "".join(source.content for source in sources)
    return SystemPromptInspection(
        text=text, sources=tuple(source for source in sources if source.content)
    )


def format_system_prompt_inspection(inspection: SystemPromptInspection) -> str:
    """Render a Markdown source map for the local ``/system`` command."""
    sections: list[str] = []
    for index, source in enumerate(inspection.sources, start=1):
        origin = "".join(
            " " if ord(character) < 32 or ord(character) == 127 else character
            for character in source.source
        ).strip()
        delimiter = "`"
        while delimiter in origin:
            delimiter += "`"
        content = source.content.lstrip("\n")
        sections.append(
            f"#### {index:02d} · {source.label}\n\n"
            f"**Source:** {delimiter}{origin}{delimiter}\n\n"
            f"{content}"
        )
    return "\n\n---\n\n".join(sections)


def format_prompt_section(section: PromptSection) -> str:
    """Render one optional-title free-form prompt section."""
    if section.title is None:
        return section.body
    return f"## {section.title}\n\n{section.body}"


def format_tau_documentation() -> str:
    """Format Pi-style routing hints to Tau's installed reference material."""
    readme_path = _format_path(tau_readme_path())
    docs_path = _format_path(tau_docs_path())
    examples_path = _format_path(tau_examples_path())
    return (
        f"{BRANDING.product_name} documentation (read only when the "
        f"user asks about {BRANDING.product_name} itself, its SDK, "
        "extensions, skills, providers, models, commands, or TUI):\n"
        f"- Main documentation: {readme_path}\n"
        f"- Additional docs: {docs_path}\n"
        f"- Examples: {examples_path} (extensions and custom tools)\n"
        f"- When reading {BRANDING.product_name} docs or "
        "examples, resolve docs/... under Additional docs and "
        "examples/... under Examples, not the current working directory\n"
        "- When asked about: creating or modifying extensions (docs/extensions.md, "
        "examples/extensions/), skills and prompt templates (docs/skills.md), custom "
        "providers or adding built-in providers/models (docs/models.md), CLI and slash "
        "commands (docs/cli.md), TUI usage "
        f"(docs/tui.md), {BRANDING.product_name} architecture and packages (docs/architecture.md)\n"
        f"- When working on {BRANDING.product_name} topics, "
        "read the docs and examples, and follow .md "
        "cross-references before implementing\n"
        f"- Always read relevant {BRANDING.product_name} .md files completely "
        "and follow links to related docs"
    )


def format_available_tools(tools: Sequence[AgentTool]) -> str:
    """Format visible tools using prompt snippets."""
    lines = [f"- {tool.name}: {tool.prompt_snippet}" for tool in tools if tool.prompt_snippet]
    return "\n".join(lines) if lines else "(none)"


def collect_prompt_guidelines(
    tools: Sequence[AgentTool], extra_guidelines: Sequence[str] = ()
) -> list[str]:
    """Collect and de-duplicate system prompt guidelines."""
    names = {tool.name for tool in tools}
    guidelines: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        normalized = value.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        guidelines.append(normalized)

    has_bash = "bash" in names
    has_exploration_tools = bool({"grep", "find", "ls"} & names)
    if has_bash and not has_exploration_tools:
        add("Use bash for file operations like ls, rg, find")
    elif has_bash and has_exploration_tools:
        add(
            "Prefer grep/find/ls tools over bash for file exploration (faster, respects .gitignore)"
        )

    for tool in tools:
        for guideline in tool.prompt_guidelines:
            add(guideline)
    for guideline in extra_guidelines:
        add(guideline)

    add("Inspect relevant files and project instructions before editing")
    add("Make focused changes that preserve the project's architecture and style")
    add("Do not overwrite or discard unrelated user changes")
    add("Use the project's documented commands and package manager")
    add("Run relevant tests, formatting, linting, and type checks after changes")
    add("Report checks honestly; never claim a command passed unless you ran it")
    add("Ask before destructive operations or materially ambiguous design choices")
    add("Be concise in your responses")
    add("Show file paths clearly when working with files")
    return guidelines


def format_guidelines(tools: Sequence[AgentTool], extra_guidelines: Sequence[str] = ()) -> str:
    """Format prompt guidelines as markdown bullets."""
    return "\n".join(
        f"- {guideline}" for guideline in collect_prompt_guidelines(tools, extra_guidelines)
    )


def format_project_context(context_files: Sequence[ProjectContextFile]) -> str:
    """Format project context files using Pi's XML-like wrapper."""
    if not context_files:
        return ""

    lines = [
        "\n\n<project_context>",
        "",
        "Project-specific instructions and guidelines:",
        "",
    ]
    for context_file in context_files:
        lines.append(f'<project_instructions path="{escape(context_file.path)}">')
        lines.append(context_file.content)
        lines.append("</project_instructions>")
        lines.append("")
    lines.append("</project_context>")
    return "\n".join(lines)


def _project_context_sources(
    context_files: Sequence[ProjectContextFile],
) -> tuple[SystemPromptSource, ...]:
    if not context_files:
        return ()
    sources: list[SystemPromptSource] = []
    for index, context_file in enumerate(context_files):
        prefix = (
            "\n\n<project_context>\n\nProject-specific instructions and guidelines:\n\n"
            if index == 0
            else ""
        )
        suffix = "\n</project_context>" if index == len(context_files) - 1 else "\n"
        sources.append(
            SystemPromptSource(
                kind="context",
                label="Project instructions",
                source=context_file.path,
                content=(
                    f'{prefix}<project_instructions path="{escape(context_file.path)}">\n'
                    f"{context_file.content}\n</project_instructions>{suffix}"
                ),
            )
        )
    return tuple(sources)


#: How much of a skill's description its index line carries. The whole of it
#: is in the skill's file, which the model reads when the task may match.
SKILL_SUMMARY_MAX_CHARS = 100

_SKILLS_PREAMBLE = (
    "\n\nThe following skills provide specialized instructions for specific tasks.\n"
    "Each is listed by name with the start of its description. Read a skill's file "
    "when the task may match it; the file says when it applies.\n"
    "When a skill file references a relative path, resolve it against the skill directory "
    "(parent of SKILL.md / dirname of the path) and use that absolute path in tool "
    "commands.\n\n"
    "<available_skills>"
)


#: Abbreviations whose full stop does not end a sentence.
_ABBREVIATIONS = frozenset({"e.g", "i.e", "etc", "vs", "cf", "approx"})

#: A sentence this short is more likely a fragment than the description's point.
_MIN_SENTENCE_CHARS = 20


def skill_summary(description: str | None) -> str:
    """The first sentence of a skill's description, cut to the index's length."""
    text = " ".join((description or "").split())
    sentence = text
    for end in re.finditer(r"[.?!](?=\s|$)", text):
        head = text[: end.start()]
        last_word = head.rsplit(" ", 1)[-1].lower()
        if len(head) >= _MIN_SENTENCE_CHARS and last_word not in _ABBREVIATIONS:
            sentence = head if text[end.start()] == "." else head + text[end.start()]
            break
    else:
        sentence = text.removesuffix(".")
    if not sentence.strip(" .?!"):
        return "No description"
    if len(sentence) <= SKILL_SUMMARY_MAX_CHARS:
        return sentence
    cut = sentence[: SKILL_SUMMARY_MAX_CHARS - 1].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:—-") + "…"


def _skill_index_lines(skills: Sequence[Skill]) -> list[tuple[Skill, str]]:
    """Each visible skill with the index text it adds, header lines included.

    Skills laid out as ``<dir>/<name>/SKILL.md`` are listed under their
    directory, named once; any other file is listed with its own path, last.
    """
    visible = sorted(
        (skill for skill in skills if not skill.disable_model_invocation),
        key=lambda item: item.name,
    )
    grouped: dict[Path | None, list[Skill]] = {}
    for skill in visible:
        standard = skill.path.name == "SKILL.md" and skill.path.parent.name == skill.name
        grouped.setdefault(skill.path.parent.parent if standard else None, []).append(skill)
    entries: list[tuple[Skill, str]] = []
    for directory in sorted(grouped, key=lambda key: key is None):
        for index, skill in enumerate(grouped[directory]):
            line = f"- {escape(skill.name)}: {escape(skill_summary(skill.description))}"
            if directory is None:
                line += f" ({escape(_format_path(skill.path))})"
            if index == 0:
                header = (
                    f"In {escape(_format_path(directory).rstrip('/'))}/, each at <name>/SKILL.md:"
                    if directory is not None
                    else "Elsewhere:"
                )
                line = f"{header}\n{line}"
            entries.append((skill, line))
    return entries


def _skill_sources(skills: Sequence[Skill]) -> tuple[SystemPromptSource, ...]:
    entries = _skill_index_lines(skills)
    return tuple(
        SystemPromptSource(
            kind="skill",
            label=f"Skill: {skill.name}",
            source=str(skill.path),
            content=(_SKILLS_PREAMBLE if index == 0 else "")
            + f"\n{line}"
            + ("\n</available_skills>" if index == len(entries) - 1 else ""),
        )
        for index, (skill, line) in enumerate(entries)
    )


def format_skills_for_prompt(skills: Sequence[Skill]) -> str:
    """Format skills as a compact index for the system prompt.

    Pi lists each skill's whole description; Tau lists its first sentence, cut
    to ``SKILL_SUMMARY_MAX_CHARS``, so a large skill set stays cheap on every
    request. Skills with ``disable_model_invocation`` set are excluded from the
    prompt; they remain invocable explicitly via ``/skill:<name>``.
    """
    return "".join(source.content for source in _skill_sources(skills))


def _has_tool(tools: Sequence[AgentTool], name: str) -> bool:
    return any(tool.name == name for tool in tools)


def _format_path(path: Path) -> str:
    return str(path).replace("\\", "/")
