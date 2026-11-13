from datetime import date
from pathlib import Path

from tau_agent import AgentTool, AgentToolResult
from tau_coding import Skill
from tau_coding.system_prompt import (
    SKILL_SUMMARY_MAX_CHARS,
    BuildSystemPromptOptions,
    ProjectContextFile,
    PromptSection,
    build_system_prompt,
    build_system_prompt_inspection,
    collect_prompt_guidelines,
    format_available_tools,
    format_skills_for_prompt,
    skill_summary,
)
from tau_coding.tools import create_coding_tools


async def _unused_executor(
    tool_call_id: str,
    _arguments: object,
    signal: object | None = None,
    on_update: object | None = None,
) -> AgentToolResult:
    del tool_call_id, signal, on_update
    return AgentToolResult(content="")


def test_default_prompt_includes_tools_guidelines_date_and_cwd(tmp_path: Path) -> None:
    tools = create_coding_tools(cwd=tmp_path)

    prompt = build_system_prompt(
        BuildSystemPromptOptions(
            cwd=tmp_path,
            tools=tools,
            current_date=date(2026, 6, 17),
        )
    )

    assert "You are an expert coding assistant operating inside Tau" in prompt
    assert "Available tools:\n- read: Read file contents" in prompt
    assert "- Use bash for file operations like ls, rg, find" in prompt
    assert "- When using bash, include a brief present-participle description" in prompt
    assert "- Use read to examine files instead of cat or sed." in prompt
    assert "- Inspect relevant files and project instructions before editing" in prompt
    assert "- Do not overwrite or discard unrelated user changes" in prompt
    assert "- Report checks honestly; never claim a command passed unless you ran it" in prompt
    assert "Tau documentation (read only when the user asks about Tau itself" in prompt
    assert "custom providers or adding built-in providers/models (docs/models.md)" in prompt
    assert "creating or modifying extensions (docs/extensions.md" in prompt
    assert prompt.endswith(f"Current date: 2026-06-17\nCurrent working directory: {tmp_path}")


def test_tool_without_prompt_snippet_is_hidden_from_available_tools() -> None:
    tool = AgentTool(
        name="hidden",
        label="Hidden",
        description="Still sent to provider",
        parameters={"type": "object"},
        execute_fn=_unused_executor,  # type: ignore[arg-type]
    )

    assert format_available_tools([tool]) == "(none)"


def test_guidelines_are_deduplicated(tmp_path: Path) -> None:
    tools = create_coding_tools(cwd=tmp_path)
    duplicate = tools[0].prompt_guidelines[0]

    guidelines = collect_prompt_guidelines(tools, [duplicate])

    assert guidelines.count(duplicate) == 1


def test_custom_prompt_replaces_default_but_keeps_append_context_and_date(tmp_path: Path) -> None:
    prompt = build_system_prompt(
        BuildSystemPromptOptions(
            cwd=tmp_path,
            tools=create_coding_tools(cwd=tmp_path),
            custom_prompt="Custom base.",
            append_system_prompt="Extra rules.",
            context_files=(ProjectContextFile(path="/repo/AGENTS.md", content="Follow rules."),),
            current_date=date(2026, 6, 17),
        )
    )

    assert prompt.startswith("Custom base.\n\nExtra rules.")
    assert "Available tools:" not in prompt
    assert "Tau documentation" not in prompt
    assert '<project_instructions path="/repo/AGENTS.md">' in prompt
    assert "Follow rules." in prompt
    assert "Current date: 2026-06-17" in prompt


def test_extra_sections_follow_user_append_in_registration_order(tmp_path: Path) -> None:
    prompt = build_system_prompt(
        BuildSystemPromptOptions(
            cwd=tmp_path,
            custom_prompt="Custom base.",
            append_system_prompt="User append.",
            extra_sections=(
                PromptSection(
                    title="Extension procedure", body="First step.\n\n```bash\nuv run pytest\n```"
                ),
                PromptSection(title=None, body="Untitled extension context."),
            ),
            context_files=(ProjectContextFile(path="/repo/AGENTS.md", content="Project rules."),),
            current_date=date(2026, 6, 17),
        )
    )

    expected = (
        "Custom base.\n\nUser append.\n\n## Extension procedure\n\n"
        "First step.\n\n```bash\nuv run pytest\n```\n\n"
        "Untitled extension context."
    )
    assert prompt.startswith(expected)
    assert prompt.index("User append.") < prompt.index("## Extension procedure")
    assert prompt.index("Untitled extension context.") < prompt.index("<project_instructions")


def test_inspection_attributes_prompt_inputs_without_changing_text(tmp_path: Path) -> None:
    skill_path = tmp_path / "skills" / "review" / "SKILL.md"
    options = BuildSystemPromptOptions(
        cwd=tmp_path,
        tools=create_coding_tools(cwd=tmp_path),
        custom_prompt="Custom base.",
        custom_prompt_source="/repo/.tau/SYSTEM.md",
        append_sections=(
            PromptSection(title=None, body="Personal rules.", source="~/.tau/APPEND_SYSTEM.md"),
        ),
        extra_sections=(
            PromptSection(title="Extension rules", body="Run checks.", source="extension: checks"),
        ),
        context_files=(
            ProjectContextFile(path="/home/.agents/AGENTS.md", content="Global rules."),
            ProjectContextFile(path="/repo/AGENTS.md", content="Project rules."),
        ),
        skills=(Skill(name="review", path=skill_path, content="", description="Review code"),),
        current_date=date(2026, 6, 17),
    )

    inspection = build_system_prompt_inspection(options)

    assert inspection.text == build_system_prompt(options)
    assert "".join(source.content for source in inspection.sources) == inspection.text
    assert [source.source for source in inspection.sources] == [
        "/repo/.tau/SYSTEM.md",
        "~/.tau/APPEND_SYSTEM.md",
        "extension: checks",
        "/home/.agents/AGENTS.md",
        "/repo/AGENTS.md",
        str(skill_path),
        "Tau runtime",
        str(tmp_path),
    ]


def test_empty_custom_prompt_is_still_custom(tmp_path: Path) -> None:
    prompt = build_system_prompt(
        BuildSystemPromptOptions(
            cwd=tmp_path,
            tools=create_coding_tools(cwd=tmp_path),
            custom_prompt="",
            append_system_prompt="Extra rules.",
            current_date=date(2026, 6, 17),
        )
    )

    assert prompt.startswith("\n\nExtra rules.")
    assert "Available tools:" not in prompt
    assert "Current date: 2026-06-17" in prompt


def test_skills_are_indexed_by_directory_and_escaped(tmp_path: Path) -> None:
    skills = [
        Skill(
            name="review&check",
            path=tmp_path / "skills" / "review&check" / "SKILL.md",
            content="ignored",
            description="Review <code>.",
        ),
        Skill(name="loose", path=tmp_path / "loose.md", content="", description=None),
        Skill(
            name="testing",
            path=tmp_path / "skills" / "testing" / "SKILL.md",
            content="",
            description="Write tests, e.g. unit tests. Use when adding a feature.",
        ),
    ]

    formatted = format_skills_for_prompt(skills)

    assert formatted.endswith(
        "<available_skills>\n"
        f"In {tmp_path}/skills/, each at <name>/SKILL.md:\n"
        "- review&amp;check: Review &lt;code&gt;\n"
        "- testing: Write tests, e.g. unit tests\n"
        "Elsewhere:\n"
        f"- loose: No description ({tmp_path}/loose.md)\n"
        "</available_skills>"
    )


def test_a_long_skill_description_is_cut_to_its_summary() -> None:
    long = "Run heavy, long, or fork-prone tools inside a resource-capped systemd scope " * 3

    summary = skill_summary(long + ". Use when building.")

    assert len(summary) <= SKILL_SUMMARY_MAX_CHARS
    assert summary.endswith("…") and long.startswith(summary[:-1])
    assert skill_summary("A short summary line. Use when.") == "A short summary line"


def test_skill_summary_handles_abbreviations_other_endings_and_empty_text() -> None:
    assert (
        skill_summary("A tool for many things, e.g. unit tests and more. Use when.")
        == "A tool for many things, e.g. unit tests and more"
    )
    assert skill_summary("Does this skill apply here? Yes. Use when.") == (
        "Does this skill apply here?"
    )
    assert skill_summary("   ") == "No description"
    assert skill_summary(".") == "No description"
    assert skill_summary(None) == "No description"
    word = "x" * 300
    assert skill_summary(word) == "x" * (SKILL_SUMMARY_MAX_CHARS - 1) + "…"


def test_skill_index_groups_directories_in_order_and_lists_odd_layouts_elsewhere(
    tmp_path: Path,
) -> None:
    skills = [
        Skill(name="b", path=tmp_path / "two" / "b" / "SKILL.md", content="", description="B."),
        Skill(name="a", path=tmp_path / "one" / "a" / "SKILL.md", content="", description="A."),
        Skill(name="c", path=tmp_path / "one" / "c" / "SKILL.md", content="", description="C."),
        Skill(
            name="renamed",
            path=tmp_path / "one" / "dir" / "SKILL.md",
            content="",
            description="R.",
        ),
        Skill(name="root", path=Path("/root/SKILL.md"), content="", description="Top."),
    ]

    formatted = format_skills_for_prompt(skills)

    assert formatted.endswith(
        "<available_skills>\n"
        f"In {tmp_path}/one/, each at <name>/SKILL.md:\n"
        "- a: A\n"
        "- c: C\n"
        f"In {tmp_path}/two/, each at <name>/SKILL.md:\n"
        "- b: B\n"
        "In /, each at <name>/SKILL.md:\n"
        "- root: Top\n"
        "Elsewhere:\n"
        f"- renamed: R ({tmp_path}/one/dir/SKILL.md)\n"
        "</available_skills>"
    )


def test_format_skills_for_prompt_excludes_disabled_skills(tmp_path: Path) -> None:
    visible = Skill(
        name="visible",
        path=tmp_path / "skills" / "visible" / "SKILL.md",
        content="",
        description="Visible skill",
    )
    hidden = Skill(
        name="hidden",
        path=tmp_path / "skills" / "hidden" / "SKILL.md",
        content="",
        description="Hidden skill",
        disable_model_invocation=True,
    )

    formatted = format_skills_for_prompt([visible, hidden])

    assert "- visible: Visible skill" in formatted
    assert "hidden" not in formatted

    assert format_skills_for_prompt([hidden]) == ""


def test_skills_are_included_only_when_read_tool_is_available(tmp_path: Path) -> None:
    skill = Skill(name="testing", path=tmp_path / "testing.md", content="", description="Test")
    no_read_tool = AgentTool(
        name="custom",
        label="Custom",
        description="Custom",
        parameters={"type": "object"},
        execute_fn=_unused_executor,  # type: ignore[arg-type]
        prompt_snippet="Custom tool",
    )

    without_read = build_system_prompt(
        BuildSystemPromptOptions(cwd=tmp_path, tools=[no_read_tool], skills=[skill])
    )
    with_read = build_system_prompt(
        BuildSystemPromptOptions(
            cwd=tmp_path, tools=create_coding_tools(cwd=tmp_path), skills=[skill]
        )
    )

    assert "<available_skills>" not in without_read
    assert "<available_skills>" in with_read
