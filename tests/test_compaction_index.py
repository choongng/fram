"""The compaction index tree: built by code, captioned at compaction, rendered to a cap."""

from __future__ import annotations

from pathlib import Path

from tau_agent import AssistantMessage, ToolCall
from tau_agent.messages import ToolResultMessage, UserMessage, assistant_content
from tau_agent.session import MessageEntry
from tau_coding.history_index import (
    CHAPTER_LEVEL,
    SECTION_LEVEL,
    SECTION_TURNS,
    TURN_LEVEL,
    IndexNode,
    add_chapter,
    apply_captions,
    build_chapter,
    caption_request,
    parse_captions,
    render_index,
    uncaptioned,
)
from tau_coding.session_recall import (
    COMPACTION_INDEX_HEADING,
    NumberedEntry,
    split_compaction_index,
    with_compaction_index,
)


def _turn(n: int, *, error: bool = False, edit: str | None = None) -> list[MessageEntry]:
    name, arguments = ("edit", {"path": edit}) if edit else ("bash", {"command": f"pytest -k t{n}"})
    call = ToolCall(id=f"c{n}", name=name, arguments=arguments)
    return [
        MessageEntry(message=UserMessage(content=f"task {n}\nmore detail")),
        MessageEntry(message=AssistantMessage(content=assistant_content("ok", [call]))),
        MessageEntry(
            message=ToolResultMessage(
                tool_call_id=f"c{n}", tool_name=name, content="out", is_error=error
            )
        ),
        MessageEntry(message=AssistantMessage(content="done")),
    ]


def _chapter(turns: range, start: int = 1, outputs_dir: Path | None = None) -> IndexNode:
    entries = [entry for n in turns for entry in _turn(n, edit="a.py" if n % 2 else None)]
    items = [NumberedEntry(start + i, entry) for i, entry in enumerate(entries)]
    chapter = build_chapter(items, outputs_dir=outputs_dir)
    assert chapter is not None
    return chapter


def test_turn_lines_name_the_range_ask_calls_errors_and_files(tmp_path: Path) -> None:
    (tmp_path / "bash-c2.txt").write_text("out")
    entries = [*_turn(1), *_turn(2, error=True)]
    items = [NumberedEntry(10 + i, entry) for i, entry in enumerate(entries)]

    chapter = build_chapter(items, outputs_dir=tmp_path)

    assert chapter is not None and chapter.level == CHAPTER_LEVEL
    assert [child.text for child in chapter.children] == [
        '#10-#13 user: "task 1" -> bash `pytest -k t1`',
        '#14-#17 user: "task 2" -> bash `pytest -k t2` (error) [bash-c2.txt]',
    ]
    assert chapter.facts.turns == 2 and chapter.facts.errors == 1
    assert chapter.facts.tool_calls == {"bash": 2}


def test_a_long_chapter_is_grouped_into_sections_with_merged_facts() -> None:
    chapter = _chapter(range(SECTION_TURNS * 2 + 1))

    assert [child.level for child in chapter.children] == [SECTION_LEVEL] * 3
    assert [len(child.children) for child in chapter.children] == [SECTION_TURNS] * 2 + [1]
    assert chapter.facts.turns == SECTION_TURNS * 2 + 1
    assert chapter.facts.edited == ["a.py"]


def test_chapters_merge_like_a_binary_counter() -> None:
    forest: list[IndexNode] = []
    for n in range(5):
        forest = add_chapter(forest, _chapter(range(n * 3, n * 3 + 3), start=n * 100 + 1))

    assert [node.level for node in forest] == [CHAPTER_LEVEL + 2, CHAPTER_LEVEL]
    assert forest[0].first == 1 and forest[0].last == forest[1].first - 89
    assert forest[0].facts.turns == 12


def test_only_the_newest_chapters_keep_turn_lines() -> None:
    forest: list[IndexNode] = []
    for n in range(4):
        forest = add_chapter(forest, _chapter(range(n * 3, n * 3 + 3), start=n * 100 + 1))

    chapters = [
        node
        for root in forest
        for node in [root, *root.children, *(c for ch in root.children for c in ch.children)]
        if node.level == CHAPTER_LEVEL
    ]
    assert [bool(chapter.children) for chapter in chapters] == [False, False, True, True]


def test_captions_are_asked_for_new_and_merged_nodes_and_applied_by_range() -> None:
    forest = add_chapter([], _chapter(range(3)))
    forest[0].caption = "first chapter"
    forest = add_chapter(forest, _chapter(range(3, 6), start=101))
    wanted = uncaptioned(forest)

    assert [(node.first, node.level) for node in wanted] == [(101, CHAPTER_LEVEL), (1, 3)]
    request = caption_request(wanted)
    assert "#101-#112 (chapter)" in request and "#1-#112 (part)" in request

    reply = (
        "```captions\n#101-#112: Second chapter work.\n- #1 - #112: Both chapters.\n```\n"
        "## Goal\nShip it"
    )
    captions, summary = parse_captions(reply)
    assert summary == "## Goal\nShip it"
    assert apply_captions(forest, captions) == 2
    assert uncaptioned(forest) == []


def test_a_reply_without_captions_is_all_summary() -> None:
    assert parse_captions("## Goal\nx") == ({}, "## Goal\nx")


def test_a_captions_block_left_open_is_not_taken_for_the_summary() -> None:
    captions, summary = parse_captions("```captions\n#1-#4: Did the work.\n#5-#8: Started")

    assert summary == ""
    assert captions[(1, 4)] == "Did the work."


def test_rendering_expands_the_newest_history_first_within_the_cap() -> None:
    forest: list[IndexNode] = []
    for n in range(6):
        chapter = _chapter(range(n * 10, n * 10 + 10), start=n * 100 + 1)
        chapter.caption = f"chapter {n}"
        for section in chapter.children:
            section.caption = f"section of {n}"
        forest = add_chapter(forest, chapter)
    for node in uncaptioned(forest):
        node.caption = "merged"

    small = render_index(forest, max_chars=1_200)
    large = render_index(forest, max_chars=20_000)

    assert len(small) <= 1_200
    assert small.startswith(COMPACTION_INDEX_HEADING)
    lines = large.splitlines()[3:]
    # Contiguous ranges, no gap, oldest first; the newest turns are shown whole.
    firsts = [int(line.split()[0].lstrip("#").split("-")[0]) for line in lines]
    assert firsts == sorted(firsts) and firsts[0] == 1
    assert 'user: "task 59"' in large
    assert 'user: "task 0"' not in large
    # Old chapters kept their sections only: those show as captioned sections.
    assert "[section, " in large and "section of 0" in large
    # The small render shows old history as captions, never as a gap.
    assert "(no caption)" not in small and "merged" in small


def test_an_uncaptioned_node_shows_its_facts() -> None:
    forest = add_chapter([], _chapter(range(SECTION_TURNS * 3)))
    text = render_index(forest, max_chars=600)

    assert "(no caption)" in text and "edited a.py" in text


def test_the_tree_round_trips_through_json() -> None:
    forest = add_chapter([], _chapter(range(12)))
    stored = [node.model_dump(mode="json", exclude_defaults=True) for node in forest]

    assert [IndexNode.model_validate(node) for node in stored] == forest
    assert forest[0].children[0].children[0].level == TURN_LEVEL


def test_split_compaction_index_finds_the_summary_after_it() -> None:
    forest = add_chapter([], _chapter(range(3)))
    text = with_compaction_index(render_index(forest), "## Goal\nShip it")

    index, summary = split_compaction_index(text)

    assert index == render_index(forest)
    assert summary == "## Goal\nShip it"
