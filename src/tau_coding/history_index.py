"""The compaction index as a tree, captioned while its range is still in context.

Each compaction adds one chapter: its user turns, grouped into sections, under
one chapter node. Neighbouring top-level nodes of the same level then merge
into one node a level up, like a binary counter, so the forest keeps about
log2(chapters) roots. Turn lines and facts come from code. Captions for every
new section, chapter and merge are asked of the model in the same continuation
request that writes the summary, while the whole range is still in its context.

Rendering never calls the model. It starts from the roots and expands the most
recent collapsed node into its children while the result fits the budget, so
recent history shows turn by turn and old history as captions, with no gap.
See dev-notes/session-recall-and-continuation-compaction.md.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from pydantic import Field

from tau_agent.messages import (
    AssistantMessage,
    BashExecutionMessage,
    ToolResultMessage,
    UserMessage,
    WireModel,
)
from tau_coding.session_recall import (
    COMPACTION_INDEX_HEADING,
    INDEX_NOTE,
    NumberedEntry,
    entry_message,
    index_turn_line,
)

TURN_LEVEL = 0
SECTION_LEVEL = 1
CHAPTER_LEVEL = 2
#: User turns per section.
SECTION_TURNS = 8
#: A caption is held to this many characters, whatever range it covers.
CAPTION_MAX_CHARS = 280
#: Characters the rendered index may take.
INDEX_MAX_CHARS = 6_000
#: Chapters (newest first) whose turn lines are kept in the stored tree. Older
#: chapters keep their sections; `history` still has every turn.
KEEP_TURN_DETAIL_CHAPTERS = 2
#: Files named on one node's facts before the rest are counted.
FACT_FILES = 6
#: Tools whose `path` argument names a file the call changed.
EDITING_TOOLS = frozenset({"edit", "write"})

_CAPTION_LINE = re.compile(r"^\s*[-*]?\s*#(\d+)(?:\s*(?:-|–)\s*#(\d+))?\s*[:\-–]\s*(.+?)\s*$")
CAPTIONS_FENCE = "captions"


class NodeFacts(WireModel):
    """What a range did, counted by code."""

    turns: int = 0
    edited: list[str] = Field(default_factory=list)
    tool_calls: dict[str, int] = Field(default_factory=dict)
    errors: int = 0


class IndexNode(WireModel):
    """One node of the index tree: a turn, a section, a chapter or a merge."""

    first: int
    last: int
    level: int
    text: str = ""
    """A turn's code-built line; empty above the turn level."""
    caption: str | None = None
    facts: NodeFacts = Field(default_factory=NodeFacts)
    children: list[IndexNode] = Field(default_factory=list)

    @property
    def span(self) -> str:
        return f"#{self.first}" if self.first == self.last else f"#{self.first}-#{self.last}"


def build_chapter(items: Sequence[NumberedEntry], *, outputs_dir: Path | None) -> IndexNode | None:
    """The chapter for one compaction's range: turns, grouped into sections."""
    turns = _group_turns(items)
    if not turns:
        return None
    turn_nodes = [
        IndexNode(
            first=turn[0].number,
            last=turn[-1].number,
            level=TURN_LEVEL,
            text=index_turn_line(turn, outputs_dir=outputs_dir),
            facts=_turn_facts(turn),
        )
        for turn in turns
    ]
    sections = [
        _parent(turn_nodes[start : start + SECTION_TURNS], SECTION_LEVEL)
        for start in range(0, len(turn_nodes), SECTION_TURNS)
    ]
    if len(sections) == 1:
        # One section would only repeat its chapter: the chapter holds the turns.
        return _parent(turn_nodes, CHAPTER_LEVEL)
    return _parent(sections, CHAPTER_LEVEL)


def add_chapter(forest: Sequence[IndexNode], chapter: IndexNode) -> list[IndexNode]:
    """Append a chapter and merge equal-level neighbours, newest last.

    Returns a new forest; nodes created by a merge have no caption yet.
    """
    roots = [*forest, chapter]
    while len(roots) >= 2 and roots[-1].level == roots[-2].level:
        right = roots.pop()
        left = roots.pop()
        roots.append(_parent([left, right], left.level + 1))
    return _prune_turn_detail(roots)


def uncaptioned(forest: Sequence[IndexNode]) -> list[IndexNode]:
    """Nodes still to caption, oldest first: new ones, and any a reply left out.

    A node whose children were pruned has nothing left to caption from, so it
    keeps its facts instead.
    """
    found: list[IndexNode] = []
    for node in _walk(forest):
        if node.level > TURN_LEVEL and node.caption is None and node.children:
            found.append(node)
    return found


def apply_captions(forest: Sequence[IndexNode], captions: Mapping[tuple[int, int], str]) -> int:
    """Set captions on uncaptioned nodes by exact range; return how many were set."""
    applied = 0
    for node in _walk(forest):
        if node.level == TURN_LEVEL or node.caption is not None:
            continue
        caption = captions.get((node.first, node.last))
        if caption:
            node.caption = _clip(" ".join(caption.split()), CAPTION_MAX_CHARS)
            applied += 1
    return applied


def caption_request(nodes: Sequence[IndexNode]) -> str:
    """The part of the compaction request that asks for captions.

    The reply is a block to copy with each line filled in, rather than a list
    to interpret: a model asked for an enclosing range tends to caption its
    parts instead, and a fixed set of lines leaves no room for that.
    """
    if not nodes:
        return ""
    # Widest first: a reply that runs short drops the narrow ranges, whose turn
    # lines say most of it anyway, not the merged ones nothing else covers.
    ordered = sorted(nodes, key=lambda node: (-node.level, node.first))
    contents = "\n\n".join(_caption_ask(node) for node in ordered)
    template = "\n".join(f"{node.span}: <caption>" for node in ordered)
    return (
        "Before the summary, copy this block exactly, replacing each <caption>, and "
        "add no other lines:\n\n"
        f"```{CAPTIONS_FENCE}\n{template}\n```\n\n"
        f"A caption is one or two sentences (at most {CAPTION_MAX_CHARS} characters) "
        "saying what was done and decided in that range and why, for someone deciding "
        "whether to look it up. Ranges nest: a wide range gets its own caption covering "
        "all of it, besides any captions its parts have. Describe only what is inside "
        "the range, using no #N references of your own. Name files and results; skip "
        "what the lines below already say. Each range holds:\n\n"
        f"{contents}"
    )


def _caption_ask(node: IndexNode) -> str:
    """One caption ask: the range, and what it holds, so the caption stays inside it."""
    parts = [
        child.text if child.level == TURN_LEVEL else _node_line(child) for child in node.children
    ]
    contents = "\n".join(f"    {_clip(part, 160)}" for part in parts)
    return f"{node.span} ({_level_name(node.level)}):\n{contents}"


def parse_captions(text: str) -> tuple[dict[tuple[int, int], str], str]:
    """Split a reply into its captions block and the rest (the summary).

    A block left open (a reply cut off inside it) runs to the end of the
    reply, so none of it is taken for the summary.
    """
    fence = re.search(
        rf"```{CAPTIONS_FENCE}[^\n]*\n(?P<body>.*?)(?:\n```[ \t]*\n?|\Z)", text, flags=re.DOTALL
    )
    if fence is None:
        return {}, text.strip()
    captions: dict[tuple[int, int], str] = {}
    for line in fence.group("body").splitlines():
        match = _CAPTION_LINE.match(line)
        if match is not None:
            first = int(match.group(1))
            last = int(match.group(2) or first)
            caption = match.group(3)
            if caption.strip() and caption.strip() != "<caption>":
                captions[(min(first, last), max(first, last))] = caption
    rest = (text[: fence.start()] + text[fence.end() :]).strip()
    return captions, rest


def render_index(forest: Sequence[IndexNode], *, max_chars: int = INDEX_MAX_CHARS) -> str:
    """Render the forest within `max_chars`, newest history in the most detail.

    Deterministic for a given forest, so the summary it goes into is stable.
    """
    if not forest:
        return ""
    shown: list[IndexNode] = list(forest)
    while True:
        candidate = _expand_newest(shown, max_chars=max_chars)
        if candidate is None:
            break
        shown = candidate
    lines = [_node_line(node) for node in shown]
    return "\n".join([COMPACTION_INDEX_HEADING, INDEX_NOTE, "", *lines])


def _expand_newest(shown: list[IndexNode], *, max_chars: int) -> list[IndexNode] | None:
    """Expand the newest node that has children, if the result fits."""
    for position in range(len(shown) - 1, -1, -1):
        node = shown[position]
        if not node.children:
            continue
        expanded = [*shown[:position], *node.children, *shown[position + 1 :]]
        if _rendered_size(expanded) <= max_chars:
            return expanded
        # Older nodes are not expanded ahead of a newer one that does not fit.
        return None
    return None


def _rendered_size(nodes: Iterable[IndexNode]) -> int:
    header = len(COMPACTION_INDEX_HEADING) + len(INDEX_NOTE) + 3
    return header + sum(len(_node_line(node)) + 1 for node in nodes)


def _node_line(node: IndexNode) -> str:
    if node.level == TURN_LEVEL:
        return node.text
    label = f"{node.span} [{_level_name(node.level)}, {node.facts.turns} turns]"
    facts = _facts_text(node.facts)
    caption = node.caption or "(no caption)"
    return f"{label} {caption}" + (f" | {facts}" if facts else "")


def _facts_text(facts: NodeFacts) -> str:
    parts: list[str] = []
    if facts.edited:
        files = ", ".join(facts.edited[:FACT_FILES])
        more = len(facts.edited) - FACT_FILES
        parts.append(f"edited {files}" + (f" +{more} more" if more > 0 else ""))
    if facts.tool_calls:
        top = sorted(facts.tool_calls.items(), key=lambda item: (-item[1], item[0]))[:4]
        parts.append(", ".join(f"{count} {name}" for name, count in top))
    if facts.errors:
        parts.append(f"{facts.errors} errors")
    return "; ".join(parts)


def _level_name(level: int) -> str:
    if level == SECTION_LEVEL:
        return "section"
    if level == CHAPTER_LEVEL:
        return "chapter"
    return "part"


def _parent(children: Sequence[IndexNode], level: int) -> IndexNode:
    return IndexNode(
        first=children[0].first,
        last=children[-1].last,
        level=level,
        facts=_merge_facts(child.facts for child in children),
        children=list(children),
    )


def _merge_facts(all_facts: Iterable[NodeFacts]) -> NodeFacts:
    edited: list[str] = []
    calls: Counter[str] = Counter()
    turns = errors = 0
    for facts in all_facts:
        turns += facts.turns
        errors += facts.errors
        calls.update(facts.tool_calls)
        edited.extend(path for path in facts.edited if path not in edited)
    return NodeFacts(turns=turns, edited=edited, tool_calls=dict(calls), errors=errors)


def _turn_facts(turn: Sequence[NumberedEntry]) -> NodeFacts:
    edited: list[str] = []
    calls: Counter[str] = Counter()
    errors = 0
    for item in turn:
        message = entry_message(item.entry)
        if isinstance(message, AssistantMessage):
            for call in message.tool_calls:
                calls[call.name] += 1
                path = call.arguments.get("path")
                if call.name in EDITING_TOOLS and isinstance(path, str) and path not in edited:
                    edited.append(path)
        elif isinstance(message, ToolResultMessage) and message.is_error:
            errors += 1
    return NodeFacts(turns=1, edited=edited, tool_calls=dict(calls), errors=errors)


def _group_turns(items: Sequence[NumberedEntry]) -> list[list[NumberedEntry]]:
    turns: list[list[NumberedEntry]] = []
    for item in items:
        message = entry_message(item.entry)
        if message is None:
            continue
        if isinstance(message, UserMessage | BashExecutionMessage) or not turns:
            turns.append([item])
        else:
            turns[-1].append(item)
    return turns


def _prune_turn_detail(roots: list[IndexNode]) -> list[IndexNode]:
    """Drop turn lines from all but the newest chapters, keeping their sections.

    The stored tree is a snapshot on every compaction, so without this it would
    grow with every turn ever compacted.
    """
    chapters = [node for node in _walk(roots) if node.level == CHAPTER_LEVEL]
    for chapter in chapters[:-KEEP_TURN_DETAIL_CHAPTERS]:
        if chapter.children and chapter.children[0].level == SECTION_LEVEL:
            for section in chapter.children:
                section.children = []
        else:
            chapter.children = []
    return roots


def _walk(nodes: Iterable[IndexNode]) -> Iterable[IndexNode]:
    for node in nodes:
        yield from _walk(node.children)
        yield node


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
