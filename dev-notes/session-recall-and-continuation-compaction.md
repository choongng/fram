# Session Recall and Continuation Compaction

## Why

Compaction used to be a separate model request: its own system prompt, no tools,
and the dropped messages flattened into one new user message. None of that
matched the cached conversation, so the model re-read the whole transcript at
the full input price, and the write it paid for was never read back. It then
wrote an 8,000-character summary, and output costs about five times input.

The redesign has four parts:

1. Every tool result except a successful `read` is saved to its own plain-text
   file, so the agent can `grep` and `read` old outputs with the tools it
   already has. A successful `read` is not saved: its source file is already on
   disk.
2. Every session entry has a stable reference, `#N`, that the harness can
   resolve. The model never sees one during ordinary turns.
3. At compaction, code rather than the model writes an index of the summarized
   range that cites those references and output files, and a `history` tool
   turns a reference back into the entry.
4. The summary is asked for as a continuation of the cached conversation, so the
   transcript is a cache read.

## Entry references

`#N` is the entry's 1-based position in the session file. The file is
append-only: `append` adds a line and `append_batch` rewrites the old bytes
unchanged with the batch after them, and branches share the file. So a position
never changes and is unique across branches, and it needs no stored field. A
copied session (export, fork) is a new file and numbers its own entries.

References appear only in text the harness writes after a compaction (the index)
and in `history` results. Ordinary messages carry no tag, so references cost no
context, cannot disturb the cached prefix, and the model has no pattern to
imitate when writing its own replies.

## Tool output files

`CodingSession._buffer_result` writes the full text of each tool result other
than a successful `read` to `<session-id>.outputs/<tool>-<call-id>.txt` beside
the session file (`output_file_name`). Small results are written as well as
large ones. The directory is created 0700: outputs can hold secrets. They are
already in the JSONL, but the old shared `/tmp/tau-buffers` fallback was
readable by other users.

Files are named by tool-call ID, not by `#N`, because the truncation marker of
an oversized result must name the file before the result becomes an entry. The
name is derived from the call ID, so code can find the file for any recorded
tool result without storing a path. A user terminal command has no call ID and
is named by a hash of its output.

This replaces the old `buffers/` directory, which was shared by every session in
a project directory and named by content hash. Oversized results still show a
window plus a marker, now pointing at the per-call file.

## The `history` tool

`history` reads the current branch of the session file, including entries a
compaction has dropped from context:

- `refs`: `#12`, `#12-#40` or `12,15`: those entries in full, up to the answer's
  character limit. The answer then names the first entry left out, so a wide
  range is read in parts instead of filling the context. A first entry longer
  than the limit is cut from its start and says where it stops.
- `query`: a case-insensitive search over entry text and tool arguments. Earlier
  `history` calls and answers are skipped, because each one holds the query that
  is being searched for. They are searched only with `tool="history"`.
- `tool`: only tool calls and results of that tool.

The character limit is the session's inline limit for tool results (see
[tool-result-buffers.md](tool-result-buffers.md)), at most `REFS_MAX_CHARS`
(20,000), and a search listing stops at it too. A larger answer would be
windowed like any other tool result, which keeps its end: the model would see
the last entries of a range without the first, and a note about an entry it can
no longer see.

Each entry comes back with its `#N`, role, and for a tool result the path of its
output file, when it has one. It reads only the current branch, so a branch the
model never saw does not appear. It is added with the default coding tools, so a
session built with an explicit tool list (tests, some embedders) does not get
it.

## The compaction index

The index is a tree (`src/tau_coding/history_index.py`), stored as JSON on each
`CompactionEntry.index` and rendered into the start of its summary.

```text
part      #1-#400    merged chapters: a caption written from its parts' captions
chapter   #181-#260  one compaction's range
section   #181-#204  about eight user turns (only when a chapter has more)
turn      #181-#186  user: "..." -> edit `session.py`; bash `pytest` (error) [bash-toolu_01.txt]
```

- **Turns** are lines built by code (`index_turn_line`): the range, the first
  line of what was asked, each tool call with its key argument, error status and
  output file. A turn an earlier compaction cut at its boundary shows as
  `(turn continued)`.
- **Facts** on every node are counted by code: turns, files edited, tool calls
  by name, errors. They stand in for a missing caption.
- **Captions** are written by the model at the compaction that creates the node,
  while its entries are still in context. The continuation request ends with a
  fill-in block, one `#first-#last: <caption>` line per new node, widest range
  first, each listed with what it holds. Asked for an enclosing range in a plain
  list, a model tends to caption its parts instead; with the fill-in block it
  captions every range. Code keeps the captions whose ranges match exactly.
- **Merging:** each compaction appends its chapter, and two neighbouring roots
  of the same level merge into a node one level up, like a binary counter. So
  there are about log2(compactions) roots, and every merge is known, and
  captioned, at the compaction that causes it.
- **Pruning:** only the newest two chapters keep their turn lines in the stored
  tree; older ones keep their sections. The stored tree is a snapshot on each
  compaction, so without this it would grow with every turn compacted. `history`
  still has every entry.

Rendering (`render_index`) never calls a model. It starts from the roots and
expands the newest node that has children while the result fits the cap
(`INDEX_MAX_CHARS`), so recent history shows turn by turn and older history as
captions, with no gap. Expansion goes one level at a time, so at a cap where the
newest turns alone would fit but their chapter's sibling caption does not, it
stops at the merged caption.

Compactions written before the tree have no `index`. The next compaction starts
a fresh tree from its own range; their summaries still hold what they
summarized.

## Continuation compaction

`_generate_compaction_summary` first sends the conversation's own request, with
the same system prompt, tools, `provider_context` messages and cache TTL
(`long_lived`), plus one more user message. That message holds the index and
asks for a summary of everything up to the kept boundary (`#N`), citing
references instead of copying detail. The prefix matches the previous request,
so the transcript is read from cache.

The model must not call a tool here. Tools stay in the request, because removing
them, or setting `tool_choice`, would change the cached prefix. A reply with
tool calls is treated as a failure.

When the conversation plus the request and `CONTINUATION_SUMMARY_OUTPUT_TOKENS`
would not fit the context window, the request is sent again without the kept
tail. The model is not asked to summarize the tail anyway, so leaving it out
costs nothing. The summarized range usually ends where an earlier request ended,
the one that produced the first kept assistant message, so that prefix was
cached when that request was sent. A cache entry that has expired is written
again, which costs no more than the separate request would. Overflow compaction
skips the full request: the provider has already refused the context, so only
the tail-less request is tried.

A reply that stopped at the output limit (`stop_reason == "length"`) keeps its
summary, marked `[Summary cut off at the output limit.]`. A reply cut off inside
its captions block has no summary: `parse_captions` treats an unclosed block as
running to the end, so none of it is taken for the summary.

Tau falls back to the old separate request when:

- the request would not fit the window even without the kept tail,
- the continuation fails, calls a tool, or returns no summary.

Either path asks for captions and stores the tree with the summary. The separate
request labels its messages by their `#N` references (`ref=#N`), so the ranges
it is asked to caption match the messages it sees. Only when an entry has no
reference does it fall back to numbering them from 1.

Measured on `claude-haiku-4-5` with a 1h-cached session of four large turns: the
last turn read 48,047 from cache and wrote 15,567. The compaction then read
63,614 (the whole conversation) from cache, wrote 576 (its own request) and sent
3 uncached input tokens. A separate request would have sent all ~64k at the full
input price.

## How it maps to Pi

Pi summarizes in a separate request, and Tau inherited that. Pi has no output
files, no entry references and no `history` tool. The summary format is
unchanged here. Replacing it with a shorter one that leans on the index needs a
side-by-side comparison on real long sessions first.

## Testing

```bash
uv run pytest tests/test_tool_output_files.py tests/test_session_history.py \
  tests/test_compaction_index.py tests/test_continuation_compaction.py
```
