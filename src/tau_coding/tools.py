"""Built-in filesystem and shell tools for Tau coding sessions.

The module exposes factory functions that create provider-neutral `AgentTool`
objects plus richer `ToolDefinition` objects for callers that need prompt
metadata and JSON schemas. The tools operate relative to a configurable working
directory, return structured `AgentToolResult` values, and keep local
filesystem/shell behavior outside the reusable `tau_agent` package.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import signal
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Any

from tau_agent.messages import ImageContent, TextContent
from tau_agent.tools import (
    AgentTool,
    AgentToolResult,
    ToolArgumentPreparer,
    ToolCancellationToken,
    ToolUpdateCallback,
)
from tau_agent.types import JSONValue
from tau_coding.image_processing import (
    DEFAULT_MAX_SOURCE_IMAGE_BYTES,
    ImageProcessingFailure,
    detect_image_family_mime_type,
    detect_supported_image_mime_type,
    process_image,
    unsupported_image_reason,
)
from tau_coding.line_anchors import WRAP_WIDTH, line_rows, strip_row_prefixes

DEFAULT_MAX_OUTPUT_BYTES = 50 * 1024
DEFAULT_MAX_OUTPUT_LINES = 2_000
#: Seconds a model-issued bash command may run when the call names no timeout.
#: Long enough for a test suite or a build; short enough that a command that
#: never ends hands control back instead of holding the turn forever.
DEFAULT_BASH_TIMEOUT_SECONDS = 600.0
#: Environment variable that overrides `DEFAULT_BASH_TIMEOUT_SECONDS` for the
#: tools `create_coding_tools` builds, so one session can run with a different
#: bound without changing anyone else's.
BASH_TIMEOUT_ENV_VAR = "TAU_BASH_TIMEOUT_SECONDS"
#: What `bash` returns for a command that printed nothing.
NO_OUTPUT = "(no output)"
IMAGE_SNIFF_BYTES = 64 * 1024
UTF8_BOM = "\ufeff"


class ToolInputError(ValueError):
    """Raised when a tool receives invalid structured arguments."""


def _cut_off_arguments_error(name: str, text: str, required: Sequence[str] = ()) -> str:
    """Why a call whose arguments are only unparsed text did not run.

    Providers hand a call whose argument text is not valid JSON to the tool as
    `{"_raw_arguments": text}`. Reporting the first argument the tool looks for
    as a wrong type sends the model back to repeat the same call, so the error
    says which arguments arrived, which string never closed and how it ends,
    which required arguments never came, and how to send less in one call. A
    call whose strings all closed but hold a bad escape is told that instead.
    """
    try:
        json.loads(text)
    except ValueError as exc:
        error = str(exc)
    else:
        error = "not an object of named arguments"
    outcome = f"the {name} tool did not run"
    if name == "write":
        outcome += " and nothing was written"
    if not text.lstrip().startswith("{") or "Extra data" in error:
        return (
            f"The {name} call's arguments are not one JSON object of named arguments "
            f"({error}), so {outcome}. Send the call again with a single JSON object."
        )
    closed, unclosed, length = _partial_arguments(text)
    arrived = ""
    if unclosed is not None and length == 0:
        arrived = f" `{unclosed}` has no value."
    elif unclosed is not None:
        tail = text[-CUT_OFF_TAIL:].lstrip()
        arrived = (
            f' `{unclosed}` opened and never closed: it ran {length} characters, ending "…{tail}".'
        )
    missing = [key for key in required if key not in closed and key != unclosed]
    if unclosed is None and not missing and closed and text.rstrip().endswith("}"):
        return (
            f"The {name} call's arguments are not valid JSON ({error}), so {outcome}. "
            f"Every argument arrived ({_names(closed)}), but a string holds a character "
            "JSON does not take as written: inside a string, a backslash is written "
            "`\\\\` (so `\\d` is `\\\\d`), a newline `\\n` and a tab `\\t`. Send the call "
            "again with those escaped."
        )
    if missing:
        arrived += f" {_names(missing)} never came."
    first = ""
    if missing:
        first = (
            f"Send {_names(missing)} first, then each string holding only its own text, "
            "with any reasoning before the call, not inside it. "
        )
    return (
        f"The {name} call's arguments are not valid JSON after {len(text)} characters "
        f"({error}), so {outcome}.{arrived} This is how a reply cut off at the output "
        f"limit ends. {first}Send less in one call: write a long file in parts (write "
        "the first part, then add the rest with edit or by appending through bash, "
        "since a second write replaces the file), or generate patterned "
        "data with a short script through bash."
    )


#: Characters of an unclosed argument string `_cut_off_arguments_error` quotes.
CUT_OFF_TAIL = 80


def _names(keys: Sequence[str]) -> str:
    return " and ".join(f"`{key}`" for key in keys)


def _partial_arguments(text: str) -> tuple[list[str], str | None, int]:
    """The arguments in object text `text` whose values closed, the one whose value
    did not, and how many characters that value ran."""
    decoder = json.JSONDecoder()
    at = len(text) - len(text.lstrip())
    if not text.startswith("{", at):
        return [], None, 0
    at += 1
    closed: list[str] = []
    while True:
        while at < len(text) and text[at] in " \t\r\n,":
            at += 1
        if not text.startswith('"', at):
            return closed, None, 0
        try:
            key, at = decoder.raw_decode(text, at)
        except ValueError:
            return closed, None, 0
        while at < len(text) and text[at] in " \t\r\n:":
            at += 1
        try:
            _, at = decoder.raw_decode(text, at)
        except ValueError:
            # A string with a bad escape (`\d`) or a raw newline closed, though
            # it did not parse; only one with no closing quote ran on.
            end = _string_end(text, at)
            if end is None:
                return closed, str(key), len(text) - at
            at = end
        closed.append(str(key))


def _string_end(text: str, at: int) -> int | None:
    """Where the JSON string opening at `at` ends, past its closing quote, read
    leniently (any character after a backslash is skipped); None when it does
    not close or `at` opens no string."""
    if not text.startswith('"', at):
        return None
    index = at + 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == '"':
            return index + 1
        index += 1
    return None


@dataclass(slots=True)
class ImageSupportState:
    """Mutable active-model image capability shared with built-in tools."""

    supported: bool | None = None


@dataclass(frozen=True, slots=True)
class ReadOperations:
    """Pluggable filesystem operations used by the read tool."""

    validate_path: Callable[[Path], None]
    read_bytes: Callable[[Path], bytes]
    size_bytes: Callable[[Path], int] | None = None
    read_prefix: Callable[[Path, int], bytes] | None = None


@dataclass(frozen=True, slots=True)
class TruncationResult:
    """Metadata describing how a tool output was shortened.

    `content` contains the returned slice. The remaining fields record whether
    truncation happened, whether the line or byte limit was responsible, the
    total size of the original output, the size of the returned output, and
    edge cases such as partial-line output or a first line that is too large to
    display safely.
    """

    content: str
    truncated: bool
    truncated_by: str | None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    first_line_exceeds_limit: bool
    max_lines: int
    max_bytes: int

    def to_json(self) -> dict[str, JSONValue]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Complete definition for a coding tool before provider conversion.

    A definition contains the tool name, user-facing description, prompt
    metadata, JSON input schema, and async executor. `to_agent_tool()` converts
    it into the smaller `AgentTool` type consumed by the provider-neutral agent
    loop while preserving prompt metadata for clients that render tool guidance.
    """

    name: str
    description: str
    prompt_snippet: str
    prompt_guidelines: tuple[str, ...]
    input_schema: Mapping[str, JSONValue]
    executor: Callable[
        [Mapping[str, JSONValue], ToolCancellationToken | None], Awaitable[AgentToolResult]
    ]
    changes_workspace: bool = False
    prepare_arguments: ToolArgumentPreparer | None = None

    def to_agent_tool(self) -> AgentTool:
        """Convert the coding definition to the Pi-compatible core tool."""

        async def execute(
            tool_call_id: str,
            arguments: Mapping[str, JSONValue],
            signal: ToolCancellationToken | None = None,
            on_update: ToolUpdateCallback | None = None,
        ) -> AgentToolResult:
            del tool_call_id, on_update
            if set(arguments) == {"_raw_arguments"}:
                required = self.input_schema.get("required")
                raise ToolInputError(
                    _cut_off_arguments_error(
                        self.name,
                        str(arguments["_raw_arguments"]),
                        [key for key in required if isinstance(key, str)]
                        if isinstance(required, list)
                        else (),
                    )
                )
            return await self.executor(arguments, signal)

        return AgentTool(
            name=self.name,
            label=self.name,
            description=self.description,
            parameters=self.input_schema,
            execute_fn=execute,
            prompt_snippet=self.prompt_snippet,
            prompt_guidelines=self.prompt_guidelines,
            changes_workspace=self.changes_workspace,
            prepare_arguments=self.prepare_arguments,
        )


_file_locks: dict[Path, asyncio.Lock] = {}


def _validate_local_read_path(path: Path) -> None:
    if not path.exists():
        raise ToolInputError(f"File not found: {path}")
    if path.is_dir():
        raise ToolInputError(f"Path is a directory: {path}")


def _local_file_size(path: Path) -> int:
    return path.stat().st_size


def _local_read_prefix(path: Path, limit: int) -> bytes:
    with path.open("rb") as file:
        return file.read(limit)


DEFAULT_READ_OPERATIONS = ReadOperations(
    validate_path=_validate_local_read_path,
    read_bytes=Path.read_bytes,
    size_bytes=_local_file_size,
    read_prefix=_local_read_prefix,
)


def bash_default_timeout_seconds() -> float:
    """Return the bash default timeout: `TAU_BASH_TIMEOUT_SECONDS` if set, else 600 s.

    An unset, empty, non-numeric or non-positive value falls back to
    `DEFAULT_BASH_TIMEOUT_SECONDS` rather than unbounding or breaking the tool.
    """
    raw = os.environ.get(BASH_TIMEOUT_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_BASH_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_BASH_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_BASH_TIMEOUT_SECONDS


def create_coding_tools(
    *,
    cwd: str | Path | None = None,
    shell_command_prefix: str | None = None,
    image_support: ImageSupportState | None = None,
) -> list[AgentTool]:
    """Create the default coding-tool set for a local project.

    The returned tools are ordered as `read`, `write`, `edit`, and `bash`.
    Relative paths used with those tools are resolved against `cwd`; when `cwd`
    is omitted, the process current working directory at factory-call time is
    used. The tools share per-path write/edit locks within this process so
    concurrent mutations of the same file do not interleave. When configured,
    `shell_command_prefix` is prepended to every bash tool command. The bash
    tool's default timeout is `bash_default_timeout_seconds()`.
    """
    root = Path.cwd() if cwd is None else Path(cwd)
    return [
        create_read_tool(cwd=root, image_support=image_support),
        create_write_tool(cwd=root),
        create_edit_tool(cwd=root),
        create_bash_tool(
            cwd=root,
            shell_command_prefix=shell_command_prefix,
            default_timeout=bash_default_timeout_seconds(),
        ),
    ]


def create_read_tool_definition(
    *,
    cwd: str | Path | None = None,
    operations: ReadOperations | None = None,
    image_support: ImageSupportState | None = None,
) -> ToolDefinition:
    """Create a definition for the `read` tool.

    The tool reads a file resolved relative to `cwd` unless an absolute path is
    supplied. Text files are decoded as UTF-8 and may be sliced with optional
    1-indexed `offset` and positive integer `limit` arguments. Each line is
    returned as anchored rows, `N#hhh| text` with long lines wrapped into
    `N.k| ` rows (see `tau_coding.line_anchors`). Returned text is
    truncated to `DEFAULT_MAX_OUTPUT_LINES` lines or `DEFAULT_MAX_OUTPUT_BYTES`
    bytes, whichever comes first, and continuation hints are appended when more
    lines remain. Supported images (`jpg`, `png`, `gif`, `webp`, and `bmp`) are
    detected from file content and returned as provider-neutral image blocks.
    Images are validated and resized or converted when needed to fit inline limits.

    The executor raises `ToolInputError` for invalid arguments, missing files,
    directories, and offsets beyond the end of the file. Successful results
    include the resolved path and truncation metadata in `data`.
    """
    root = Path.cwd() if cwd is None else Path(cwd)
    read_operations = operations or DEFAULT_READ_OPERATIONS

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
    ) -> AgentToolResult:
        del signal
        raw_path = _str_arg(arguments, "path")
        path = _path_arg(arguments, "path", cwd=root)
        offset = _optional_int_arg(arguments, "offset")
        limit = _optional_int_arg(arguments, "limit")

        if offset is not None and offset < 0:
            raise ToolInputError("offset must be at least 0")
        if limit is not None and limit < 1:
            raise ToolInputError("limit must be at least 1")
        read_operations.validate_path(path)
        if read_operations.size_bytes is not None and read_operations.read_prefix is not None:
            source_size = read_operations.size_bytes(path)
            if source_size > DEFAULT_MAX_SOURCE_IMAGE_BYTES:
                prefix = read_operations.read_prefix(path, IMAGE_SNIFF_BYTES)
                image_family = detect_image_family_mime_type(prefix)
                if image_family is not None:
                    reason = unsupported_image_reason(prefix) or (
                        f"source is {format_size(source_size)}, exceeding the "
                        f"{format_size(DEFAULT_MAX_SOURCE_IMAGE_BYTES)} processing limit"
                    )
                    return _omitted_image_result(
                        path=path,
                        source_mime_type=image_family,
                        source_bytes=source_size,
                        reason=reason,
                    )

        data = read_operations.read_bytes(path)
        unsupported_reason = unsupported_image_reason(data)
        if unsupported_reason is not None:
            image_family = detect_image_family_mime_type(data)
            assert image_family is not None
            return _omitted_image_result(
                path=path,
                source_mime_type=image_family,
                source_bytes=len(data),
                reason=unsupported_reason,
            )

        source_mime_type = detect_supported_image_mime_type(data)
        if source_mime_type is not None:
            if image_support is not None and image_support.supported is False:
                return _omitted_image_result(
                    path=path,
                    source_mime_type=source_mime_type,
                    source_bytes=len(data),
                    reason=(
                        "current model does not support image input. Image contents are "
                        "unavailable; do not infer or describe them. Ask the user to switch "
                        "to a vision-capable model"
                    ),
                )
            processed = await asyncio.to_thread(process_image, data, source_mime_type)
            image_details: dict[str, JSONValue] = {
                "path": str(path),
                "source_mime_type": source_mime_type,
                "bytes": len(data),
            }
            if isinstance(processed, ImageProcessingFailure):
                return AgentToolResult(
                    content=[
                        TextContent(
                            text=(
                                f"Read image file [{source_mime_type}]\n"
                                f"[Image omitted: {processed.message}.]"
                            )
                        )
                    ],
                    details=image_details,
                )

            image_details.update(
                {
                    "mime_type": processed.mime_type,
                    "processed_bytes": len(processed.data),
                    "width": processed.width,
                    "height": processed.height,
                }
            )
            note_lines = "".join(f"\n[{note}]" for note in processed.notes)
            return AgentToolResult(
                content=[
                    TextContent(text=f"Read image file [{processed.mime_type}]{note_lines}"),
                    ImageContent(data=_base64_text(processed.data), mime_type=processed.mime_type),
                ],
                details=image_details,
            )

        text = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        all_lines = text.split("\n")
        if len(all_lines) > 1 and all_lines[-1] == "":
            # A final newline ends the last line; it does not start another.
            all_lines.pop()
        start_line = 0 if offset is None or offset == 0 else offset - 1
        if start_line >= len(all_lines):
            raise ToolInputError(
                f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)"
            )

        user_limited_lines: int | None = None
        if limit is not None:
            end_line = min(start_line + limit, len(all_lines))
            user_limited_lines = end_line - start_line
        else:
            end_line = len(all_lines)

        start_display = start_line + 1
        truncation = truncate_anchored_rows(
            all_lines[start_line:end_line] if text else [], first_line=start_display
        )
        details: dict[str, JSONValue] = {"path": str(path), "truncation": truncation.to_json()}

        if truncation.first_line_exceeds_limit:
            rows = "\n".join(line_rows(start_display, all_lines[start_line]))
            first_line_size = format_size(len(rows.encode()))
            output = (
                f"[Line {start_display} is {first_line_size}, exceeds "
                f"{format_size(DEFAULT_MAX_OUTPUT_BYTES)} limit. Use bash: sed -n "
                f"'{start_display}p' {raw_path} | head -c {DEFAULT_MAX_OUTPUT_BYTES}]"
            )
        elif truncation.truncated:
            end_display = start_display + truncation.output_lines - 1
            next_offset = end_display + 1
            output = truncation.content
            if truncation.truncated_by == "lines":
                output += (
                    f"\n\n[Showing lines {start_display}-{end_display} of {len(all_lines)}. "
                    f"Use offset={next_offset} to continue.]"
                )
            else:
                output += (
                    f"\n\n[Showing lines {start_display}-{end_display} of {len(all_lines)} "
                    f"({format_size(DEFAULT_MAX_OUTPUT_BYTES)} limit). "
                    f"Use offset={next_offset} to continue.]"
                )
        elif user_limited_lines is not None and start_line + user_limited_lines < len(all_lines):
            remaining = len(all_lines) - (start_line + user_limited_lines)
            next_offset = start_line + user_limited_lines + 1
            output = (
                f"{truncation.content}\n\n[{remaining} more lines in file. "
                f"Use offset={next_offset} to continue.]"
            )
        else:
            output = truncation.content

        return AgentToolResult(
            content=[TextContent(text=output)],
            details=details,
        )

    return ToolDefinition(
        name="read",
        description=(
            "Read the contents of a file. Supports text files and images "
            "(jpg, png, gif, webp, bmp). "
            "Images are sent to vision-capable models as attachments. For text files, output is "
            "truncated to "
            f"{DEFAULT_MAX_OUTPUT_LINES} lines or {DEFAULT_MAX_OUTPUT_BYTES // 1024}KB "
            "(whichever is hit first). Use offset/limit for large files. When you need the "
            "full file, continue with offset until complete. Each text line is shown as "
            "`N#hhh| text`: N is its line number in the file and hhh a short hash of the "
            "line, so `N#hhh` names the line. A line over "
            f"{WRAP_WIDTH} characters is wrapped, its further rows numbered `N.2| `, `N.3| ` "
            "and so on. These prefixes are not file content; offset and limit count file lines."
        ),
        prompt_snippet="Read file contents",
        prompt_guidelines=("Use read to examine files instead of cat or sed.",),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
                "offset": {"type": "integer", "description": "Line number to start reading from"},
                "limit": {"type": "integer", "description": "Maximum number of lines to read"},
            },
            "required": ["path"],
        },
        executor=execute,
    )


def _omitted_image_result(
    *,
    path: Path,
    source_mime_type: str,
    source_bytes: int,
    reason: str,
) -> AgentToolResult:
    return AgentToolResult(
        content=[
            TextContent(text=f"Read image file [{source_mime_type}]\n[Image omitted: {reason}.]")
        ],
        details={
            "path": str(path),
            "source_mime_type": source_mime_type,
            "bytes": source_bytes,
        },
    )


def create_read_tool(
    *,
    cwd: str | Path | None = None,
    operations: ReadOperations | None = None,
    image_support: ImageSupportState | None = None,
) -> AgentTool:
    """Create an `AgentTool` for reading UTF-8 text files and supported images."""
    return create_read_tool_definition(
        cwd=cwd,
        operations=operations,
        image_support=image_support,
    ).to_agent_tool()


def create_write_tool_definition(*, cwd: str | Path | None = None) -> ToolDefinition:
    """Create a definition for the `write` tool.

    The tool writes the supplied string `content` to `path`, resolving relative
    paths against `cwd`. Parent directories are created automatically and any
    existing file is overwritten. Writes use UTF-8 text encoding and are guarded
    by a per-path async lock so multiple writes/edits to the same resolved file
    are serialized within this process.

    The executor raises `ToolInputError` when `path` or `content` has the wrong
    type. Successful results include the resolved path and number of characters
    written in `data`.
    """
    root = Path.cwd() if cwd is None else Path(cwd)

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
    ) -> AgentToolResult:
        del signal
        path = _path_arg(arguments, "path", cwd=root)
        content = _str_arg(arguments, "content")

        async with _file_lock(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        return AgentToolResult(
            content=[TextContent(text=f"Successfully wrote to {path}.")],
            details={"path": str(path), "characters": len(content)},
        )

    return ToolDefinition(
        name="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories."
        ),
        prompt_snippet="Create or overwrite files",
        prompt_guidelines=("Use write only for new files or complete rewrites.",),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
        },
        executor=execute,
        changes_workspace=True,
    )


def create_write_tool(*, cwd: str | Path | None = None) -> AgentTool:
    """Create an `AgentTool` for creating or overwriting UTF-8 text files."""
    return create_write_tool_definition(cwd=cwd).to_agent_tool()


def create_edit_tool_definition(*, cwd: str | Path | None = None) -> ToolDefinition:
    """Create a definition for the `edit` tool.

    The tool applies one or more exact text replacements to a single UTF-8 file
    resolved relative to `cwd`. Each edit item contains `oldText` and `newText`.
    Every `oldText` must be non-empty, must occur exactly once in the original
    file, and must not overlap another edit span. All replacements are validated
    before writing, so the file is left unchanged if any edit fails.

    File content and edit text are normalized to LF for matching, then the
    original file's dominant line ending is restored after replacement. UTF-8
    byte-order marks are preserved. The executor also accepts legacy top-level
    `oldText`/`newText` arguments and JSON-string `edits` values by normalizing
    them into the canonical edits list. An `oldText` that does not match but was
    copied from `read` output with its row prefixes on every line is retried
    without them (`_without_row_prefixes`).

    Successful results include the resolved path, edit count, an ndiff-style
    diff, a unified patch, and the first changed line in `data`.
    """
    root = Path.cwd() if cwd is None else Path(cwd)

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
    ) -> AgentToolResult:
        del signal
        prepared = _prepare_edit_arguments(arguments)
        path = _path_arg(prepared, "path", cwd=root)
        edits = _edits_arg(prepared)

        if not path.exists():
            raise ToolInputError(f"Could not edit file: {path}. File not found.")
        if path.is_dir():
            raise ToolInputError(f"Could not edit file: {path}. Path is a directory.")

        async with _file_lock(path):
            raw_content = path.read_text(encoding="utf-8")
            bom, content = _strip_bom(raw_content)
            original_ending = detect_line_ending(content)
            normalized = normalize_to_lf(content)
            base_content, new_content = apply_edits_to_normalized_content(
                normalized, edits, str(path)
            )
            final_content = bom + restore_line_endings(new_content, original_ending)
            path.write_text(final_content, encoding="utf-8")

        diff_text, first_changed_line = generate_diff_string(base_content, new_content)
        patch = generate_unified_patch(str(path), base_content, new_content)
        return AgentToolResult(
            content=[TextContent(text=f"Successfully replaced {len(edits)} block(s) in {path}.")],
            details={
                "path": str(path),
                "edits": len(edits),
                "diff": diff_text,
                "patch": patch,
                "first_changed_line": first_changed_line,
            },
        )

    return ToolDefinition(
        name="edit",
        description=(
            "Edit a single file using exact text replacement. Every edits[].oldText must match "
            "a unique, non-overlapping region of the original file. If two changes affect the "
            "same block or nearby lines, merge them into one edit instead of emitting overlapping "
            "edits. Do not include large unchanged regions just to connect distant changes."
        ),
        prompt_snippet=(
            "Make precise file edits with exact text replacement, including multiple disjoint "
            "edits in one call"
        ),
        prompt_guidelines=(
            "Use edit for precise changes (edits[].oldText must match exactly)",
            "When changing multiple separate locations in one file, use one edit call with "
            "multiple entries in edits[] instead of multiple edit calls",
            "Each edits[].oldText is matched against the original file, not after earlier "
            "edits are applied. Do not emit overlapping or nested edits. Merge nearby "
            "changes into one edit.",
            "Keep edits[].oldText as small as possible while still being unique in the file. "
            "Do not pad with large unchanged regions.",
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit"},
                "edits": {
                    "type": "array",
                    "description": "One or more targeted replacements.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {"type": "string"},
                            "newText": {"type": "string"},
                        },
                        "required": ["oldText", "newText"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["path", "edits"],
            "additionalProperties": False,
        },
        executor=execute,
        changes_workspace=True,
        prepare_arguments=_prepare_edit_arguments,
    )


def create_edit_tool(*, cwd: str | Path | None = None) -> AgentTool:
    """Create an `AgentTool` for exact, validated text replacement in one file."""
    return create_edit_tool_definition(cwd=cwd).to_agent_tool()


def create_bash_tool_definition(
    *,
    cwd: str | Path | None = None,
    shell_command_prefix: str | None = None,
    default_timeout: float | None = DEFAULT_BASH_TIMEOUT_SECONDS,
) -> ToolDefinition:
    """Create a definition for the `bash` tool.

    The tool runs a shell command with `cwd` as the subprocess working
    directory, disconnects stdin, and combines stdout and stderr into one UTF-8
    decoded output stream. Disconnecting stdin keeps interactive programs from
    competing with Tau for terminal input. The optional `timeout` argument must
    be positive when supplied; without it the command gets `default_timeout`
    seconds, and `None` there leaves it unbounded. On timeout, POSIX commands
    are started in a new session and the entire process group is killed so
    shell children from pipelines or compound commands do not continue running;
    non-POSIX platforms fall back to killing the direct subprocess.

    Output is returned whole; a coding session windows an oversized result and
    saves the full text (see `tau_coding.tool_buffers`). Successful and failed
    command results both include exit code, timeout state and duration.
    """
    root = Path.cwd() if cwd is None else Path(cwd)
    prefix = shell_command_prefix.strip() if shell_command_prefix else None
    if default_timeout is not None and default_timeout <= 0:
        raise ValueError("default_timeout must be greater than 0")

    async def execute(
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
    ) -> AgentToolResult:
        command = _str_arg(arguments, "command")
        shell_command = _prefixed_shell_command(command, prefix)
        timeout = _optional_float_arg(arguments, "timeout")
        if timeout is not None and timeout <= 0:
            raise ToolInputError("timeout must be greater than 0")
        timeout_defaulted = timeout is None and default_timeout is not None
        if timeout is None:
            timeout = default_timeout
        if signal is not None and signal.is_cancelled():
            raise ToolInputError("Command cancelled")

        start = monotonic()
        if os.name == "posix":
            process = await asyncio.create_subprocess_shell(
                shell_command,
                cwd=root,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
                executable="bash" if prefix else None,
            )
        else:
            process = await asyncio.create_subprocess_shell(
                shell_command,
                cwd=root,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        output_bytes, _stderr, timed_out, cancelled = await _communicate_with_cancellation(
            process,
            timeout=timeout,
            signal=signal,
        )

        # Not cut here: the session buffers any oversized tool result, this one
        # included, and keeps the whole of it in a file the model can page.
        output_text = output_bytes.decode(errors="replace") or NO_OUTPUT

        exit_code = process.returncode
        status: str | None = None
        if timed_out:
            status = (
                f"Command timed out after {timeout:g} seconds" if timeout else "Command timed out"
            )
            if timeout_defaulted:
                status += (
                    " (the default; pass a larger `timeout` if the command"
                    " is expected to run longer)"
                )
        elif cancelled:
            status = "Command cancelled"
        elif exit_code not in (0, None):
            status = f"Command exited with code {exit_code}"
        if status:
            output_text = append_status_block(output_text, status)

        return AgentToolResult(
            content=[TextContent(text=output_text)],
            details={
                "command": command,
                "exit_code": exit_code,
                "timed_out": timed_out,
                "timeout_seconds": timeout,
                "cancelled": cancelled,
                "duration_seconds": round(monotonic() - start, 3),
                "shell_command_prefix_applied": prefix is not None,
            },
        )

    return ToolDefinition(
        name="bash",
        description=(
            "Execute a bash command in the current working directory. Returns stdout and stderr. "
            "In a session, long output is shortened to its end and the full output is saved "
            "to a file named in the result. Optionally provide a timeout in seconds"
            + (
                f"; without one the command is killed after {default_timeout:g} seconds."
                if default_timeout is not None
                else "."
            )
        ),
        prompt_snippet="Execute bash commands (ls, grep, find, etc.)",
        prompt_guidelines=(
            "When using bash, include a brief present-participle description of the "
            "command's purpose (for example, 'Running tests').",
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Bash command to execute"},
                "description": {
                    "type": "string",
                    "description": (
                        "Brief present-participle summary of the command's purpose, such as "
                        "'Running tests' or 'Validating and committing changes'"
                    ),
                },
                "timeout": {
                    "type": "number",
                    "description": (
                        f"Timeout in seconds (optional, default {default_timeout:g})"
                        if default_timeout is not None
                        else "Timeout in seconds (optional, no default timeout)"
                    ),
                },
            },
            # `description` only labels the call, so a call without it still runs.
            "required": ["command"],
        },
        executor=execute,
    )


def create_bash_tool(
    *,
    cwd: str | Path | None = None,
    shell_command_prefix: str | None = None,
    default_timeout: float | None = DEFAULT_BASH_TIMEOUT_SECONDS,
) -> AgentTool:
    """Create an `AgentTool` for executing shell commands with captured output."""
    return create_bash_tool_definition(
        cwd=cwd,
        shell_command_prefix=shell_command_prefix,
        default_timeout=default_timeout,
    ).to_agent_tool()


def _prefixed_shell_command(command: str, prefix: str | None) -> str:
    """Return a shell command with an opt-in setup prefix applied."""
    if prefix is None:
        return command
    return f"{prefix}\n{command}"


def format_size(bytes_count: int) -> str:
    if bytes_count < 1024:
        return f"{bytes_count}B"
    if bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f}KB"
    return f"{bytes_count / (1024 * 1024):.1f}MB"


def append_status_block(text: str, status: str) -> str:
    """Append command status text after a blank line when output already exists."""
    return f"{text}\n\n{status}" if text else status


async def _communicate_with_cancellation(
    process: asyncio.subprocess.Process,
    *,
    timeout: float | None,
    signal: ToolCancellationToken | None,
) -> tuple[bytes, bytes | None, bool, bool]:
    communicate = asyncio.create_task(process.communicate())
    cancel_watch: asyncio.Task[None] | None = None
    try:
        wait_for: set[asyncio.Task[Any]] = {communicate}
        if signal is not None:
            cancel_watch = asyncio.create_task(_wait_for_cancel(signal))
            wait_for.add(cancel_watch)

        done, _pending = await asyncio.wait(
            wait_for,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if communicate in done:
            output_bytes, stderr = communicate.result()
            return output_bytes, stderr, False, False

        cancelled = cancel_watch is not None and cancel_watch in done
        _kill_process_tree(process)
        try:
            output_bytes, stderr = await communicate
        except asyncio.CancelledError:
            output_bytes = b""
            stderr_result: bytes | None = None
        else:
            stderr_result = stderr
        return output_bytes, stderr_result, not cancelled, cancelled
    except asyncio.CancelledError:
        _kill_process_tree(process)
        if not communicate.done():
            communicate.cancel()
        raise
    finally:
        if cancel_watch is not None:
            cancel_watch.cancel()


async def _wait_for_cancel(signal: ToolCancellationToken) -> None:
    while not signal.is_cancelled():
        await asyncio.sleep(0.05)


def detect_line_ending(content: str) -> str:
    crlf_index = content.find("\r\n")
    lf_index = content.find("\n")
    if lf_index == -1 or crlf_index == -1:
        return "\n"
    return "\r\n" if crlf_index < lf_index else "\n"


def normalize_to_lf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


def apply_edits_to_normalized_content(
    normalized_content: str,
    edits: list[dict[str, str]],
    path: str,
) -> tuple[str, str]:
    normalized_edits = [
        {"oldText": normalize_to_lf(edit["oldText"]), "newText": normalize_to_lf(edit["newText"])}
        for edit in edits
    ]
    for index, edit in enumerate(normalized_edits):
        if not edit["oldText"]:
            raise ToolInputError(_empty_old_text_error(path, index, len(normalized_edits)))

    matches: list[tuple[int, int, str]] = []
    for index, edit in enumerate(normalized_edits):
        old_text = edit["oldText"]
        occurrences = _count_occurrences(normalized_content, old_text)
        if occurrences == 0:
            stripped = _without_row_prefixes(edit, normalized_content)
            if stripped is None:
                raise ToolInputError(_not_found_error(path, index, len(normalized_edits)))
            edit = stripped
            old_text = edit["oldText"]
            occurrences = _count_occurrences(normalized_content, old_text)
        if occurrences > 1:
            raise ToolInputError(_duplicate_error(path, index, len(normalized_edits), occurrences))
        start = normalized_content.index(old_text)
        matches.append((start, start + len(old_text), edit["newText"]))

    _validate_non_overlapping(matches)
    new_content = normalized_content
    for start, end, new_text in sorted(matches, reverse=True):
        new_content = f"{new_content[:start]}{new_text}{new_content[end:]}"
    if new_content == normalized_content:
        raise ToolInputError(_no_change_error(path, len(normalized_edits)))
    return normalized_content, new_content


def _without_row_prefixes(edit: dict[str, str], content: str) -> dict[str, str] | None:
    """`edit` with the `read` row prefixes taken off its old text, if that matches.

    A model often copies an edit's old text from `read` output with the row
    prefixes (`12#3fa| `, `12.2| `) still on. When every line of the old text
    carries one and the text without them occurs in `content`, the edit is
    retried without them; the new text loses its prefixes only when every one
    of its lines carries one too, so new text that merely starts like a row is
    written as given.
    """
    old_text = strip_row_prefixes(edit["oldText"])
    if not old_text or _count_occurrences(content, old_text) == 0:
        return None
    new_text = strip_row_prefixes(edit["newText"]) if edit["newText"] else None
    return {"oldText": old_text, "newText": edit["newText"] if new_text is None else new_text}


def generate_diff_string(old: str, new: str) -> tuple[str, int | None]:
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    diff = "\n".join(difflib.ndiff(old_lines, new_lines))
    first_changed_line: int | None = None
    new_line_number = 0
    for line in difflib.ndiff(old_lines, new_lines):
        if line.startswith("  "):
            new_line_number += 1
        elif line.startswith("+"):
            new_line_number += 1
            if first_changed_line is None:
                first_changed_line = new_line_number
        elif line.startswith("-") and first_changed_line is None:
            first_changed_line = max(new_line_number + 1, 1)
    return diff, first_changed_line


def generate_unified_patch(path: str, old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=path,
            tofile=path,
        )
    )


def truncate_anchored_rows(
    lines: list[str],
    *,
    first_line: int = 1,
    max_lines: int = DEFAULT_MAX_OUTPUT_LINES,
    max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> TruncationResult:
    """`lines` as anchored rows (see `tau_coding.line_anchors`), cut at a limit.

    The cut always falls between file lines, never inside a wrapped one, so
    `output_lines` and `total_lines` count file lines, not rows; `max_bytes`
    counts the rows as shown, prefixes included.
    """
    rendered = ["\n".join(line_rows(number, line)) for number, line in enumerate(lines, first_line)]
    total_lines = len(rendered)
    content = "\n".join(rendered)
    total_bytes = len(content.encode())
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return _truncation_result(
            content, False, None, total_lines, total_bytes, total_lines, total_bytes
        )
    if rendered and len(rendered[0].encode()) > max_bytes:
        return _truncation_result(
            "", True, "bytes", total_lines, total_bytes, 0, 0, first_line=True
        )
    output_lines: list[str] = []
    output_bytes = 0
    truncated_by = "lines"
    for index, block in enumerate(rendered[:max_lines]):
        block_bytes = len(block.encode()) + (1 if index > 0 else 0)
        if output_bytes + block_bytes > max_bytes:
            truncated_by = "bytes"
            break
        output_lines.append(block)
        output_bytes += block_bytes
    output = "\n".join(output_lines)
    return _truncation_result(
        output,
        True,
        truncated_by,
        total_lines,
        total_bytes,
        len(output_lines),
        len(output.encode()),
    )


def _truncation_result(
    content: str,
    truncated: bool,
    truncated_by: str | None,
    total_lines: int,
    total_bytes: int,
    output_lines: int,
    output_bytes: int,
    *,
    first_line: bool = False,
) -> TruncationResult:
    return TruncationResult(
        content=content,
        truncated=truncated,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=output_lines,
        output_bytes=output_bytes,
        first_line_exceeds_limit=first_line,
        max_lines=DEFAULT_MAX_OUTPUT_LINES,
        max_bytes=DEFAULT_MAX_OUTPUT_BYTES,
    )


def _str_arg(arguments: Mapping[str, JSONValue], name: str) -> str:
    """`arguments[name]` as a string; an absent one is named as missing."""
    if name not in arguments:
        carried = ", ".join(f"`{key}`" for key in arguments) or "no arguments"
        raise ToolInputError(
            f"`{name}` is missing: the call carried {carried}, and this tool needs "
            f"`{name}` too; nothing was run. Send the call again with `{name}`."
        )
    value = arguments[name]
    if not isinstance(value, str):
        raise ToolInputError(f"{name} must be a string")
    return value


def _path_arg(arguments: Mapping[str, JSONValue], name: str, *, cwd: Path) -> Path:
    value = _str_arg(arguments, name)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return path


def _optional_int_arg(arguments: Mapping[str, JSONValue], name: str) -> int | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ToolInputError(f"{name} must be an integer")
    return value


def _optional_float_arg(arguments: Mapping[str, JSONValue], name: str) -> float | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise ToolInputError(f"{name} must be a number")
    return float(value)


def _prepare_edit_arguments(arguments: object) -> Mapping[str, JSONValue]:
    prepared: dict[str, JSONValue] = dict(arguments) if isinstance(arguments, Mapping) else {}
    edits_value = prepared.get("edits")
    if isinstance(edits_value, str):
        try:
            parsed = json.loads(edits_value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            prepared["edits"] = parsed

    old_text = prepared.get("oldText")
    new_text = prepared.get("newText")
    if isinstance(old_text, str) and isinstance(new_text, str):
        edits = prepared.get("edits")
        edit_list = edits if isinstance(edits, list) else []
        prepared["edits"] = [*edit_list, {"oldText": old_text, "newText": new_text}]
        prepared.pop("oldText", None)
        prepared.pop("newText", None)
    _hoist_shared_edit_path(prepared)
    # Refused here rather than in the tool: the loop checks prepared arguments
    # against the schema before the tool runs, and its generic "missing
    # required" would not show where `path` goes.
    if "path" not in prepared:
        raise ToolInputError(_edit_without_path_error(prepared))
    edits = prepared.get("edits")
    if isinstance(edits, list):
        prepared["edits"] = [
            {key: value for key, value in item.items() if key != "path"}
            if isinstance(item, dict)
            else item
            for item in edits
        ]
    return prepared


#: The edit call's form, as the refusals of a misplaced `path` spell it out.
EDIT_FORM_EXAMPLE = '{"path": "%s", "edits": [{"oldText": "...", "newText": "..."}]}'


def _hoist_shared_edit_path(prepared: dict[str, JSONValue]) -> None:
    """Move a `path` every edit object shares to the top level, where it belongs.

    Only a call with no top-level `path`, whose edits are all objects naming the
    same one file, is changed.
    """
    edits = prepared.get("edits")
    if "path" in prepared or not isinstance(edits, list) or not edits:
        return
    if not all(isinstance(item, dict) for item in edits):
        return
    paths = {item.get("path") for item in edits if isinstance(item, dict)}
    if len(paths) != 1:
        return
    (shared,) = paths
    if not isinstance(shared, str) or not shared:
        return
    prepared["path"] = shared
    prepared["edits"] = [
        {key: value for key, value in item.items() if key != "path"}
        for item in edits
        if isinstance(item, dict)
    ]


def _edit_without_path_error(prepared: Mapping[str, JSONValue]) -> str:
    """Why an edit call has no top-level `path`, and the form that has one.

    A bare "`path` is missing" doesn't show where it goes when the path was sent
    inside `edits`, so the model tends to resend the same call.
    """
    edits = prepared.get("edits")
    items = edits if isinstance(edits, list) else []
    strays = [item for item in items if not isinstance(item, dict)]
    named = [item.get("path") for item in items if isinstance(item, dict) and "path" in item]
    if strays:
        shown = ", ".join(f"`{json.dumps(item)}`" for item in strays[:4])
        problem = (
            f"`edits` holds {len(strays)} value(s) that are not oldText/newText objects "
            f"({shown}), and the call has no top-level `path`"
        )
    elif len(set(map(str, named))) > 1:
        problem = "the edits name different files in `path`; one edit call changes one file"
    else:
        carried = ", ".join(f"`{key}`" for key in prepared) or "no arguments"
        problem = f"`path` is missing: the call carried {carried}"
    guess = next((item for item in strays if isinstance(item, str)), None)
    if guess is None:
        guess = next((item for item in named if isinstance(item, str)), "path/to/file")
    example = EDIT_FORM_EXAMPLE % guess.replace('"', '\\"')
    return (
        f"{problem}; nothing was run. Put `path` at the top level, once, beside an "
        f"`edits` list holding only oldText/newText objects: {example}"
    )


def _edits_arg(arguments: Mapping[str, JSONValue]) -> list[dict[str, str]]:
    value = arguments.get("edits")
    if not isinstance(value, list) or not value:
        raise ToolInputError(
            "Edit tool input is invalid. edits must contain at least one replacement."
        )

    edits: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ToolInputError(f"edits[{index}] must be an object")
        old_text = item.get("oldText")
        new_text = item.get("newText")
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            raise ToolInputError(
                f"edits[{index}].oldText and edits[{index}].newText must be strings"
            )
        edits.append({"oldText": old_text, "newText": new_text})
    return edits


def _validate_non_overlapping(spans: list[tuple[int, int, str]]) -> None:
    previous_end = -1
    for start, end, _new_text in sorted(spans):
        if start < previous_end:
            raise ToolInputError("Edits must not overlap")
        previous_end = end


def _count_occurrences(content: str, text: str) -> int:
    count = 0
    start = 0
    while True:
        index = content.find(text, start)
        if index == -1:
            return count
        count += 1
        start = index + len(text)


def _strip_bom(content: str) -> tuple[str, str]:
    return (UTF8_BOM, content[1:]) if content.startswith(UTF8_BOM) else ("", content)


def _not_found_error(path: str, edit_index: int, total_edits: int) -> str:
    if total_edits == 1:
        return (
            f"Could not find the exact text in {path}. The old text must match exactly "
            "including all whitespace and newlines."
        )
    return (
        f"Could not find edits[{edit_index}] in {path}. The oldText must match exactly "
        "including all whitespace and newlines."
    )


def _duplicate_error(path: str, edit_index: int, total_edits: int, occurrences: int) -> str:
    if total_edits == 1:
        return (
            f"Found {occurrences} occurrences of the text in {path}. The text must be unique. "
            "Please provide more context to make it unique."
        )
    return (
        f"Found {occurrences} occurrences of edits[{edit_index}] in {path}. "
        "Each oldText must be unique. Please provide more context to make it unique."
    )


def _empty_old_text_error(path: str, edit_index: int, total_edits: int) -> str:
    if total_edits == 1:
        return f"oldText must not be empty in {path}."
    return f"edits[{edit_index}].oldText must not be empty in {path}."


def _no_change_error(path: str, total_edits: int) -> str:
    if total_edits == 1:
        return (
            f"No changes made to {path}. The replacement produced identical content. "
            "This might indicate an issue with special characters or the text not existing "
            "as expected."
        )
    return f"No changes made to {path}. The replacements produced identical content."


def _base64_text(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    if os.name == "posix":
        # `getattr` keeps mypy happy on the Windows stubs (see issue #513).
        killpg = getattr(os, "killpg")  # noqa: B009
        sigkill = getattr(signal, "SIGKILL")  # noqa: B009
        try:
            killpg(process.pid, sigkill)
        except ProcessLookupError:
            return
    else:
        try:
            process.kill()
        except ProcessLookupError:
            return


class _FileLockContext:
    def __init__(self, path: Path) -> None:
        self._path = path.resolve()
        self._lock: asyncio.Lock | None = None

    async def __aenter__(self) -> None:
        lock = _file_locks.setdefault(self._path, asyncio.Lock())
        self._lock = lock
        await lock.acquire()

    async def __aexit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        if self._lock is not None:
            self._lock.release()


def _file_lock(path: Path) -> _FileLockContext:
    return _FileLockContext(path)
