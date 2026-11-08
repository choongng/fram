"""The stream stall timer's per-attempt ramp (tau_ai.stall) and retry count."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import pytest

from tau_ai.env import DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES
from tau_ai.stall import (
    STREAM_STALL_CAP_SECONDS,
    STREAM_STALL_START_SECONDS,
    StreamStallError,
    ramped_stream_stall_seconds,
    stall_guarded,
    stream_stall_seconds,
)


def test_ramp_doubles_per_attempt_to_the_cap() -> None:
    assert [ramped_stream_stall_seconds(attempt) for attempt in range(6)] == [
        60.0,
        120.0,
        240.0,
        480.0,
        600.0,
        600.0,
    ]
    assert STREAM_STALL_START_SECONDS == 60.0
    assert STREAM_STALL_CAP_SECONDS == 600.0


def test_ramp_is_bounded_for_negative_attempts() -> None:
    assert ramped_stream_stall_seconds(-3) == 60.0


def test_default_retries_cover_the_ramp() -> None:
    """Five tries: the ramp runs 60 -> 120 -> 240 -> 480 -> 600 s (capped)."""
    assert DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES == 4
    assert (
        len({stream_stall_seconds(a) for a in range(DEFAULT_OPENAI_COMPATIBLE_MAX_RETRIES + 1)})
        == 5
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("off", None),
        ("0", None),
        ("-5", None),
        ("300", 300.0),
    ],
)
def test_environment_override(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: float | None, attempt: int = 2
) -> None:
    monkeypatch.setenv("TAU_STREAM_STALL_SECONDS", raw)
    # An explicit value applies as-is to every attempt, overriding the ramp.
    assert stream_stall_seconds(attempt) == expected


def test_environment_override_applies_to_every_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAU_STREAM_STALL_SECONDS", "300")
    assert stream_stall_seconds(0) == stream_stall_seconds(4) == 300.0


def test_unset_and_garbage_fall_back_to_the_ramp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAU_STREAM_STALL_SECONDS", raising=False)
    assert stream_stall_seconds(3) == 480.0
    monkeypatch.setenv("TAU_STREAM_STALL_SECONDS", "not-a-number")
    assert stream_stall_seconds(3) == 480.0


async def _paced(*gaps: float) -> AsyncGenerator[int]:
    for index, gap in enumerate(gaps):
        await asyncio.sleep(gap)
        yield index


@pytest.mark.anyio
async def test_stall_guard_ends_a_silent_stream() -> None:
    with pytest.raises(StreamStallError):
        _ = [item async for item in stall_guarded(_paced(0.0, 0.2), 0.02)]


@pytest.mark.anyio
async def test_stall_guard_resets_with_every_event() -> None:
    items = [item async for item in stall_guarded(_paced(0.03, 0.03, 0.03), 0.08)]

    assert items == [0, 1, 2]
