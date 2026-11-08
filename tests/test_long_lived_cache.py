"""A long-lived (interactive) session keeps the long prompt cache for its life.

Anthropic serves a 1-hour request only from a prefix written at 1 hour, so the
TTL is chosen per conversation, not per request: the interactive session asks
for it on every request, a worker never does, and replacing the session in
place (``/new``, ``/resume``) keeps the choice.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tau_agent.session import JsonlSessionStorage
from tau_ai import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig
from tau_coding.paths import TauPaths
from tau_coding.resources import TauResourcePaths
from tau_coding.session_manager import SessionManager


async def _session(tmp_path: Path, *, long_lived: bool) -> tuple[CodingSession, SessionManager]:
    tau_home = tmp_path / "tau-home"
    tau_home.mkdir()
    manager = SessionManager(TauPaths(home=tau_home, agents_home=tmp_path / "agents-home"))
    record = manager.create_session(cwd=tmp_path, model="fake")
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=FakeProvider([]),
            model="fake",
            system="You are Tau.",
            storage=JsonlSessionStorage(record.path),
            cwd=tmp_path,
            session_id=record.id,
            session_manager=manager,
            resource_paths=TauResourcePaths(root=tau_home, agents_root=None),
            long_lived=long_lived,
        )
    )
    return session, manager


@pytest.mark.anyio
@pytest.mark.parametrize("long_lived", [True, False])
async def test_session_hands_its_lifetime_to_the_harness(tmp_path: Path, long_lived: bool) -> None:
    session, _ = await _session(tmp_path, long_lived=long_lived)

    assert session._harness.config.long_lived is long_lived


@pytest.mark.anyio
async def test_new_session_stays_long_lived(tmp_path: Path) -> None:
    session, _ = await _session(tmp_path, long_lived=True)

    await session.new_session()

    assert session._harness.config.long_lived is True


@pytest.mark.anyio
async def test_resumed_session_stays_long_lived(tmp_path: Path) -> None:
    session, manager = await _session(tmp_path, long_lived=True)
    other = manager.create_session(cwd=tmp_path, model="fake")

    await session.resume(other.id)

    assert session._harness.config.long_lived is True
