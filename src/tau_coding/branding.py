"""Product branding: the name, command and marks Tau shows to people.

Display text reads these values instead of spelling out "Tau", so a
redistribution can rebrand without editing every message. Names that other
tools rely on are identity, not branding, and stay as they are: ``~/.tau`` and
``TAU_HOME``, ``TAU_*`` environment variables, ``.tau`` project folders,
``[tool.tau]`` extension metadata, OAuth originators and the extension API's
``tau`` object.

A distribution rebrands by declaring one entry point in the
``tau_coding.branding`` group. Its object maps ``Branding`` field names to
values; missing fields keep Tau's. The entry point is resolved once, when this
module is first imported, because some branded text (CLI help, the TUI title)
is built at import time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points

ENTRY_POINT_GROUP = "tau_coding.branding"


@dataclass(frozen=True, slots=True)
class Branding:
    """What the product calls itself in user-visible text."""

    product_name: str = "Tau"
    """Name used in prose, such as "Tau is still working"."""
    command_name: str = "tau"
    """Console command used in usage text, such as ``tau update``."""
    mark: str = "τ"
    """One-cell glyph for the prompt prefix and terminal title."""
    logo: str = "τ = 2π"
    """Sidebar logo, shown before the version."""


def load_branding() -> Branding:
    """Return the branding declared by an installed distribution, or Tau's.

    A rebranding that fails to load falls back to Tau's.
    """
    for entry_point in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda ep: ep.name):
        try:
            values = entry_point.load()
            if isinstance(values, Mapping):
                return Branding(**values)
        except Exception:
            continue
    return Branding()


BRANDING = load_branding()
