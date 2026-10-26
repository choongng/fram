"""Fram: the Fram fork of Tau, packaged as the ``fram-agent`` distribution.

The version's first three parts are the upstream Tau release Fram is based on;
the fourth counts Fram releases on top of it.
"""

from __future__ import annotations

__version__ = "0.4.8.1"


def tau_base_version(version: str = __version__) -> str:
    """Return the upstream Tau release a Fram version is based on."""
    return ".".join(version.split(".")[:3])
