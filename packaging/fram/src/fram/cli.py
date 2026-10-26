"""The ``fram`` command: Tau's CLI, run from the ``fram`` distribution.

Tau's sources are bundled unchanged, so the few places where Tau assumes it was
installed as ``tau-ai`` are adjusted here, before Tau's CLI is imported:

- ``--version`` names Fram's version and the Tau release it is based on.
- ``update`` is refused: Tau's updater would install ``tau-ai`` from PyPI.
- The startup update check is off by default, since it compares against
  ``tau-ai`` on PyPI.
- Tau's version lookup reads the ``fram`` distribution, so the TUI and the
  release-notes state see Fram's version instead of ``0+unknown``.
"""

from __future__ import annotations

import os
import sys

from fram import __version__, tau_base_version

_REINSTALL_ADVICE = (
    "fram update is not supported: Tau's updater installs tau-ai from PyPI.\n"
    "Upgrade with: uv tool upgrade fram  (or reinstall from your Fram index)"
)


def main(argv: list[str] | None = None) -> None:
    """Run Tau's CLI as ``fram``."""
    args = sys.argv[1:] if argv is None else argv
    if args in (["--version"], ["-v"]):
        print(f"fram {__version__} (tau {tau_base_version()})")
        return
    if args[:1] == ["update"] and "--models" not in args:
        print(_REINSTALL_ADVICE, file=sys.stderr)
        raise SystemExit(1)

    adopt_fram_identity()
    from tau_coding.cli import app

    app(args=args, prog_name="fram")


def adopt_fram_identity() -> None:
    """Point Tau's update check and version lookup at Fram instead of tau-ai."""
    os.environ.setdefault("TAU_NO_UPDATE_CHECK", "1")
    import tau_coding
    import tau_coding.version

    tau_coding.version._DISTRIBUTION_NAME = "fram"
    tau_coding.__version__ = tau_coding.version.current_version()
