"""The ``fram`` command: Tau's CLI, run from the ``fram-agent`` distribution.

Tau's sources are bundled unchanged, so the few places where Tau assumes it was
installed as ``tau-ai`` are adjusted here, before Tau's CLI is imported:

- ``--version`` names Fram's version and the Tau release it is based on.
- ``update`` is refused: Tau's updater would install ``tau-ai`` from PyPI.
  ``update --models`` still works, so that is what ``--help`` lists.
- The startup update check is off by default, since it compares against
  ``tau-ai`` on PyPI.
- Fram keeps its own configuration and sessions in ``~/.fram``: ``TAU_HOME``
  defaults there instead of ``~/.tau``, and project resources live in
  ``.fram`` folders: ``TAU_PROJECT_DIRNAME`` defaults to ``.fram``. Explicit
  values still win. Both are environment variables so that extensions and the
  processes Fram starts, such as subagents and jobs, see them too.
- Tau's version lookup reads the ``fram-agent`` distribution, so the TUI and the
  release-notes state see Fram's version instead of ``0+unknown``.
- ``FRAM_NO_SUBAGENTS`` turns off the bundled tau-subagents extension, by adding
  it to ``TAU_DISABLE_PACKAGE_EXTENSIONS``.

Fram's name and marks in Tau's own text come from ``fram.branding``, which Tau
finds through the wheel's ``tau_coding.branding`` entry point rather than from
here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from fram import __version__, tau_base_version

if TYPE_CHECKING:
    from typer import Typer

SUBAGENTS_EXTENSION = "tau_subagents"

_REINSTALL_ADVICE = (
    "fram update is not supported: Tau's updater installs tau-ai from PyPI.\n"
    "Upgrade with: uv tool upgrade fram-agent  (or reinstall from your Fram index)"
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
    apply_subagents_switch()
    from tau_coding.cli import app

    list_only_supported_update(app)
    app(args=args, prog_name="fram")


def list_only_supported_update(app: Typer) -> None:
    """Make the help epilog list ``update --models``, the one form Fram supports."""
    from tau_coding.branding import BRANDING

    command, product = BRANDING.command_name, BRANDING.product_name
    app.info.epilog = str(app.info.epilog).replace(
        f"{command} update - Upgrade {product}.",
        f"{command} update --models - Refresh model catalogs.",
    )


def adopt_fram_identity() -> None:
    """Point Tau's home, project folders, update check and version lookup at Fram."""
    if not os.environ.get("TAU_HOME"):
        os.environ["TAU_HOME"] = str(Path.home() / ".fram")
    if not os.environ.get("TAU_PROJECT_DIRNAME", "").strip():
        os.environ["TAU_PROJECT_DIRNAME"] = ".fram"
    os.environ.setdefault("TAU_NO_UPDATE_CHECK", "1")
    import tau_coding
    import tau_coding.version

    tau_coding.version._DISTRIBUTION_NAME = "fram-agent"
    tau_coding.__version__ = tau_coding.version.current_version()


def apply_subagents_switch() -> None:
    """Skip the bundled tau-subagents extension when ``FRAM_NO_SUBAGENTS`` is set."""
    if os.environ.get("FRAM_NO_SUBAGENTS", "") in ("", "0"):
        return
    from tau_coding.extensions.loader import DISABLE_PACKAGE_EXTENSIONS_ENV

    disabled = os.environ.get(DISABLE_PACKAGE_EXTENSIONS_ENV, "")
    names = [name.strip() for name in disabled.split(",") if name.strip()]
    if SUBAGENTS_EXTENSION not in names:
        names.append(SUBAGENTS_EXTENSION)
    os.environ[DISABLE_PACKAGE_EXTENSIONS_ENV] = ",".join(names)
