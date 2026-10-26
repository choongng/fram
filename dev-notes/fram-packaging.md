# Fram packaging: the `fram-agent` wheel

T0948, 2026-10-26. This note explains how Fram is packaged as an installable
`fram` command without changing any upstream Tau file. That way, merging a
Tau release never conflicts with the packaging.

## What was added

Everything lives under `packaging/fram/`:

- `pyproject.toml` defines the `fram-agent` distribution. It builds Tau's
  `src/tau_ai`, `src/tau_agent` and `src/tau_coding` from `../../src` with
  hatchling's `force-include`, plus a small `fram` package, and installs a
  `fram` console script.
- `src/fram/__init__.py` holds Fram's version, for example `0.4.8.1`. The
  first three parts are the Tau release Fram bundles; the fourth counts Fram
  releases on top of it.
- `src/fram/cli.py` is the `fram` entry point. It adjusts the few places
  where Tau assumes it was installed as `tau-ai`, then runs Tau's CLI.
- `publish.sh` builds the wheel and uploads it with `uv publish`, to the
  index named by `FRAM_INDEX_URL`, using uv's own credential variables.
- `tests/test_fram_packaging.py` holds the tests described below.

## Why the entry point adjusts Tau

Tau's sources are bundled unchanged, but they look themselves up as
`tau-ai`, which is not installed under `fram`. Left alone, that causes four
problems:

1. `--version` would print `tau 0+unknown`. `fram --version` prints
   `fram 0.4.8.1 (tau 0.4.8)` instead.
2. `tau update` installs `tau-ai` from PyPI, which would replace nothing and
   install the wrong program. `fram update` is refused with upgrade advice.
   `fram update --models` still refreshes model catalogs.
3. The startup update check compares against `tau-ai` on PyPI. `fram` sets
   `TAU_NO_UPDATE_CHECK=1` unless the environment already sets it.
4. Tau's version lookup (`tau_coding.version._DISTRIBUTION_NAME`) is pointed
   at `fram-agent`, so the TUI header and the release-notes state show Fram's
   version. Because `0.4.8.1` sorts just after `0.4.8`, a Tau install sharing
   the same `~/.tau` sees no spurious release notes.

Item 4 sets a private module variable. That is a deliberate trade: it keeps
every upstream file unchanged. If an upstream merge renames it, the
end-to-end test fails.

## Using the same import names

The wheel ships `tau_ai`, `tau_agent` and `tau_coding` under their usual
names, so Tau extensions keep working. As a result, `fram` and `tau-ai`
cannot share one Python environment. Install `fram-agent` as a tool:

```bash
uv tool install fram-agent --index <fram-index-url>
```

`fram` shares Tau's configuration and sessions (`~/.tau` or `TAU_HOME`).

**Why `fram-agent`, not `fram`.** Public PyPI already has an unrelated
`fram` (1.0.0, a biology model), and in testing uv picked it over a local
`--find-links` wheel. The distribution is `fram-agent`, which was free on
PyPI on 2026-10-26; the command and import package are still `fram`. Name the
Fram index with `--index` anyway: uv searches it before PyPI, so a later
`fram-agent` upload by someone else could not take its place.

## Wheels only

The sources sit outside `packaging/fram/`, so an sdist cannot carry them in a
form a later wheel build can find. Build and publish wheels only:

```bash
uv build --wheel --project packaging/fram --out-dir dist/
FRAM_INDEX_URL=https://... UV_PUBLISH_TOKEN=... packaging/fram/publish.sh
```

Publish only from `main` (see `dev-notes/fram-branches.md`). Wheels built
from any other branch are for local testing.

## Taking a Tau release

After merging a Tau release into `main`:

1. Set `__version__` in `src/fram/__init__.py` to the new Tau version plus
   `.1`, for example `0.4.9.1`.
2. Copy any dependency changes from the root `pyproject.toml`.

Tests fail until both are done: one checks the version against the root
`pyproject.toml`, and one checks the dependency lists match.

## Testing

```bash
uv run pytest packaging/fram/tests
```

The root test suite does not collect these tests, because its `testpaths` is
`tests/`. The end-to-end test builds the wheel into a small `file://` simple
index, runs `uv tool install fram-agent --index ...` into a temporary tool
directory, then checks four things:

- `fram --version` and `fram --help` work.
- The installed `fram` reports its own version.
- `tau-ai` is not installed in the tool environment.
- `tau` from the repository still works.

## How this maps to Pi and Tau

This adds nothing to the agent architecture. `fram` is one more frontend
entry over the same `tau_coding` CLI app, the way the `tau` script is.

## Still to come

The rest of the packaging plan: a branding seam for visible "Tau" strings,
extension discovery through entry points, PyPI publishing under a free name,
and CI that watches upstream releases.
