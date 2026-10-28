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
  index named by `FRAM_INDEX_URL`, using uv's own credential variables. It is
  for indexes other than PyPI.
- `.github/workflows/fram-publish.yml` publishes to PyPI; see
  [Publishing to PyPI](#publishing-to-pypi).
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
uv tool install fram-agent
```

`fram` shares Tau's configuration and sessions (`~/.tau` or `TAU_HOME`).

**Why `fram-agent`, not `fram`.** Public PyPI already has an unrelated
`fram` (1.0.0, a biology model), and in testing uv picked it over a local
`--find-links` wheel. The distribution is `fram-agent`, which was free on
PyPI on 2026-10-26; the command and import package are still `fram`.

## Wheels only

The sources sit outside `packaging/fram/`, so an sdist cannot carry them in a
form a later wheel build can find. Build and publish wheels only:

```bash
uv build --wheel --project packaging/fram --out-dir dist/
FRAM_INDEX_URL=https://... UV_PUBLISH_TOKEN=... packaging/fram/publish.sh
```

Publish only from `main` (see `dev-notes/fram-branches.md`). Wheels built
from any other branch are for local testing.

## Publishing to PyPI

`fram-agent` is published to PyPI by `.github/workflows/fram-publish.yml`,
through PyPI's trusted publishing: PyPI accepts the upload because it comes
from that workflow in Fram's GitHub repository, in the `pypi` environment, so
no API token is stored anywhere. To release:

1. Merge to `main` and publish `main` to GitHub.
2. On GitHub, create a release whose tag is `v` plus `__version__`, for
   example `v0.4.8.1`, on `main`.
3. Approve the run if the `pypi` environment requires a reviewer.

The workflow's first job refuses a tag that does not match `__version__` or is
not on `main`, then runs the root and packaging tests. The second builds the
wheel and runs `uv publish`, which finds the GitHub credentials itself. Only
the second job may request the token PyPI checks.

PyPI never accepts the same version twice, even after a release is deleted,
so a failed upload of `0.4.8.1` is retried as `0.4.8.2`.

Upstream's `.github/workflows/publish.yml` also runs on a published release.
It builds `tau-ai` from the root `pyproject.toml`, so it is disabled in the
repository's Actions settings rather than edited, which keeps the file
unchanged for upstream merges. Its PyPI upload would fail anyway, since PyPI
trusts only `fram-publish.yml` for `fram-agent`.

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
extension discovery through entry points, and CI that watches upstream
releases.
