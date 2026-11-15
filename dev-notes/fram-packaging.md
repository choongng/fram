# Fram packaging: the `fram-agent` wheel

Fram is packaged as an installable `fram` command from `packaging/fram/`, which
upstream does not touch, so merging a Tau release does not conflict with it.

## What was added

Everything lives under `packaging/fram/`:

- `pyproject.toml` defines the `fram-agent` distribution. It builds Tau's
  `src/tau_ai`, `src/tau_agent` and `src/tau_coding` from `../../src` with
  hatchling's `force-include`, plus a small `fram` package, and installs a
  `fram` console script. It also bundles the tau-subagents extension from
  `vendor/` (see "Bundled tau-subagents" below).
- `src/fram/__init__.py` holds Fram's version, for example `0.4.8.1`. The first
  three parts are the Tau release Fram bundles; the fourth counts Fram releases
  on top of it.
- `src/fram/cli.py` is the `fram` entry point. It adjusts the few places where
  Tau assumes it was installed as `tau-ai`, then runs Tau's CLI.
- `src/fram/branding.py` holds Fram's name, command and marks. The wheel
  declares it as a `tau_coding.branding` entry point (see "Branding" below).
- `publish.sh` builds the wheel and uploads it with `uv publish`, to the index
  named by `FRAM_INDEX_URL`, using uv's own credential variables. It is for
  indexes other than PyPI.
- `.github/workflows/fram-publish.yml` publishes to PyPI; see [Publishing to
  PyPI](#publishing-to-pypi).
- `tests/test_fram_packaging.py` holds the tests described below.

## Why the entry point adjusts Tau

Tau's sources are bundled unchanged, but they look themselves up as `tau-ai`,
which is not installed under `fram`. The entry point makes these adjustments:

1. `--version` would print `tau 0+unknown`. `fram --version` prints
   `fram 0.4.8.1 (tau 0.4.8)` instead.
2. `tau update` installs `tau-ai` from PyPI, which would replace nothing and
   install the wrong program. `fram update` is refused with upgrade advice.
   `fram update --models` still refreshes model catalogs.
3. The startup update check compares against `tau-ai` on PyPI. `fram` sets
   `TAU_NO_UPDATE_CHECK=1` unless the environment already sets it.
4. Tau's version lookup (`tau_coding.version._DISTRIBUTION_NAME`) is pointed at
   `fram-agent`, so the TUI header and the release-notes state show Fram's
   version. Because `0.4.8.1` sorts just after `0.4.8`, a Tau install sharing
   the same home sees no spurious release notes.
5. `TAU_HOME` defaults to `~/.fram`, so Fram's providers, credentials, sessions
   and settings are its own. An explicit `TAU_HOME`, even `~/.tau`, still wins.
   Commands Fram's shell tool runs inherit it, so a `tau` started from inside
   Fram uses `~/.fram` too.

Item 4 sets a private module variable, which keeps every upstream file
unchanged. If an upstream merge renames it, the end-to-end test fails.

## Using the same import names

The wheel ships `tau_ai`, `tau_agent` and `tau_coding` under their usual names,
so Tau extensions keep working. As a result, `fram` and `tau-ai` cannot share
one Python environment. Install `fram-agent` as a tool:

```bash
uv tool install fram-agent
```

`fram` keeps its configuration and sessions in `~/.fram` (item 5 above). To
start from an existing Tau setup, copy it: `cp -r ~/.tau ~/.fram`.

**Why `fram-agent`, not `fram`.** Public PyPI already has an unrelated `fram`
(1.0.0, a biology model), and in testing uv picked it over a local
`--find-links` wheel. The distribution is `fram-agent`; the command and import
package are still `fram`.

## Wheels only

The sources sit outside `packaging/fram/`, so an sdist cannot carry them in a
form a later wheel build can find. Build and publish wheels only:

```bash
uv build --wheel --project packaging/fram --out-dir dist/
FRAM_INDEX_URL=https://... UV_PUBLISH_TOKEN=... packaging/fram/publish.sh
```

Publish only from `main`.

## Branding

Tau's display text (CLI help and usage, TUI title and messages, the prompt mark,
the sidebar logo, the system prompt, export pages) reads its product name,
command and marks from `tau_coding.branding.BRANDING`. Tau resolves that once,
on import, from an entry point in the `tau_coding.branding` group. The
`fram-agent` wheel declares one:

```toml
[project.entry-points."tau_coding.branding"]
fram = "fram.branding:BRANDING"
```

So the installed `fram` says "Fram" and `fram install ...`, shows `✦` as its
prompt mark and `✦ fram` as its logo, while `tau` from a checkout, which has no
such entry point, is unchanged. `fram.branding` must not import Tau, because Tau
loads it while `tau_coding` is still importing.

Branding is display only. Names other tools rely on stay Tau's: the `TAU_HOME`
variable and the other `TAU_*` variables, `.tau` project folders, `[tool.tau]`,
OAuth originators and the extension API's `tau` object. So `fram` and `tau`
still share project `.tau/` folders; only the global home differs (`~/.fram`,
item 5).

The seam itself is a change to upstream files (unlike the rest of this note);
`dev-notes/branding.md` describes it.

## Bundled tau-subagents

Fram ships [tau-subagents](https://github.com/rian-dolphin/tau-subagents) (MIT,
by Rian Dolphin) and loads it by default.

**Where the code comes from.** `vendor/tau-subagents/` is a squashed
`git subtree` of a fork of tau-subagents that carries changes not yet upstream.
A subtree, not a submodule, because the fork is not public; the files are
ordinary tracked files, so every clone and CI run has them. The squash commit's
`git-subtree-split` trailer records the fork commit. To take new work from the
fork:

```bash
git subtree pull --prefix vendor/tau-subagents <fork URL> <branch> --squash
```

Make fixes in the tau-subagents repository and pull them, rather than editing
`vendor/` here. Ruff and the Markdown wrap check skip `vendor/`, since the code
keeps its own style.

**How it ships.** `pyproject.toml` force-includes
`vendor/tau-subagents/src/tau_subagents` as the `tau_subagents` package, its
`LICENSE` as `tau_subagents/LICENSE`, and its `tau-job` script under that name,
which the extension's own text tells the model to use. Its dependencies
(`textual`) are already Tau's; its `tau-ai` dependency is not declared, because
the wheel is Tau.

**How Tau finds it.** The wheel declares it as a package extension (see
`dev-notes/package-extensions.md`):

```toml
[project.entry-points."tau_coding.extensions"]
tau_subagents = "tau_subagents.extension:setup"
```

So it follows the user-extension rules: `--no-extensions` turns it off, a child
session started with `isolated: true` does not get it, and the TUI sidebar lists
it with the session's other extensions. The name is the one Tau gives a
tau-subagents checkout, so `fram -e /path/to/tau-subagents` (or a checkout in
`~/.fram/extensions/`) loads in its place.

**Turning it off.** `fram` maps `FRAM_NO_SUBAGENTS=1` onto Tau's
`TAU_DISABLE_PACKAGE_EXTENSIONS`, adding `tau_subagents` to any names already
there.

**Versioning.** Syncing tau-subagents does not need its own Fram release; it
goes out with the next one.

**Its tests.** `fram-ci.yml` runs `vendor/tau-subagents/tests` against Fram's
Tau through `scripts/test_vendored_subagents.py`, so a change to Fram's Tau that
breaks the extension fails CI. The suite also passes on tau-ai 0.3.5, which
tau-subagents locks, and on upstream 0.4. Where the versions differ, its tests
accept each one; the one behavior 0.4 dropped, `skills: true` pinning discovery
under worktree isolation, is a strict expected failure there.

## Publishing to PyPI

`fram-agent` is published to PyPI by `.github/workflows/fram-publish.yml`,
through PyPI's trusted publishing: PyPI accepts the upload because it comes from
that workflow in Fram's GitHub repository, in the `pypi` environment, so no API
token is stored anywhere. To release:

1. Merge to `main` and publish `main` to GitHub.
2. On GitHub, create a release whose tag is `v` plus `__version__`, for example
   `v0.4.8.1`, on `main`.
3. Approve the run if the `pypi` environment requires a reviewer.

The workflow's first job refuses a tag that does not match `__version__` or is
not on `main`, then runs the root and packaging tests. The second builds the
wheel and runs `uv publish`, which finds the GitHub credentials itself. Only the
second job may request the token PyPI checks.

PyPI never accepts the same version twice, even after a release is deleted, so a
failed upload of `0.4.8.1` is retried as `0.4.8.2`.

Upstream's `.github/workflows/publish.yml` also runs on a published release. It
builds `tau-ai` from the root `pyproject.toml`, so it is disabled in the
repository's Actions settings rather than edited, which keeps the file unchanged
for upstream merges. Its PyPI upload would fail anyway, since PyPI trusts only
`fram-publish.yml` for `fram-agent`.

## Taking a Tau release

After merging a Tau release into `main`:

1. Set `__version__` in `src/fram/__init__.py` to the new Tau version plus `.1`,
   for example `0.4.9.1`.
2. Copy any dependency changes from the root `pyproject.toml`.
3. Look for new display text that spells out "Tau" instead of reading
   `BRANDING`; `dev-notes/branding.md` says how.

Tests fail until the first two are done: one checks the version against the root
`pyproject.toml`, and one checks the dependency lists match.

## Testing

```bash
uv run pytest packaging/fram/tests
```

The root test suite does not collect these tests, because its `testpaths` is
`tests/` and adding this directory would load two `conftest` modules under one
name. `.github/workflows/fram-ci.yml` runs them on every push to `main` and on
pull requests, so a merge that leaves the version behind fails there. The
end-to-end test builds the wheel into a small `file://` simple index, runs
`uv tool install fram-agent --index ...` into a temporary tool directory, then
checks:

- `fram --version` and `fram --help` work, and the help says "Fram" and
  `fram install`, not `tau install`.
- The TUI's title is "Fram".
- The installed `fram` reports its own version.
- `tau-ai` is not installed in the tool environment.
- The bundled tau-subagents loads as a package extension with its license,
  `FRAM_NO_SUBAGENTS=1` leaves it out, a checkout given with `-e` replaces it,
  and `tau-job` runs a job.
- `tau` from the repository still works and still says "Tau".

## How this maps to Pi and Tau

This adds nothing to the agent architecture. `fram` is one more frontend entry
over the same `tau_coding` CLI app, the way the `tau` script is.

## Still to come

CI that watches upstream releases.
