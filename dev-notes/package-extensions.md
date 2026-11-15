# Package extensions: extensions from installed distributions

Tau discovers extensions that installed Python distributions declare in the
`tau_coding.extensions` entry-point group, as well as extensions in directories
and `-e` paths.

## What was added

In `src/tau_coding/extensions/loader.py`:

- `discover_extensions` lists the group's entry points after the directory and
  explicit sources. Each becomes a `DiscoveredExtension` with
  `source="package"`, no `path`, and the `EntryPoint` itself. Its source ID is
  `package:<module>:<function>`.
- `load_extensions` loads them with `EntryPoint.load()` instead of importing a
  file, and checks the result the way it checks a file's `setup`: callable and
  not async. Import errors and bad entry points become error diagnostics, as
  they do for files.
- `TAU_DISABLE_PACKAGE_EXTENSIONS` names entry points to skip, comma-separated.
- `include_packages=False` leaves them out. The extension installer passes it,
  because it validates a staged directory and must not count what happens to be
  installed.

## Why it exists

An application built on Tau needs a way to ship an extension that is on by
default. Built-ins (`src/tau_coding/built_in_extensions.py`) were the only
mechanism, and they don't fit an extension that adds tools:

- Built-ins load even with `--no-extensions`, and even in sessions created with
  `extensions_enabled=False`. A tau-subagents child started with
  `isolated: true` is such a session; it is meant to get core tools only.
- Built-ins load before every other source, so a working copy loaded with `-e`
  could not replace the shipped one.
- The list is a constant in Tau's own source, so a downstream package can add to
  it only by patching it.

Package extensions follow the user-extension rules instead. They need
`include_resource_dirs` and `include_user_dir`, so `--no-extensions`, isolated
child sessions, and the project-only second load all leave them out. They are
discovered last, so the existing first-loaded-wins name check lets a directory
or `-e` copy of the same name shadow them.

Fram uses this to ship tau-subagents; see `dev-notes/fram-packaging.md`.

## How it maps to Pi

Pi finds a package's extensions through the `pi.extensions` field of its
`package.json`, which Tau's `[tool.tau]` manifest mirrors for directories. Entry
points are the Python packaging equivalent of that field for installed
distributions. Installing the distribution is the opt-in; there is no separate
settings list.

## Limits

- Modules import under their real package names, not the namespaced
  `tau_extension_*` names file extensions get, so `/reload` re-runs `setup`
  without re-importing the code. Module-level state survives a reload.
- Project extensions load in a second pass after project trust. A project
  extension with the same name as a package extension does not shadow it; the
  first registration of each tool and command wins, as for any two extensions.

## Testing

```bash
uv run pytest tests/test_package_extensions.py
```

The tests replace the loader's `entry_points` function with a list of
`EntryPoint` objects pointing at a module in a temporary directory, so they do
not depend on what the environment has installed.
