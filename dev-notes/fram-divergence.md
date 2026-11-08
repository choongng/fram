# Fram's permanent divergences from Tau

One entry per deliberate, lasting difference between Fram and upstream Tau.
These are not candidates to send upstream. Each entry says what differs, where
it lives, and what an upstream merge must check.

## 1. The `fram-agent` distribution

- **What:** Fram installs as the `fram-agent` distribution and `fram` command,
  alongside upstream's unchanged `tau-ai` packaging.
- **Where:** `packaging/fram/` only. No upstream file changes.
- **Touches Tau at:** `tau_coding.version._DISTRIBUTION_NAME` and
  `tau_coding.__version__`, both set at startup by `fram.cli`. It also defaults
  `TAU_HOME` to `~/.fram` and sets the `TAU_NO_UPDATE_CHECK` variable, and
  intercepts the `--version` and `update` arguments before Tau's CLI sees them.
- **On each upstream merge:** if Tau's version changed, bump
  `packaging/fram/src/fram/__init__.py`. Copy dependency changes, then run
  `uv run pytest packaging/fram/tests`.
- **Details:** `dev-notes/fram-packaging.md`.

## 2. Fram's branding

- **What:** the installed `fram` calls itself Fram in Tau's display text, with
  `✦` as its prompt mark and `✦ fram` as its sidebar logo.
- **Where:** `packaging/fram/src/fram/branding.py`, declared as the wheel's
  `tau_coding.branding` entry point.
- **Touches Tau at:** the branding seam, `src/tau_coding/branding.py`, and
  the display strings in 18 `tau_coding` files that read it.
  Unlike entry 1, the seam changes upstream files. It is meant to go upstream,
  after which only the entry point remains Fram's.
- **On each upstream merge:** resolve conflicts in branded strings, then
  run `uv run pytest packaging/fram/tests`; its scan fails on new display
  text that spells out "Tau" (`dev-notes/branding.md` says how to fix it).
- **Details:** `dev-notes/branding.md`, `dev-notes/fram-packaging.md`.
