# Fram's permanent divergences from Tau

One entry per deliberate, lasting difference between Fram and upstream Tau.
These are not candidates to send upstream (those are in
`dev-notes/upstreaming-candidates.md`). Each entry says what differs, where
it lives, and what an upstream merge must check.

## 1. The `fram-agent` distribution (T0948, 2026-10-26)

- **What:** Fram installs as the `fram-agent` distribution and `fram` command,
  alongside upstream's unchanged `tau-ai` packaging.
- **Where:** `packaging/fram/` only. No upstream file changes.
- **Touches Tau at:** `tau_coding.version._DISTRIBUTION_NAME` and
  `tau_coding.__version__`, both set at startup by `fram.cli`. It also sets
  the `TAU_NO_UPDATE_CHECK` variable, and intercepts the `--version` and
  `update` arguments before Tau's CLI sees them.
- **On each upstream merge:** if Tau's version changed, bump
  `packaging/fram/src/fram/__init__.py`. Copy dependency changes, then run
  `uv run pytest packaging/fram/tests`.
- **Details:** `dev-notes/fram-packaging.md`.
