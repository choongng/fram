# Fram

Fram is a fork of [Tau](https://github.com/huggingface/tau), the Python
coding-agent harness. The `fram-agent` distribution installs the `fram` command.

It bundles Tau's own packages (`tau_ai`, `tau_agent`, `tau_coding`) under their
usual import names, so Tau extensions keep working. Because of that, `fram`
replaces `tau-ai` in any environment it is installed into. Install it in its own
tool environment:

```bash
uv tool install fram-agent
```

The distribution is `fram-agent` because the shorter name `fram` is taken on
PyPI by an unrelated project. The command is still `fram`.

`fram` calls itself Fram in help, messages and the TUI, and keeps its
configuration and sessions in `~/.fram` (set `TAU_HOME` to use another folder,
including Tau's `~/.tau`). Project `.tau/` folders are shared with Tau.
`fram --version` names the Tau release it is based on. `fram update` is not
supported; upgrade with `uv tool upgrade fram-agent`.
