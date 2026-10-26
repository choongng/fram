# Fram

Fram is a fork of [Tau](https://github.com/huggingface/tau), the Python
coding-agent harness. This distribution installs the `fram` command.

It bundles Tau's own packages (`tau_ai`, `tau_agent`, `tau_coding`) under
their usual import names, so Tau extensions keep working. Because of that,
`fram` replaces `tau-ai` in any environment it is installed into. Install it
in its own tool environment:

```bash
uv tool install fram --index <your-fram-index-url>
```

Always name the Fram index with `--index`. Public PyPI has an unrelated
package called `fram`, and `--index` makes uv look for `fram` on your index
before PyPI. With `--find-links`, uv picked the PyPI package instead.

`fram` shares Tau's configuration and sessions (`~/.tau`, or `TAU_HOME`).
`fram --version` names the Tau release it is based on. `fram update` is not
supported; upgrade with `uv tool upgrade fram`.
