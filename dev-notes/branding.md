# Branding seam

A seam that lets a redistribution of Tau change the name it shows to people
without editing every message.

## What was added

`src/tau_coding/branding.py` holds a frozen `Branding` dataclass:

| Field | Tau's value | Used for |
|---|---|---|
| `product_name` | `Tau` | Prose: "Tau is still working", the TUI title, the system prompt, export and usage pages, the OAuth page |
| `command_name` | `tau` | Usage text: `tau install`, `tau --session ...`, the CLI's program name |
| `mark` | `τ` | One-cell glyph: the prompt prefix and the terminal title |
| `logo` | `τ = 2π` | The sidebar logo, before the version |

`BRANDING` is the instance in effect. Display text in 18 `tau_coding` files
reads it instead of spelling out "Tau". With Tau's defaults every string
renders as before.

## How a distribution rebrands

It declares one entry point in the `tau_coding.branding` group. The object it
names is a mapping from `Branding` field names to values; missing fields keep
Tau's. A wheel declares:

```toml
[project.entry-points."tau_coding.branding"]
acme = "acme.branding:BRANDING"
```

The entry point is resolved once, when `tau_coding.branding` is first imported.
It cannot be later: importing `tau_coding` imports the CLI and the TUI, and
their help text, `TauTuiApp.TITLE` and the mark constants are built at import
time. So the module that holds the mapping must not import `tau_coding`. A
rebranding that fails to load, or isn't a mapping, falls back to Tau's. When
several are installed they are tried in entry-point name order.

The lookup scans installed distributions' metadata, about 19 ms including the
`importlib.metadata` import that `tau_coding.version` already needs. Importing
`tau_coding` takes about 300 ms.

## What is not branding

Names that other tools, files or services rely on stay Tau's, so by default a
rebranded build shares configuration, sessions and extensions with Tau:

- `~/.tau`, `TAU_HOME` and the other `TAU_*` variables
- `.tau` project folders and `[tool.tau]` extension metadata
- OAuth originators and the Codex `User-Agent` (`tau_ai`)
- the extension API's `tau` object, RPC source identifiers and protocol errors
- the credentials file's error messages, which describe Tau's file format
- `updater.py` and the update-check notice, which are about `tau-ai` on PyPI (a
  redistribution should turn these off)

Docstrings and comments are not display text and still say Tau.

## Keeping it complete

New display text that spells out "Tau" would show "Tau" in a rebranded build.
Use `f"{BRANDING.product_name} ..."` or `f"... {BRANDING.command_name} ..."`
unless the text names something in the list above.

## How this maps to Pi

Pi has no equivalent; it is branded by its own package. The seam sits in
`tau_coding`, the application layer, because product text belongs there.
`tau_agent` and `tau_ai` show no product name to users.
