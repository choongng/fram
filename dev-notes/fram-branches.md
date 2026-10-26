# Fram branch structure

Agreed with the owner on 2026-10-26. Fram is a fork of huggingface/tau. This
note says which branch holds what, how upstream releases come in, and how
our own changes go back out.

```text
upstream/main  (huggingface/tau, read-only)
  │  merge each release tag, never rebase (main is public)
  ▼
main           (published) = latest upstream release + curated Fram commits
  │  merge main into the development branch after each upstream merge
  ▼
development    (private) raw work; topics get rebuilt cleanly onto main

upstream/<topic>  cut from upstream/main, rebase freely, send as PRs
```

## Branches

- **`upstream/main`** is huggingface/tau's `main`, fetched from the
  `upstream` remote. Its push URL is disabled. Nobody commits to it.
- **`main`** is Fram as a product: the latest upstream release we have taken,
  plus Fram commits curated by hand into clean, topical form. The owner
  publishes it to the GitHub mirror when they choose.
- **The development branch** holds raw work and unreviewed history. It is
  private and never published.
- **`upstream/<topic>`** branches each hold one feature's final state, cut from
  `upstream/main`, with comments that don't cite internal run IDs. They
  become PRs to huggingface/tau.

## Flows

**Taking an upstream release.** Merge the release tag into `main` with a
merge commit. Do not rebase, because `main` is published and a rebase would
rewrite history that mirror users may have pulled. Then merge `main` into
the development branch. Before `main` had any Fram commits, release v0.4.8
was merged straight into the development branch. From now on a release
goes through `main` first.

**Curating Fram work.** Rebuild a finished topic from the development branch
as clean commits on `main`, from the end state rather than by
cherry-picking the raw chain. Then merge `main` back into the development
branch, so the two branches share history and later merges stay small.

**Sending work upstream.** Cut `upstream/<topic>` from the current
`upstream/main`. Rebase it freely until it is merged, because it holds only
that topic's commits. Once upstream accepts it, the next release merge brings
the change in, and the Fram-only version can be dropped.

## Keeping merges cheap

Fram's own additions sit behind seams rather than edits to upstream files
wherever possible: extensions, and a separate `packaging/fram/pyproject.toml`
for the wheel instead of changing upstream's `pyproject.toml`. The
smaller our diff to upstream files, the fewer conflicts each release merge
brings.

## Guards

A local pre-push hook enforces the publishing rules. Pushing `main` to the
private remote needs `FRAM_MAIN=1`. Pushing to GitHub needs
`FRAM_PUBLISH=1`, accepts only `refs/heads/main`, and only at the commit
the private remote's `main` already holds. Agents set neither variable
unless the owner asks in that conversation.

## State on 2026-10-26

- `main` was fast-forwarded to v0.4.8, then took its first Fram commits: this
  note and the `fram` packaging (`dev-notes/fram-packaging.md`). With no Fram
  commits on it yet, v0.4.8 needed no merge commit.
- The development branch merged v0.4.8 (`3bd42c04`). Upstream `main` is 7
  commits past that release.
- GitHub's `upstream-main` branch is also at v0.4.8.
