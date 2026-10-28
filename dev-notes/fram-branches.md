# Fram branch structure

Agreed with the owner on 2026-10-26. Fram is a fork of huggingface/tau. This
note says which branch holds what, how upstream changes come in, and how our
own changes go back out.

```text
upstream/main  (huggingface/tau, read-only)
  │  merge upstream/main, never rebase (main is public)
  ▼
main           (published) = upstream main + curated Fram commits
  │  merge main into the development branch after each upstream merge
  ▼
development    (private) raw work; topics get rebuilt cleanly onto main

upstream/<topic>  cut from upstream/main, rebase freely, send as PRs
```

## Branches

- **`upstream/main`** is huggingface/tau's `main`, fetched from the
  `upstream` remote. Its push URL is disabled. Nobody commits to it.
- **`main`** is Fram as a product: upstream's `main` as of our last merge,
  plus Fram commits curated by hand into clean, topical form. The owner
  publishes it to the GitHub mirror when they choose.
- **The development branch** holds raw work and unreviewed history. It is
  private and never published.
- **`upstream/<topic>`** branches each hold one feature's final state, cut from
  `upstream/main`, with comments that don't cite internal run IDs. They
  become PRs to huggingface/tau.

## Flows

**Taking upstream changes.** Fetch `upstream` and merge `upstream/main` into
`main` with a merge commit. Do not rebase, because `main` is published and a
rebase would rewrite history that mirror users may have pulled. Then merge
`main` into the development branch. Merge before curating a topic that
touches code upstream has changed, so the topic is rebuilt on upstream's
current code rather than conflicting with it at the next merge. Upstream
changes always reach the development branch through `main`.

We track `upstream/main` rather than release tags (changed 2026-10-26; the
first plan merged only tags). Upstream's work lands on `main` well before
a release, and Fram's TUI fixes overlap it, so waiting for tags meant
curating against code that was already out of date. The cost is that `main`
can hold unreleased upstream code. Fram's version still starts with the last
Tau release (`packaging/fram/src/fram/__init__.py`), so a Fram build between
releases means "that release plus later upstream commits".

**Curating Fram work.** Rebuild a finished topic from the development branch
as clean commits on `main`, from the end state rather than by
cherry-picking the raw chain. Then merge `main` back into the development
branch, so the two branches share history and later merges stay small.

**Sending work upstream.** Cut `upstream/<topic>` from the current
`upstream/main`. Rebase it freely until it is merged, because it holds only
that topic's commits. Once upstream accepts it, the next upstream merge brings
the change in, and the Fram-only version can be dropped.

## Keeping merges cheap

Fram's own additions sit behind seams rather than edits to upstream files
wherever possible: extensions, and a separate `packaging/fram/pyproject.toml`
for the wheel instead of changing upstream's `pyproject.toml`. The
smaller our diff to upstream files, the fewer conflicts each upstream merge
brings.

## Guards

A local pre-push hook enforces the publishing rules. Pushing `main` to the
private remote needs `FRAM_MAIN=1`. Pushing to GitHub needs `FRAM_PUBLISH=1`
and accepts two kinds of branch: `main`, only at the commit the private
remote's `main` already holds, and `upstream/<topic>`, which may be pushed,
force-pushed or deleted because each one is the head of a PR to
huggingface/tau. Wherever `main` is pushed, the hook refuses it if its files,
the lines its new commits add, or their messages name the development branch
or the private host. Agents set neither variable unless the owner asks in
that conversation.

## State on 2026-10-26

- `main` was fast-forwarded to v0.4.8, then took its first Fram commits: this
  note and the `fram` packaging (`dev-notes/fram-packaging.md`). With no Fram
  commits on it yet, v0.4.8 needed no merge commit.
- The development branch merged v0.4.8 (`3bd42c04`). Upstream `main` is 7
  commits past that release.
- GitHub's `upstream-main` branch is also at v0.4.8.
