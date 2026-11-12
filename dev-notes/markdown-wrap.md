# Wrapping Markdown prose at 80 columns

Fram's Markdown prose wraps at 80 columns. `scripts/check_md_wrap.py` checks the
paragraphs Fram changed and can rewrap them. CI runs it in the `markdown-wrap`
job of `.github/workflows/fram-ci.yml`.

## Why

Much of Tau's Markdown wraps near 80 columns, but nothing enforced it. Edits
made inside a wrapped paragraph left it with a 120-column line, or with a line
that ended a few words early. Across Fram's commits, 128 paragraphs in 26 files
had drifted this way. They were rewrapped in the commits that wrote them, and
this check keeps new ones from creeping in.

## The rule

A paragraph or list item is filled greedily: each line takes as many words as
fit in 80 columns. A line is longer only when a single word, such as a URL or an
inline code span, is longer. One rule means anyone who rewraps a paragraph gets
the same lines, so an edit and its rewrap make a predictable diff.

Only paragraphs that differ from upstream are checked. The base is the upstream
commit Fram last merged, `git merge-base HEAD upstream/main`. Rewrapping
upstream's text would turn every later upstream merge into conflicts, so a
paragraph Fram never touched is never reported. A touched paragraph or list
whose untouched lines are already longer than 80 columns follows upstream's
style there and is left alone too. `AGENTS.md`, which upstream writes with one
line per bullet, is the main example.

Skipped: fenced and indented code, tables, headings, blockquotes, HTML, front
matter (`---` or `+++`), link definitions, and paragraphs with a hard line
break. Files under a hidden directory, such as `.github/`, are not listed unless
named on the command line.

## Using it

```bash
uv run python scripts/check_md_wrap.py          # list paragraphs, exit 1 if any
uv run python scripts/check_md_wrap.py --fix    # rewrap them in place
uv run python scripts/check_md_wrap.py --base 2e98f0b8 docs/x.md
```

The check reads the working tree, so it sees uncommitted edits, but it finds new
files only once they are added with `git add`, or when named on the command
line. It needs an `upstream` remote for Tau
(`git remote add upstream https://github.com/huggingface/tau.git`) unless
`--base` names a commit. `--fix` moves words between lines and changes nothing
else; the script asserts that the words are unchanged. When rewrapping existing
commits, fold the rewrap into the commit that wrote the paragraph rather than
adding a formatting commit.

Not covered: nested list items indented four columns or more, and paragraphs
mixing indentation the script can't attribute to a list marker. Both are left as
written.

## Tests

`uv run pytest tests/test_check_md_wrap.py`.
