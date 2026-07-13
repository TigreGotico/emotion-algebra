# Contributing

## Setup

```bash
uv venv && uv pip install -e '.[test]'
uv run pytest
```

The `[test]` extra pulls in every optional backend (`lexicon`, `fast`, `viz`)
plus `pytest` and `hypothesis`. Tests never skip on a missing dependency — if an
import fails, that is a failure, not a skip.

## Ground rules

**The theory comes first.** Every emotion, coefficient, and mapping in this
library traces to a published model. A change that improves a benchmark but
cannot be grounded in the literature is not an improvement. Cite the paper, or
mark the number `# calibrated:` with an explicit rationale and put it in a named
constant — never a bare literal in an expression.

**Document what the model cannot do.** Several behaviours here look like bugs and
are theorems: the Lövheim cube cannot reach positive Aptitude, its bridge to the
Hourglass space is not injective, and Cambria's polarity formula scores both
Attention poles positive. Each is proven, documented, and locked by a test.
If you find a similar limit, write it down and test it rather than papering over
it.

**Laws are machine-checked.** Anything claimed in [docs/laws.md](docs/laws.md)
must have a property test in `test/test_laws.py`. A law with a boundary must
state the boundary — an unconditional law that is only conditionally true is
worse than no law.

## Tests

- One test directory: `test/`.
- Property tests use `hypothesis` and run deterministically in CI
  (`derandomize=True`).
- Doctests run under `pytest --doctest-modules`; every example in a docstring
  must actually produce the output it claims.

## Commits and PRs

- Conventional commits drive the version bump — `fix:` (patch), `feat:` (minor),
  `feat!:` / `breaking change:` (major). Never hand-edit `version.py`.
- Branch off `dev` and open the PR against `dev`. `master` is release-only.
- Deprecations compute their removal version from `version.py`, never hardcode it.
- Keep the diff free of scratch files.
