# Contributing

## Setup

```bash
git clone https://github.com/<you>/agentic-reliability-lab
cd agentic-reliability-lab
make install
uv run pre-commit install
```

## Before opening a PR

```bash
make check   # ruff + mypy --strict + pytest
```

All three must pass. CI runs the same commands (`.github/workflows/ci.yml`).

## Code style

- Strict typing (`mypy --strict`); avoid `Any` unless crossing a real
  external-library boundary that lacks stubs (see `pyproject.toml`'s
  `[[tool.mypy.overrides]]` for the one case where this applies today:
  scikit-learn).
- No placeholder implementations (`pass`, `raise NotImplementedError`) in
  code presented as finished. If something is genuinely an extension
  point, say so in a docstring, don't leave a stub silently.
- Keep modules focused — see `docs/architecture.md` for the layer
  boundaries and why they exist. New code should fit into an existing
  layer; if it doesn't, that's worth raising in the PR description before
  writing a lot of code.
- No generic `utils.py`. If you're tempted to add one, the function
  probably belongs in the module that actually uses it.

## Tests

- Unit tests (`tests/unit/`) for pure logic: domain models, detectors,
  feature extraction, graders.
- Integration tests (`tests/integration/`) for anything touching the
  filesystem, SQLite, or the FastAPI test client.
- `tests/end_to_end/test_smoke_pipeline.py` is the one test that proves
  the whole system works — if you change the runner, evaluation engine, or
  analysis engine, make sure this still passes and add a case if your
  change adds new behavior worth covering end to end.

## Research contributions

If your PR touches the reliability model, benchmark suite, or fault
injection, please update `docs/research/experiments.md` with real,
re-run numbers — not projected or estimated ones. See
`docs/research/limitations.md` for the standard of honesty this repo
tries to hold to; PRs that add capability claims not backed by code that
actually does what's claimed will be asked to either implement it for
real or move the claim to "future work."

## Reporting bugs / requesting features

Use the issue templates under `.github/ISSUE_TEMPLATE/`. There's a
dedicated "research idea" template if what you have is a hypothesis and an
experiment design rather than a bug or a concrete feature.
