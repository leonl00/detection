# Notes for Claude Code

The full specification is in `docs/SPEC.md`, decisions made so far are in
`docs/decisions.md`. Read both before starting work.

## How to work with me

- Answer in German. Code, docstrings and comments are in English.
- I am a beginner in Python and machine learning. Before each file, explain in
  two to three sentences what it does and why it is built that way.
- Work through the stages in "Umsetzung in Etappen" one at a time and stop after
  each stage until I say "weiter". Do not build everything at once.
- After each stage, name the command to check it and the commit messages.
- Prefer simple, readable code over clever abstractions.
- Never write placeholder numbers into results; numbers only come from real runs.

## Conventions

- Commits: small, thematic, German, imperative, e.g.
  `Fuege Duplikatserkennung ueber phash hinzu`.
- Every function: type hints, Google-style docstring, at least one test for the
  normal case and one for the error case. Tests use `tmp_path` and create their
  own data.
- Paths via `pathlib.Path`. Errors via `ValueError` / `FileNotFoundError`.
- New packages are added to `requirements.txt` with a pinned version in the stage
  that needs them. The project is installed with `pip install -e .`.
- Never commit `data/`, `reports/`, `runs/`, `*.pt` or `.env`.

## Checks

```
pytest
ruff check .
ruff format --check .
```

## Status

- Stage 1 (scaffold): done.
- Stage 2 (download): done. Dataset is in `data/dataset/`.
- Stage 3 (dataset check): done. Report in `reports/dataset_report.md`,
  decisions on the classes in `docs/decisions.md`.
- Open: delete the empty placeholders `src/detection/preprocessing.py` and
  `tests/test_preprocessing.py`.
- Next: stage 4, `grouping.py` (near-duplicates via `imagehash` phash) and
  `resplit.py` (group-wise split 70/15/15 into `data/dataset_clean/`, using its
  own fixed seed, separate from the training seed in the config).
