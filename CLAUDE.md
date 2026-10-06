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
- Stage 4 (groups and resplit): done. `grouping.py` links images with a phash
  distance of at most 14 bits (reasoning in `docs/decisions.md`): 8 of 46 test
  images of the original split have a twin in train. `resplit.py` writes a
  group-wise, class-stratified 70/15/15 split to `data/dataset_clean/` (its own
  seed `DEFAULT_SEED`, separate from the training seed); 0 of 70 test images
  there have a twin in train.
- Stage 5 (training): code done, run pending. `train.py` writes
  `runs/<name>/run_info.yaml` (config, seed, device, package versions) before
  training; tested locally with a 1-epoch CPU smoke run. Training runs in
  Colab via `notebooks/train_colab.ipynb`: data comes from
  `MyDrive/detection/data.zip` (both datasets, built with
  `python -m zipfile -c data/data.zip data/dataset data/dataset_clean`),
  results go to `MyDrive/detection/runs/<name>/`.
- Dataset switched (2026-10-06): the first baseline run reached mAP50 0.056
  because the labels of the first dataset were unusable (tiny boxes inside large
  spaghetti clumps). New dataset: `3d-test/3d-print-error-box` version 26
  (2624 images, no augmentation; classes layer shift, spaghetti, stringing,
  warping). Stages 2 and 3 are redone; label quality checked by box sizes per
  class and sample images. The old data is in `data/old_3d-print-defect/`, the
  old run in `runs/baseline_old_dataset/`. Reasoning in `docs/decisions.md`.
- Stage 4 redone on the new dataset: phash limit 14 checked again and kept.
  Original split: 64 of 254 test images (25 %) have a twin in train. Clean split
  (1837 / 394 / 393): 0 of 393. `data/data.zip` rebuilt with both datasets.
- Next: user replaces `data.zip` in Drive and trains `configs/baseline.yaml`
  again in Colab, then copies the run to `runs/baseline/`; then stage 6,
  `evaluate.py`. The config for the clean split is added in stage 7.
