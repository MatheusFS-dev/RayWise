# Documentation Guide

This folder contains operational notes, methodology details, and references used across RayWise experiments.

## How To Use This Folder

Use this file as the index when navigating documentation for the public version of the repository.

## Files And Purpose

- `docs/howto.md`: practical environment and tooling commands (Git, virtual environments, Docker, Jupyter).
- `docs/ErrorsCaller.md`: known historical mistakes found in some older scripts/notebooks and how to validate before reuse.
- `docs/accuracy_metric.md`: notes on evaluation metrics used in experiments.
- `docs/convert_to_sparse_labels.md`: explanation of sparse-label formatting used by loaders and training scripts.
- `docs/dataset-label-imbalance.md`: dataset imbalance observations and considerations.
- `docs/Kullback--Leibler.md`: KL-divergence notes used for distillation-related reasoning.
- `docs/stratified_split.md`: split strategy notes for controlled train/validation partitions.
- `docs/tips.md`: short practical tips collected during experimentation.
- `docs/TODO.md`: pending documentation or research follow-ups.
- `docs/references/references.md`: bibliography and links.
- `docs/related_works/related_works.md`: related-work notes and summaries.
- `docs/related_works/summary.md`: concise companion summary to related works.

## Recommended Reading Order For New Users

1. `README.md` (repository root) for scope and run instructions.
2. `docs/ErrorsCaller.md` to avoid legacy pitfalls.
3. `docs/howto.md` for environment setup.
4. `docs/accuracy_metric.md` and `docs/stratified_split.md` for experimental interpretation.

## Public Release Notes

- `src/architectures/cnn1d/nas/v7/nas_v7_seeds_val_acc.py` is the best architecture search script.
- `src/architectures/cnn1d/ensemble/nas_mo_kd_cnn1d.py` is the knowledge distillation implementation built around the best architecture family.
- `weights/` stores best model artifacts found during the project.
- Most other architecture folders are exploratory tests and historical variants.
