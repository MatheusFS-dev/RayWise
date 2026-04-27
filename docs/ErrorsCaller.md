# ErrorsCaller

This note exists to warn public users about historical issues in older files.

## Scope

Some old scripts and notebooks in `legacy/` and versioned architecture folders were created during rapid experimentation and may contain mistakes.

Newer versions are generally more reliable and are the recommended starting point.

## Common Historical Problems

- Callback or utility wiring mismatches after quick refactors.
- Tensor shape assumptions that no longer hold after architecture edits.
- Hardcoded or machine-specific paths left from local runs.
- Experimental branches where a module was intentionally left disconnected.

## What To Do Before Reusing Old Files

1. Validate input/output tensor shapes end-to-end.
2. Confirm dataset paths and external model paths are valid on your machine.
3. Check callbacks and custom metrics are actually connected to the training loop.
4. Compare with newer equivalents before trusting historical results.

## Recommended Stable Starting Points

- `src/architectures/cnn1d/nas/v7/nas_v7_seeds_val_acc.py`
- `src/architectures/cnn1d/ensemble/nas_mo_kd_cnn1d.py`

These scripts represent the most relevant baseline for public use in this repository.
