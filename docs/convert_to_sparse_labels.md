# `convert_to_sparse_labels`

`convert_to_sparse_labels` (defined in `src/_load_dataset.py`) turns each beam-score matrix into a single integer class id so the models can be trained with `SparseCategoricalCrossentropy`/`SparseCategoricalAccuracy`.

## What it expects
- Input `y` shape: `(N, 8, 32)` where each sample stores a score per beam on an 8×32 grid (originally complex, cast to `float32` before conversion).
- Output shape: `(N,)` `int64` labels in the range `[0, 255]`.

## How the label is computed
1. Flatten each `(8, 32)` slice to length 256 using NumPy’s default C order (`y.reshape(y.shape[0], -1)`).
2. Take `argmax` along the flattened axis → index of the highest-scoring beam.
3. That index becomes the class id. Mapping back to grid:  
   `row = idx // 32`, `col = idx % 32`.

### Tiny example
For a single sample with a `(2, 3)` grid shown row-wise:

```
[[0.1, 0.2, 0.9],
 [0.05, 0.8, 0.4]]
```

Flattened order → `[0.1, 0.2, 0.9, 0.05, 0.8, 0.4]`; `argmax` picks index 2 (the `0.9`), so the sparse label is `2` (row 0, col 2).

## Why this is needed
- Keeps labels compact (one integer instead of a 256-long one-hot), matching Keras’ sparse loss/metric APIs used in the notebooks (e.g., `nas_cnn1d_v7.0`).
- Ensures the model’s `Dense(256, softmax)` outputs align with the beam classes: predicted class is `argmax` over those 256 logits, compared directly to this integer label.

## Gotchas
- If multiple beams tie for the maximum score, NumPy’s `argmax` returns the first in flattened order (upper rows, then left to right).
- Changing the flattening order would change label numbering; keep the reshape as-is to stay consistent with existing checkpoints and metrics.
