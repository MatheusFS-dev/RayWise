# Accuracy Metric in `nas_cnn1d_v7.0`

`metrics=["accuracy"]` in `src/nas_cnn1d_v7.0.ipynb` resolves to `SparseCategoricalAccuracy` because the model is compiled with `SparseCategoricalCrossentropy` and integer labels. The metric reports how often the predicted beam class matches the ground-truth beam class (i.e., top-1 accuracy).

## Model Outputs and Labels
- **Output layer:** `Dense(256, activation="softmax")` → one probability per beam class (256 = 8×32 grid).
- **Label preparation:** `_load_dataset.convert_to_sparse_labels` flattens each `(8, 32)` beam score matrix to length 256 (row-major via `reshape`) and takes `argmax`, producing an integer in `[0, 255]` per sample.
- **Comparison:** For each sample, Keras takes `argmax` over the 256 softmax outputs to get the predicted class id and checks if it equals the integer label.

## How Accuracy Is Computed
For a batch of size `B`:

```
pred_class_i = argmax(y_pred[i])      # index 0..255
correct_i    = 1 if pred_class_i == y_true[i] else 0
batch_acc    = (sum(correct_i for i in batch)) / B
```

Across an epoch, Keras aggregates these counts over all batches (weighted by batch size).

## Numerical Example (256-Class Case)
Suppose a batch of two samples:

- Sample 1: `y_true = 42`; top softmax probabilities (index → prob) are `{42: 0.64, 17: 0.10, 99: 0.08, ...}` → `argmax = 42` → correct.
- Sample 2: `y_true = 10`; top softmax probabilities are `{11: 0.31, 10: 0.27, 3: 0.12, ...}` → `argmax = 11` → incorrect.

Then:

```
correct predictions = 1
total predictions   = 2
accuracy            = 1 / 2 = 0.5
```

## Tips Specific to This Notebook
- Keep labels as the sparse integer IDs from `convert_to_sparse_labels`; feeding one-hot vectors would change the metric and misalign with the loss.
- Keep the final softmax; supplying logits would shift `argmax` choices and make the reported accuracy inconsistent with the intended probability outputs.
- Accuracy alone ignores class imbalance. If most beams are rarely chosen, consider also logging `SparseTopKCategoricalAccuracy(k=5)` or precision/recall per class.
- If you want “top-k” correctness (e.g., beam in the best 5 candidates), add `SparseTopKCategoricalAccuracy(k=5)` alongside accuracy; the plain `accuracy` here is top-1 only.
