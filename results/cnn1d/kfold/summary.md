# CNN1D v7.0 K-Fold Results

## Purpose
This report summarizes the k-fold cross-validation results for CNN1D v7.0 and compares 10-split vs 20-split setups.

## K-Fold Process (High Level)
1. Split the training set into k folds.
2. For each fold:
   - Train the model on k-1 folds.
   - Validate on the held-out fold.
   - Track best epoch by validation metrics.
3. Aggregate out-of-fold (OOF) predictions across folds to estimate overall CV performance.
4. For test evaluation, run the test set through each fold model and average predictions (bagging) to get the final test metrics.

## Metrics (Short Definitions)
- best_*: metric at the best validation epoch for each fold.
- val_*: metric at the final epoch for each fold.
- OOF: metric computed from out-of-fold predictions (cross-validation estimate).
- test_*_bagged: test metrics using averaged predictions from all fold models.

## Results Summary

| Setting | n_splits | OOF loss | OOF acc | Test loss (bagged) | Test acc (bagged) | Best val acc mean +/- std | Best val loss mean +/- std | Best epoch mean +/- std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v7.0 kfold | 10 | 1.6919 | 0.5680 | 1.5544 | 0.6188 | 0.5714 +/- 0.0166 | 1.6996 +/- 0.0783 | 42.6 +/- 7.9 |
| v7.0 kfold | 20 | 1.6806 | 0.5685 | 1.5521 | 0.6184 | 0.5723 +/- 0.0204 | 1.6906 +/- 0.0863 | 40.4 +/- 5.2 |

### Fold Extremes
- 10 splits: best fold = 6 (best_val_acc 0.6059, best_val_loss 1.5305), worst fold = 3 (0.5491, 1.8251).
- 20 splits: best fold = 11 (best_val_acc 0.6125, best_val_loss 1.4871), worst fold = 5 (0.5321, 1.8677).

## Comparison and Takeaways
- 20 splits slightly improves OOF and test loss (small reductions) while accuracy is effectively unchanged.
- Variance across folds is a bit higher with 20 splits, which is expected from smaller validation fold sizes.
- Train vs best-val accuracy gap is consistent (~0.032) in both settings, suggesting similar generalization behavior.

## Sources
- `results/cnn1d/kfold/train_cnn1d_v7.0_kfold_10_splits/model_training/args/model_stats.txt`
- `results/cnn1d/kfold/train_cnn1d_v7.0_kfold_10_splits/model_training/args/fold_metrics.csv`
- `results/cnn1d/kfold/train_cnn1d_v7.0_kfold_20_splits/model_training/args/model_stats.txt`
- `results/cnn1d/kfold/train_cnn1d_v7.0_kfold_20_splits/model_training/args/fold_metrics.csv`
