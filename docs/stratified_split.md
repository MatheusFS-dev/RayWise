# Class-Stratified Train/Val/Test Split

This note explains how the class-stratified splitting in `src/_load_dataset_raw.py` works and checks that the reported class counts match the intended 70/20/10 proportions. The logic below refers to the `stratify_by_class` branch that produced the logs from `nas_cnn1d_v6.2.ipynb`.

## How the split works
- **Inputs:** LOS and NLOS samples are concatenated (`lidar_all`, `coord_all`, `y_all`), labels are converted to sparse integers, and a reproducible `numpy.random.default_rng(data_seed)` is used for shuffling.
- **Per-label loop:** For each unique label in `y_all`:
  - Collect indices for that label and shuffle them with the RNG.
  - If the class has only one sample, it is kept entirely in the train set (ensures the label is learnable).
  - Otherwise compute raw counts with floors:  
    `train_n = floor(0.7 * total)`, `val_n = floor(0.2 * total)`, `test_n = total - train_n - val_n`.
  - Safety guards ensure at least one train sample, give validation one sample when the floor made it zero (if any remaining slots), prevent negative counts, and cap `val_n` if the remainder is too small.
  - Indices are appended in order: first `train_n`, then `val_n`, then `test_n`.
- **Assemble splits:** The collected indices slice `lidar_all`, `coord_all`, and `y_all` into train/val/test arrays.
- **Independent shuffles:** After slicing, each split is shuffled independently (`_shuffle_together`) so that samples are not ordered by class.
- **Coverage report:** A `np.bincount` over all seen labels prints per-split counts and percentages.

Key implications:
- Classes absent from the source data (94 out of 256 possible labels) stay absent.
- Singleton classes go 100% to train. Classes with 2 samples become 1/1/0, with 3 samples become 2/1/0, etc.; validation often starts appearing at 5 samples (3/1/1).
- For larger classes the floor math produces near-70/20/10 splits, with the remainder landing in the test split.

## Worked example with the provided run

Run metadata:
- Total samples after concatenating LOS/NLOS: **20,832**
- Unique labels present: **162** (from label range 0–255)
- Target split sizes: train 14,522 | val 4,103 | test 2,207
- Label coverage: Train 162/256 classes, Val 113/256, Test 147/256

Per-class examples showing the calculation:
- **Label 16:** total 214 samples  
  - `floor(0.7*214)=149`, `floor(0.2*214)=42`, remainder `214-149-42=23` → reported 149/42/23 (69.63%/19.63%/10.75%).
- **Label 30:** total 58 samples  
  - `floor(0.7*58)=40`, `floor(0.2*58)=11`, remainder `58-40-11=7` → reported 40/11/7 (68.97%/18.97%/12.07%).
- **Label 5:** total 18 samples  
  - `floor(0.7*18)=12`, `floor(0.2*18)=3`, remainder `18-12-3=3` → reported 12/3/3 (66.67%/16.67%/16.67%).
- **Label 23:** singleton class  
  - Total 1 → falls under the single-sample rule → reported 1/0/0 (100% train).
- **Label 68:** total 1 (similarly 100% train).  
- **High-count labels (e.g., 153):** total 640  
  - `floor(0.7*640)=448`, `floor(0.2*640)=128`, remainder `64` → reported 448/128/64 (exact 70/20/10).

These examples align with the per-label reports in the notebook output: large classes follow the intended 70/20/10 split closely, small classes skew toward train to avoid empty partitions, and singleton labels are kept only in train by design.

## Label cardinality and count checks
- `convert_to_sparse_labels` produces **one integer label per sample**; the resulting label arrays are 1-D (`y_los.shape == (7955,)`, `y_nlos.shape == (12877,)`), so each example carries exactly one class id.
- Optional `remove_null_labels=True` remaps labels to a compact `0..N-1` range using only the labels that actually appear, eliminating empty IDs from the global label space.
- After concatenation, `y_all.shape == (20832,)`, matching the summed LOS+NLOS sample count (7955 + 12877).
- Split shapes match the requested 70/20/10 allocation: train 14,522; val 4,103; test 2,207, and these sum back to 20,832.
- The printed per-class counts come from `np.bincount(y_split)`. Summing those per-class rows equals the corresponding split sizes, which is only true if every sample contributes exactly one count (i.e., there are no multi-label entries).

## Conclusion

The `stratify_by_class` implementation performs a label-wise split with floored 70/20/10 proportions, safeguarding against empty train slices and negative counts. The reported train/val/test class counts match the expected outcomes of this logic given the provided dataset.
