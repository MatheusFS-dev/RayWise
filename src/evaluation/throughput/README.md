# Throughput Evaluation

This package implements the shortlist-based throughput metric

`R_T(k) = sum(log2(1 + best oracle gain inside predicted top-k)) / sum(log2(1 + oracle best gain))`

for the repo-local S009 baseline dataset under `src/data/s009`.

The package does not add a CLI. Use it from Python:

```python
from compute_throughput_ratio import (
    build_s009_oracle_beam_scores_archive,
)
from evaluate_models_throughput import (
    evaluate_saved_models_throughput,
)

build_s009_oracle_beam_scores_archive()
results = evaluate_saved_models_throughput()
```

Default evaluated artifacts:

- `weights/nas_best_architectures_cnn1d/optuna_study/model/top_1_trial_4026.keras`
- `weights/nas_multiobjective_best_knowledge_distillation/optuna_study/model/top_1_trial_721.keras`
- `weights/ensemble_cnn1d_stacking/autogluon_meta_20260114_151507`

Outputs are written under `src/evaluation/throughput/outputs/`:

- `s009_oracle_beam_scores.npz`
- `*_scores_s009.npz`
- `*_throughput.json`
- `throughput_summary.csv`
