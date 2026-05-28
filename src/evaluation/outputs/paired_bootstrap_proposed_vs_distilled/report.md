# Paired Bootstrap Report: Proposed vs Proposed-Distilled

## Setup

- Reference model: `/home/matheus/src/RayWise/weights/nas_best_architectures_cnn1d/optuna_study/model/top_1_trial_4026.keras`
- Candidate model: `/home/matheus/src/RayWise/weights/nas_multiobjective_best_knowledge_distillation/optuna_study/model/top_1_trial_721.keras`
- Bootstrap resamples: 10000
- Confidence level: 0.95
- Random seed: 20260528

## Results

| Metric | Reference | Candidate | Difference (candidate - reference) | Confidence interval |
| --- | ---: | ---: | ---: | ---: |
| Top-1 | 61.14% | 61.48% | 0.33% | [-0.25%, 0.90%] |
| Top-10 | 92.06% | 92.30% | 0.24% | [-0.06%, 0.54%] |

## Interpretation

The key manuscript comparison is the top-1 row above.
For top-1, the MOTPE-distilled student changes accuracy by 0.33% relative to the joint 1D CNN, with a 95% paired-bootstrap interval of [-0.25%, 0.90%].
