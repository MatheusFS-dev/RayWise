# CNN2D NAS 1000 Trials Top-1 Report

## Model

- Model: `model/top_1_trial_52.keras`
- Rank: 1
- Trial ID: 52
- Objective value: 1.6802046299
- Convergence epoch: 28
- Best epoch: 36

## Model Statistics

| Metric | Value |
| --- | ---: |
| Parameters | 649,600 |
| Model size | 2.48 MB |
| FLOPs | 34.08 G |
| Inference time average | 4.6916 ms |
| Inference time peak | 5.2178 ms |

## Evaluation Top-K Results

### Test on Baseline

| K | Accuracy | Correct / Total |
| ---: | ---: | ---: |
| 1 | 54.93% | 5294 / 9638 |
| 3 | 78.85% | 7600 / 9638 |
| 5 | 84.89% | 8182 / 9638 |
| 10 | 91.07% | 8777 / 9638 |
| 20 | 95.83% | 9236 / 9638 |
| 30 | 97.40% | 9387 / 9638 |
| 50 | 98.42% | 9486 / 9638 |

### Test on Raw

| K | Accuracy | Correct / Total |
| ---: | ---: | ---: |
| 1 | 54.93% | 5294 / 9638 |
| 3 | 78.85% | 7600 / 9638 |
| 5 | 84.89% | 8182 / 9638 |
| 10 | 91.07% | 8777 / 9638 |
| 20 | 95.83% | 9236 / 9638 |
| 30 | 97.40% | 9387 / 9638 |
| 50 | 98.42% | 9486 / 9638 |

### Raw LOS

LOS samples: 1473

| K | Accuracy | Correct / Total |
| ---: | ---: | ---: |
| 1 | 55.06% | 811 / 1473 |
| 3 | 83.84% | 1235 / 1473 |
| 5 | 90.16% | 1328 / 1473 |
| 10 | 95.25% | 1403 / 1473 |
| 20 | 97.90% | 1442 / 1473 |
| 30 | 98.57% | 1452 / 1473 |
| 50 | 99.46% | 1465 / 1473 |

### Raw NLOS

NLOS samples: 8165

| K | Accuracy | Correct / Total |
| ---: | ---: | ---: |
| 1 | 54.91% | 4483 / 8165 |
| 3 | 77.95% | 6365 / 8165 |
| 5 | 83.94% | 6854 / 8165 |
| 10 | 90.31% | 7374 / 8165 |
| 20 | 95.46% | 7794 / 8165 |
| 30 | 97.18% | 7935 / 8165 |
| 50 | 98.24% | 8021 / 8165 |

## Source

- Evaluation script: `src/architectures/evaluate/evaluate_raw.py`
- Evaluation environment: `/home/matheus/.venvs/cuda`
- Model metadata source: `args/top_1_trial.txt`
