# CNN2D NAS 2000 Trials Top-1 Report

## Model

- Model: `model/top_1_trial_1154.keras`
- Rank: 1
- Trial ID: 1154
- Objective value: 1.5801450014
- Convergence epoch: 27
- Best epoch: 35

## Model Statistics

| Metric | Value |
| --- | ---: |
| Parameters | 4,874,368 |
| Model size | 18.59 MB |
| FLOPs | 1.04 T |
| Inference time average | 28.991 ms |
| Inference time peak | 29.546 ms |

## Evaluation Top-K Results

### Test on Baseline

| K | Accuracy | Correct / Total |
| ---: | ---: | ---: |
| 1 | 57.62% | 5553 / 9638 |
| 3 | 81.10% | 7816 / 9638 |
| 5 | 86.39% | 8326 / 9638 |
| 10 | 92.33% | 8899 / 9638 |
| 20 | 95.81% | 9234 / 9638 |
| 30 | 97.54% | 9401 / 9638 |
| 50 | 98.51% | 9494 / 9638 |

### Test on Raw

| K | Accuracy | Correct / Total |
| ---: | ---: | ---: |
| 1 | 57.62% | 5553 / 9638 |
| 3 | 81.10% | 7816 / 9638 |
| 5 | 86.39% | 8326 / 9638 |
| 10 | 92.33% | 8899 / 9638 |
| 20 | 95.81% | 9234 / 9638 |
| 30 | 97.54% | 9401 / 9638 |
| 50 | 98.51% | 9494 / 9638 |

### Raw LOS

LOS samples: 1473

| K | Accuracy | Correct / Total |
| ---: | ---: | ---: |
| 1 | 65.51% | 965 / 1473 |
| 3 | 89.21% | 1314 / 1473 |
| 5 | 94.70% | 1395 / 1473 |
| 10 | 97.22% | 1432 / 1473 |
| 20 | 98.30% | 1448 / 1473 |
| 30 | 99.32% | 1463 / 1473 |
| 50 | 99.59% | 1467 / 1473 |

### Raw NLOS

NLOS samples: 8165

| K | Accuracy | Correct / Total |
| ---: | ---: | ---: |
| 1 | 56.19% | 4588 / 8165 |
| 3 | 79.63% | 6502 / 8165 |
| 5 | 84.89% | 6931 / 8165 |
| 10 | 91.45% | 7467 / 8165 |
| 20 | 95.36% | 7786 / 8165 |
| 30 | 97.22% | 7938 / 8165 |
| 50 | 98.31% | 8027 / 8165 |

## Source

- Evaluation script: `src/architectures/evaluate/evaluate_raw.py`
- Evaluation environment: `/home/matheus/.venvs/cuda`
- Model metadata source: `args/top_1_trial.txt`
