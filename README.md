## Experimental Setup

### Regression

- Dataset: synthetic regression using `make_regression`
- Samples: 5000
- Features: 20
- Split: 70/15/15
- Model: MLP 20 -> 64 -> 32 -> 1
- Losses: MSE, MAE, Huber
- Noise levels: 0%, 10%, 20%
- Metrics: MAE, RMSE

### Classification

- Dataset: synthetic balanced 3-class classification
- Samples: 5000
- Features: 20
- Split: 70/15/15
- Model: MLP 20 -> 64 -> 32 -> 3
- Losses: CE, GCE, SCE
- Noise levels: 0%, 20%, 40%
- Metrics: Accuracy, Macro-F1

### Reproducibility

Seeds:

- 42
- 123
- 2026

Noise is applied only to training targets/labels.
Validation and test sets remain clean.

### Hyperparameter Tuning

Loss-specific hyperparameters are selected using
clean validation performance.

Regression tuning noise:
10%

Classification tuning noise:
20%

Selected Huber delta factor:
<RESULT>

Selected GCE q:
<RESULT>

Selected SCE alpha:
<RESULT>

Selected SCE beta:
<RESULT>

The test set is not used during tuning or Week 2 baseline experiments.