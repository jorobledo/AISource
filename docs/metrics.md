# Multivariate distribution metrics

`aisource.metrics` compares complete particle vectors shaped
`(particles, parameters)`. The target use case is five to ten parameters per
particle. No per-parameter or projection-based metrics are included.

| Metric | Output | Interpretation |
|---|---|---|
| RBF MMD² | Kernel discrepancy | Lower means closer joint distributions |
| C2ST | Cross-validated accuracy, ROC AUC, p-value | Accuracy and AUC near 0.5 mean the classifier cannot distinguish the samples |
| Energy distance | Multivariate sample distance | Lower means closer joint distributions |

MMD and energy distance use pooled feature standardization by default. This
prevents a parameter from dominating merely because of its unit scale. The
scaling choice must remain fixed across comparisons.

C2ST uses balanced samples and out-of-fold predictions. Logistic regression is
the default classifier. A random forest option can detect nonlinear or
dependence differences that a linear classifier can miss. The classifier and
its settings must remain fixed across model comparisons.

## Example

```python
from aisource.metrics import evaluate

results = evaluate(
    reference_particles,
    generated_particles,
    seed=17,
    c2st_classifier="random_forest",
)
```

Quadratic metrics use `max_samples` to bound memory and runtime. Always record
this value, the seed, parameter order, and preprocessing with each result.

The MMD implementation follows the kernel two-sample framework of
[Gretton et al.](https://www.jmlr.org/papers/v13/gretton12a.html). C2ST follows
[Lopez-Paz and Oquab](https://arxiv.org/abs/1610.06545).
