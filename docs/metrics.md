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


### Kullback Leibler divergence

We use the Kullback-Leibler Divergence (KLD), after we transform both of our samples into a gaussian space using [Gaussian Rank transformation](https://link.springer.com/rwe/10.1007/978-3-662-69359-9_511). 
By doing this we can apply the parametric KLD onto our data that is non-parametric.

The theory behind the KLD is here briefly explained based on [this article](https://arxiv.org/html/2604.11744v1) and [this wikipedia article](https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices):


For two empirical samples from multivariate gaussian distributions, $P$ and $Q$ , the KLD can be calculated as 
$D_{KL}(P||Q) = \frac{1}{2}\left(\text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_1 - \mu_2)^T\Sigma_2^{-1}(]mu_1-\mu_2) - k + \log\left(\frac{|\Sigma_2|}{|\Sigma_1|}\right)\right)$

where $\Sigma_i$ is the covariance matrix of the $i$'th distribution, and $\mu_i$ is the mean vector of the $i$'th distribution.

The covariance matrix is estimated as,

$\Sigma = E[(X-E[X])(X-E[X])^T]$

Or equivalently,

$\Sigma = \frac{1}{n-1}\sum^n_i (x_i-\mu)(x_i-\mu)^T]$

Where $\mu$ is calculated as,

$\mu = \frac{1}{n} \sum^n_i x_i$

which is then the mean vector. In the above, $X$ is the collection of all vectors, whereas $x_i$ is a specific sampled vector. 




