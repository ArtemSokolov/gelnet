# gelnet

Generalized Elastic Nets

`gelnet` implements several extensions of the elastic net regularization scheme: individual
feature penalties for the L1 term, feature-feature penalties for the L2 term, and translation
coefficients for the L2 term. It supports linear regression, binary logistic regression and
one-class logistic regression models.

## References

- Sokolov, A., Carlin, D. E., Paull, E. O., Baertsch, R., & Stuart, J. M. (2016). Pathway-Based Genomics Prediction using Generalized Elastic Net. *PLOS Computational Biology*, 12(3), e1004790. https://doi.org/10.1371/journal.pcbi.1004790
- Sokolov, A., Paull, E. O., & Stuart, J. M. (2016). One-Class Detection of Cell States in Tumor Subtypes. *Pacific Symposium on Biocomputing*, 21, 405–416. https://doi.org/10.1142/9789814749411_0037

## Installation

The package can be installed directly from GitHub using

``` r
if( !require(devtools) ) install.packages("devtools")
devtools::install_github("ArtemSokolov/gelnet")
```

## A grammar of regularization

Inspired by [ggplot2](https://ggplot2.tidyverse.org/), gelnet models are assembled from independent terms using the `+` operator. A fully-specified model definition is then provided to `gelnet_train()` for training. Consider the following example:

``` r
library( gelnet )

model <- gelnet(X) + model_lin(y) + rglz_L1(0.1) + rglz_L2(1, P = L)
fit <- gelnet_train( model )
```

where the building blocks are:

- `gelnet(X)` &mdash; the base layer with the data matrix `X`. Every model definition starts here.
- A machine learning **task**:
  - `model_lin(y, ...)` &mdash; linear regression
  - `model_blr(y, ...)` &mdash; binary logistic regression
  - `model_oclr(...)` &mdash; one-class logistic regression (no `y` needed)
- **Regularizers**, describing how the model is penalized:
  - `rglz_L1(l1, d)` &mdash; L1 penalty, optionally weighted per-feature by `d`
  - `rglz_L2(l2, P, m)` &mdash; L2 penalty, optionally with a feature-feature penalty matrix `P`
    and translation coefficients `m`
  - `rglz_nf(nFeats, d)` &mdash; an alternative to `rglz_L1()` that searches for the L1
    coefficient yielding exactly `nFeats` non-zero weights
- `gel_init(w_init, b_init)` &mdash; optional initial values for the weights and bias

The terms can be added in any order, and a partially-built definition can be reused as the starting
point for several models:

``` r
base <- gelnet(X) + model_lin(y) + rglz_L2(1, P = L)

sparse    <- base + rglz_L1(0.1)
sparser   <- base + rglz_L1(0.5)
five_feat <- base + rglz_nf(5)
```

Once assembled, a definition is fit with `gelnet_train()`:

``` r
model <- gelnet_train( sparse )
```

## Applying a model

`gelnet_train()` returns a list with the fitted weights `w` (and, for supervised tasks, a bias
term `b`). Inference on new samples can be done directly with the dot product against the weights:

``` r
scores <- Xnew %*% fit$w + fit$b
```

For `model_lin()`, `scores` are the predicted response values directly. For `model_blr()` and
`model_oclr()`, `scores` are the logits of the underlying logistic model; convert them to
probabilities with the sigmoid function, e.g. `1 / (1 + exp(-scores))`.

See `vignette("basics")` for a complete walkthrough.
