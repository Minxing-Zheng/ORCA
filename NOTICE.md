# Third-party attribution

This release is self-contained: no third-party research code is vendored in.
The attributions below record intellectual provenance for the ideas and
implementations the code follows.

## Quantile network (`orca/models.py`)

`QuantileNet` is a standard multi-quantile MLP trained with the pinball loss.
The architecture (two hidden layers, orthogonal initialisation, no-crossing by
sorting the outputs) and the loss follow the conditional quantile network used
in the reference implementation of **Conformal Histogram Regression**:

> Sesia, M. and Romano, Y. *Conformal Prediction using Conditional Histograms.*
> NeurIPS 2021. https://github.com/msesia/chr

The sampling routine (inverting the fitted conditional quantile function at a
random level) is our addition -- it is what turns a quantile regressor into the
generative sampler ORCA needs.

## PCP baseline (`orca.conformal.PCP`)

The single-radius baseline implements the method of:

> Wang, Z., Gao, R., Yin, M., Zhou, M. and Blei, D. *Probabilistic Conformal
> Prediction Using Conditional Random Samples.* AISTATS 2023.
> https://github.com/Zhendong-Wang/Probabilistic-Conformal-Prediction

The implementation here is our own; the two-line calibration rule is the
method's definition.

## Worst-slab conditional coverage (`orca.metrics.worst_slab_coverage`)

The WSC diagnostic is due to:

> Cauchois, M., Gupta, S. and Duchi, J. *Knowing what you know: valid and
> validated confidence sets in multiclass and multilabel prediction.* JMLR 2021,
> as adapted for regression by Romano, Sesia and Candès.

Our implementation searches on one split and reports on the held-out split.

## Solvers

Optimization uses **OR-Tools** (Apache License 2.0) -- CP-SAT for the MILP and
GLOP for the LP relaxation. https://github.com/google/or-tools

## Datasets

The synthetic generators build on `sklearn.datasets` manifolds
(`make_s_curve`, `make_circles`, `make_moons`, `make_swiss_roll`); scikit-learn
is BSD-3-Clause licensed. The cluster and spiral generators are our own.
