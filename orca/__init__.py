r"""ORCA -- generative conformal prediction with Optimized Ranking and Coverage Allocation.

The pipeline has five stages, one module each:

===== ============================ ==============================================
stage module                       what it does
===== ============================ ==============================================
0     :mod:`orca.data`             generate or load data, split it
1     :mod:`orca.models`           fit a generative model of ``p(Y | X)``
2     :mod:`orca.sampling`         sample, rank by local density, form residuals
3     :mod:`orca.optimize`         optimize the rank-dependent radius allocation
4     :mod:`orca.conformal`        split-conformal calibration (PCP and ORCA)
5     :mod:`orca.plotting`         figures
\-    :mod:`orca.metrics`          coverage, set size, worst-slab diagnostics
===== ============================ ==============================================

Minimal example::

    from orca import data, models, sampling, conformal, metrics

    X, Y = data.generate_synthetic("s-shape", n=4000, seed=0)
    idx_train, idx_pool = data.train_pool_split(len(X), 2000, seed=0)

    model = models.QuantileNet(num_features=X.shape[1], random_state=0)
    model.fit(X[idx_train], Y[idx_train])

    mats = sampling.build_sample_matrices(model, X[idx_pool], Y[idx_pool], n_sample=20)
    # ... split idx_pool into exploration / calibration / test, then:
    orca = conformal.ORCA(alpha=0.1, solver="milp").fit_calibrate(E_explore, E_calib)
    covered = orca.coverage(E_test)
"""

__version__ = "1.0.0"

from . import conformal, data, metrics, models, optimize, plotting, sampling

__all__ = [
    "conformal",
    "data",
    "metrics",
    "models",
    "optimize",
    "plotting",
    "sampling",
    "__version__",
]
