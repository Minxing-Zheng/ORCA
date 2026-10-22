"""Stage 4 -- split-conformal calibration.

Two predictors are provided.

``PCP``  (Probabilistic Conformal Prediction, Wang et al. 2023) takes a *single*
radius shared by all generated samples, calibrated from ``min_r E[i, r]``.

``ORCA`` takes the rank-dependent radius *direction* ``t`` produced by
:mod:`orca.optimize` on the exploration half, then rescales it on the disjoint
calibration half:

    S_i = min_r E[i, r] / t_r ,   q = Quantile_{1-alpha}(S) ,   R = q * t .

Because ``t`` is fixed before the calibration half is seen, the scores ``S_i``
are exchangeable with the test score, so

    P( Y_test in C(X_test) ) >= 1 - alpha

holds in finite samples whatever the optimiser returned -- an inexact solve
costs efficiency, never validity.  This is why splitting the calibration data
into an exploration half and a calibration half is not optional.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["conformal_quantile", "split_exploration_calibration", "PCP", "ORCA"]


def conformal_quantile(scores, alpha):
    """The ``ceil((1 - alpha)(m + 1))``-th smallest score (capped at the largest).

    This is the finite-sample conformal quantile: using it as a threshold gives
    coverage at least ``1 - alpha`` for an exchangeable test score.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    m = s.shape[0]
    if m == 0:
        raise ValueError("cannot compute a conformal quantile from an empty score set")
    k = int(math.ceil((1.0 - alpha) * (m + 1)))
    return float(s[int(np.clip(k, 1, m)) - 1])


def split_exploration_calibration(n, seed=0, frac=0.5):
    """Random disjoint split of ``range(n)`` into exploration and calibration halves."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(frac * n))
    return perm[:cut], perm[cut:]


class PCP:
    """Probabilistic conformal prediction with one radius for all samples.

    Reference: Wang, Gao & Zhou, *Probabilistic Conformal Prediction Using
    Conditional Random Samples*, AISTATS 2023.
    """

    def __init__(self, alpha=0.1, dimension_y=1):
        self.alpha = float(alpha)
        self.dimension_y = int(dimension_y)
        self.radius = None

    def calibrate(self, E_calib):
        """Calibrate on residuals ``E_calib`` of shape ``(n_cal, K)``."""
        scores = np.min(np.asarray(E_calib, dtype=float), axis=1)
        self.radius = conformal_quantile(scores, self.alpha)
        return self.radius

    def coverage(self, E_test):
        """Boolean coverage indicator per test point."""
        if self.radius is None:
            raise RuntimeError("call calibrate() before coverage()")
        return np.any(np.asarray(E_test, dtype=float) <= self.radius, axis=1)

    @property
    def radius_vector(self):
        """Radius broadcast to one value per sample, for the size estimators."""
        if self.radius is None:
            raise RuntimeError("call calibrate() before using radius_vector")
        return self.radius


class ORCA:
    """Generative conformal prediction with optimized ranking and coverage allocation.

    Parameters
    ----------
    alpha : float
        Target miscoverage.
    dimension_y : int
        Response dimension; drives the proxy objective ``sum_r t_r ** d``.
    solver : {"milp", "lp", "greedy"} or callable
        Stage-3 coverage-allocation solver.  A callable must have the signature
        ``solver(E, alpha, d=..., **kwargs) -> dict`` with a ``"thresholds"`` key.
    **solver_kwargs
        Forwarded to the solver (e.g. ``time_limit``, ``with_zero``, ``tighten``).

    Attributes
    ----------
    thresholds : np.ndarray
        Rank-dependent radius direction from :meth:`fit`.
    scale : float
        Conformal multiplier from :meth:`calibrate`.
    radius : np.ndarray
        ``scale * thresholds`` -- the deployed per-rank radii.
    """

    def __init__(self, alpha=0.1, dimension_y=1, solver="milp", **solver_kwargs):
        self.alpha = float(alpha)
        self.dimension_y = int(dimension_y)
        self.solver = solver
        self.solver_kwargs = solver_kwargs

        self.thresholds = None
        self.scale = None
        self.radius = None
        self.solver_result = None

    # -- stage 3 ------------------------------------------------------------ #
    def fit(self, E_explore):
        """Solve the coverage-allocation program on the exploration residuals."""
        from . import optimize as _optimize

        solver = self.solver
        if isinstance(solver, str):
            try:
                solver = _optimize.SOLVERS[solver]
            except KeyError:
                raise ValueError(
                    f"unknown solver {self.solver!r}; choose from {sorted(_optimize.SOLVERS)}"
                ) from None

        result = solver(np.asarray(E_explore, dtype=float), self.alpha,
                        d=self.dimension_y, **self.solver_kwargs)
        self.solver_result = result
        self.thresholds = np.asarray(result["thresholds"], dtype=float)
        return self

    # -- stage 4 ------------------------------------------------------------ #
    def calibrate(self, E_calib):
        """Rescale the radius direction on the held-out calibration residuals.

        A zero entry in the threshold direction represents a zero-radius ball:
        it covers an observation only when the corresponding residual is exactly
        zero.  Positive residuals therefore receive an infinite ratio at that
        rank instead of being divided by an arbitrary numerical epsilon.
        """
        if self.thresholds is None:
            raise RuntimeError("call fit() before calibrate()")
        E = np.asarray(E_calib, dtype=float)
        if E.ndim != 2:
            raise ValueError(f"E_calib must be 2-D (n_cal, K); got shape {E.shape}")
        if E.shape[1] != self.thresholds.shape[0]:
            raise ValueError(
                f"E_calib has {E.shape[1]} ranks but thresholds has "
                f"length {self.thresholds.shape[0]}"
            )
        if np.any(self.thresholds < 0):
            raise ValueError("thresholds must be nonnegative")

        ratios = np.full(E.shape, np.inf, dtype=float)
        positive = self.thresholds > 0
        ratios[:, positive] = E[:, positive] / self.thresholds[None, positive]
        ratios[:, ~positive] = np.where(E[:, ~positive] == 0.0, 0.0, np.inf)
        scores = np.min(ratios, axis=1)
        self.scale = conformal_quantile(scores, self.alpha)
        # Avoid the undefined floating-point product ``inf * 0``. Zero entries
        # remain zero-radius balls when at least one positive direction exists.
        self.radius = np.zeros_like(self.thresholds)
        if positive.any():
            self.radius[positive] = self.scale * self.thresholds[positive]
        elif np.isinf(self.scale):
            # An all-zero direction cannot cover a positive residual at any
            # finite scale. The infinite score level is the whole output space.
            self.radius.fill(np.inf)
        return self.radius

    def fit_calibrate(self, E_explore, E_calib):
        """:meth:`fit` then :meth:`calibrate`; returns ``self`` so calls chain."""
        self.fit(E_explore)
        self.calibrate(E_calib)
        return self

    # -- deployment --------------------------------------------------------- #
    def coverage(self, E_test):
        """Boolean coverage indicator per test point."""
        if self.radius is None:
            raise RuntimeError("call calibrate() before coverage()")
        return np.any(np.asarray(E_test, dtype=float) <= self.radius[None, :], axis=1)

    @property
    def radius_vector(self):
        if self.radius is None:
            raise RuntimeError("call calibrate() before using radius_vector")
        return self.radius
