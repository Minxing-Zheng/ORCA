"""Evaluation metrics: coverage, prediction-set size, conditional coverage.

The prediction set for a test point is a union of balls centred at its generated
samples,

    C(X) = union_r  B( Yhat_r(X), R_r ) ,

so "efficiency" means the size of that union: total interval length when
``d_y = 1``, and Monte-Carlo volume when ``d_y > 1``.  Overlapping balls are
merged, which matters -- summing ball sizes would badly overstate the set.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["region_size", "coverage_rate", "worst_slab_coverage", "summarize"]


def _as_radius_vector(radius, n_sample):
    r = np.asarray(radius, dtype=float)
    if r.ndim == 0:
        return np.full(n_sample, float(r))
    if r.shape[0] != n_sample:
        raise ValueError(f"radius has length {r.shape[0]} but there are {n_sample} samples")
    return r


def region_size(Y_samples, radius, reference_data=None, dimension_y=1,
                n_samples_per_dim=120, norm="l2", seed=0):
    """Size of the union of balls around each row's samples.

    Parameters
    ----------
    Y_samples : np.ndarray
        ``(n_test, K)`` or ``(n_test, K, d_y)`` generated samples -- use the same
        ranking that produced the residual matrix the radii were fit on.
    radius : float or array-like of shape (K,)
        Scalar for PCP, per-rank vector for ORCA.
    reference_data : np.ndarray or None
        Observed responses, used to bound the region.  For ``d_y = 1`` the length
        is clipped to the observed range; for ``d_y > 1`` its bounding box is the
        Monte-Carlo integration domain (required).
    n_samples_per_dim : int
        Monte-Carlo resolution per dimension when ``d_y > 1``
        (``n_samples_per_dim ** d_y`` Sobol points in total).
    norm : {"l2", "linf"}
        Ball shape for ``d_y > 1``.
    seed : int
        Seed of the scrambled Sobol sequence, so volumes are reproducible.

    Returns
    -------
    np.ndarray of shape ``(n_test,)``.
    """
    Y = np.asarray(Y_samples, dtype=float)
    if dimension_y == 1 and Y.ndim == 3:
        Y = Y[:, :, 0]
    n_test, n_sample = Y.shape[0], Y.shape[1]
    r = _as_radius_vector(radius, n_sample)

    if dimension_y == 1:
        span = None
        if reference_data is not None:
            ref = np.asarray(reference_data, dtype=float)
            span = float(ref.max() - ref.min())

        lengths = np.empty(n_test, dtype=float)
        for i in range(n_test):
            intervals = np.column_stack([Y[i] - r, Y[i] + r])
            intervals = intervals[np.argsort(intervals[:, 0], kind="stable")]
            total, cur_l, cur_r = 0.0, intervals[0, 0], intervals[0, 1]
            for left, right in intervals[1:]:
                if left > cur_r:
                    total += cur_r - cur_l
                    cur_l, cur_r = left, right
                else:
                    cur_r = max(cur_r, right)
            total += cur_r - cur_l
            lengths[i] = min(total, span) if span is not None else total
        return lengths

    if reference_data is None:
        raise ValueError("reference_data is required to bound the volume when dimension_y > 1")

    ref = np.asarray(reference_data, dtype=float).reshape(-1, dimension_y)
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    box_volume = float(np.prod(hi - lo))

    n_points = int(n_samples_per_dim ** dimension_y)
    sobol = torch.quasirandom.SobolEngine(dimension=dimension_y, scramble=True, seed=seed)
    grid = sobol.draw(n_points).numpy() * (hi - lo) + lo

    key = str(norm).lower()
    volumes = np.empty(n_test, dtype=float)
    for i in range(n_test):
        delta = grid[:, None, :] - Y[i]
        if key == "l2":
            dists = np.linalg.norm(delta, axis=2)
        elif key in ("linf", "l_inf", "inf"):
            dists = np.max(np.abs(delta), axis=2)
        else:
            raise ValueError("norm must be 'l2' or 'linf'")
        volumes[i] = float(np.mean(np.any(dists <= r, axis=1)) * box_volume)
    return volumes


def coverage_rate(E_test, radius):
    """Marginal coverage of a union-of-balls set with the given radii."""
    E = np.asarray(E_test, dtype=float)
    r = np.asarray(radius, dtype=float)
    r = r[None, :] if r.ndim == 1 else r
    return float(np.mean(np.any(E <= r, axis=1)))


def worst_slab_coverage(X, covered, delta=0.2, M=200, test_size=0.75, seed=42):
    """Worst-slab conditional coverage (WSC), the Cauchy--Romano--Sesia diagnostic.

    Searches random directions ``v`` for the slab ``a <= v'X <= b`` holding at
    least a ``delta`` fraction of points whose coverage is lowest.  The search
    runs on one split and the value is reported on the held-out split, which
    removes most of the optimistic bias of an in-sample search.

    Returns ``(wsc_value, v, a, b)``.
    """
    X = np.asarray(X, dtype=float)
    X = X.reshape(len(X), -1)
    covered = np.asarray(covered, dtype=float).reshape(-1)
    rng = np.random.default_rng(seed)

    n = X.shape[0]
    idx = rng.permutation(n)
    n_tr = int(round((1.0 - test_size) * n))
    if n_tr < 2 or n - n_tr < 2:
        raise ValueError("not enough points to split for the WSC search")
    tr, te = idx[:n_tr], idx[n_tr:]

    V = rng.standard_normal((M, X.shape[1]))
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    best = (np.inf, V[0], -np.inf, np.inf)
    for v in V:
        z = X[tr] @ v
        order = np.argsort(z, kind="stable")
        z_sorted, c_sorted = z[order], covered[tr][order]
        cum = np.concatenate([[0.0], np.cumsum(c_sorted)])
        m = len(z_sorted)
        min_pts = max(2, int(np.ceil(delta * m)))

        for i in range(m - min_pts + 1):
            j_lo = i + min_pts
            counts = np.arange(j_lo, m + 1) - i
            means = (cum[j_lo : m + 1] - cum[i]) / counts
            k = int(np.argmin(means))
            if means[k] < best[0]:
                best = (float(means[k]), v, float(z_sorted[i]), float(z_sorted[j_lo + k - 1]))

    _, v, a, b = best
    z_te = X[te] @ v
    inside = (z_te >= a) & (z_te <= b)
    wsc = float(covered[te][inside].mean()) if inside.sum() else float(covered[te].mean())
    return wsc, v, a, b


def summarize(coverage, sizes):
    """Headline numbers for one run."""
    coverage = np.asarray(coverage, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    covered_idx = np.nonzero(coverage > 0)[0]
    return {
        "coverage": float(coverage.mean()),
        "size_mean": float(sizes.mean()),
        "size_median": float(np.median(sizes)),
        "size_mean_covered": float(sizes[covered_idx].mean()) if covered_idx.size else float("nan"),
    }
