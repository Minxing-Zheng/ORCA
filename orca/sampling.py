"""Stage 2 -- generative sampling, local-density ranking, residual matrices.

ORCA turns a generative model into a *rank-indexed* nonconformity structure:

1. draw ``K`` samples ``Yhat[i, 1..K]`` from ``p(Y | X_i)``;
2. rank them by a local density proxy so that rank 1 is the sample sitting in
   the densest part of the cloud and rank ``K`` the most isolated one;
3. record the residual matrix ``E[i, r] = dist(Y_i, Yhat_ranked[i, r])``.

The optimiser in :mod:`orca.optimize` then chooses one radius per *rank*, which
is what lets the prediction set breathe with the local geometry of ``Y | X``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.distance import cdist

__all__ = [
    "generate_samples",
    "rank_samples_by_density",
    "residual_matrix",
    "build_sample_matrices",
]


def generate_samples(model, X, n_sample, rng=None, shared_quantile_draw=True):
    """Draw ``n_sample`` samples per row of ``X``.

    Accepts either the ``sample(X, n_sample, rng)`` interface used by
    :mod:`orca.models` or a legacy ``sample(X) -> (X, y)`` single-draw sampler.

    Returns
    -------
    np.ndarray of shape ``(n, n_sample, d_y)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    X = np.asarray(X)

    try:
        out = model.sample(X, n_sample, rng=rng, shared_quantile_draw=shared_quantile_draw)
        return np.asarray(out, dtype=float).reshape(X.shape[0], n_sample, -1)
    except TypeError:
        pass  # fall through to the single-draw interface

    draws = []
    for _ in range(n_sample):
        out = model.sample(X)
        y = out[1] if isinstance(out, (list, tuple)) and len(out) == 2 else out
        y = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)
        draws.append(y.reshape(X.shape[0], 1, -1))
    return np.concatenate(draws, axis=1).astype(float)


def rank_samples_by_density(Y_hat, k_neighbor=None):
    """Order each row's samples from densest to most isolated.

    The density proxy for a sample is its mean distance to its ``k_neighbor``
    nearest neighbours within the same cloud (the sample itself included, which
    contributes a zero).  Small proxy value == high local density == low rank
    index.

    Parameters
    ----------
    Y_hat : np.ndarray, shape (n, K, d_y)
    k_neighbor : int or None
        Neighbourhood size; defaults to ``ceil(K / 3)``.

    Returns
    -------
    np.ndarray of shape ``(n, K, d_y)``, each row permuted by increasing proxy.
    """
    Y_hat = np.asarray(Y_hat, dtype=float)
    if Y_hat.ndim != 3:
        raise ValueError(f"Y_hat must be (n, K, d_y); got shape {Y_hat.shape}")
    n, K, _ = Y_hat.shape
    if k_neighbor is None:
        k_neighbor = math.ceil(K / 3)
    k_neighbor = int(min(max(k_neighbor, 1), K - 1)) if K > 1 else 0

    ranked = np.empty_like(Y_hat)
    for i in range(n):
        cloud = Y_hat[i]
        pairwise = cdist(cloud, cloud)
        # Column 0 of the sorted distances is the self-distance (0.0).
        proxy = np.sort(pairwise, axis=1)[:, : k_neighbor + 1].mean(axis=1)
        ranked[i] = cloud[np.argsort(proxy, kind="stable")]
    return ranked


def residual_matrix(Y_true, Y_samples, norm="l2"):
    """Distances between each true response and its samples.

    Parameters
    ----------
    Y_true : np.ndarray, shape (n, d_y)
    Y_samples : np.ndarray, shape (n, K, d_y)
    norm : {"l1", "l2", "linf"}

    Returns
    -------
    np.ndarray of shape ``(n, K)``.
    """
    Y_true = np.asarray(Y_true, dtype=float)
    Y_samples = np.asarray(Y_samples, dtype=float)
    if Y_true.ndim == 1:
        Y_true = Y_true.reshape(-1, 1)
    if Y_true.shape[0] != Y_samples.shape[0]:
        raise ValueError("Y_true and Y_samples disagree on the number of points")
    if Y_true.shape[1] != Y_samples.shape[2]:
        raise ValueError("Y_true and Y_samples disagree on the response dimension")

    delta = Y_samples - Y_true[:, None, :]
    key = str(norm).lower()
    if key == "l1":
        return np.sum(np.abs(delta), axis=-1)
    if key == "l2":
        return np.linalg.norm(delta, axis=-1)
    if key in ("linf", "l_inf", "inf"):
        return np.max(np.abs(delta), axis=-1)
    raise ValueError("norm must be one of 'l1', 'l2', 'linf'")


def build_sample_matrices(model, X, Y, n_sample, rng=None, norm="l2",
                          k_neighbor=None, shared_quantile_draw=True):
    """Run stage 2 end to end for one index set.

    Returns
    -------
    dict with keys
        ``Y_hat``          raw samples, ``(n, K, d_y)``
        ``Y_hat_ranked``   density-ranked samples, ``(n, K, d_y)``
        ``E``              residuals to the ranked samples, ``(n, K)`` -- ORCA
        ``E_unranked``     residuals to the raw samples, ``(n, K)``     -- PCP
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    Y_hat = generate_samples(model, X, n_sample, rng=rng,
                             shared_quantile_draw=shared_quantile_draw)
    Y_hat_ranked = rank_samples_by_density(Y_hat, k_neighbor=k_neighbor)
    return {
        "Y_hat": Y_hat,
        "Y_hat_ranked": Y_hat_ranked,
        "E": residual_matrix(Y, Y_hat_ranked, norm=norm),
        "E_unranked": residual_matrix(Y, Y_hat, norm=norm),
    }
