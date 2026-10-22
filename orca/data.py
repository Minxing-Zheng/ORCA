"""Stage 0 -- synthetic data generation.

Every generator returns ``(X, Y)`` with ``X`` of shape ``(n, d_x)`` and ``Y`` of
shape ``(n, d_y)``.  The 1-D response datasets below are the ones used in the
synthetic experiments of the ORCA paper.

Unlike the original research code, every generator takes an explicit ``seed`` and
draws from a local ``numpy.random.Generator``, so a call is fully reproducible
and does not perturb the global NumPy random state.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_circles, make_moons, make_s_curve, make_swiss_roll

__all__ = [
    "SYNTHETIC_DATASETS",
    "generate_synthetic",
    "gen_two_spiral",
    "gen_unbalance_cluster",
    "gen_balance_cluster",
    "gen_bimodal_conditional_gaussian",
    "train_pool_split",
    "sample_cal_test_from_pool",
]

#: Datasets understood by :func:`generate_synthetic`.  The first six have a
#: scalar response; ``mix-gaussian`` has a 2-D response.
SYNTHETIC_DATASETS = (
    "s-shape",
    "circles",
    "moon",
    "roll",
    "spiral",
    "unbalance",
    "balance",
    "mix-gaussian",
)


def gen_two_spiral(n, seed=None):
    """Two interleaved spirals; ``X`` is the first coordinate, ``Y`` the second."""
    rng = np.random.default_rng(seed)
    n_half = n // 2
    theta = np.sqrt(rng.random(n_half)) * 2 * np.pi
    r = 2 * theta + np.pi

    data_a = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    data_b = np.stack([-r * np.cos(theta), -r * np.sin(theta)], axis=1)
    data = np.vstack([data_a, data_b])
    rng.shuffle(data)
    return data[:, 0].reshape(-1, 1), data[:, 1].reshape(-1, 1)


def gen_unbalance_cluster(n, seed=None):
    """Four Gaussian clusters with unequal spread (0.5 / 0.5 / 2 / 2)."""
    rng = np.random.default_rng(seed)
    n_each = n // 4
    sizes = [n_each] * 4
    sizes[0] += n - 4 * n_each

    centers = [[-10, -10], [-10, 10], [10, 10], [10, -10]]
    scales = [0.5, 0.5, 2.0, 2.0]
    blocks = [
        rng.standard_normal((size, 2)) * scale + np.asarray(center, dtype=float)
        for size, center, scale in zip(sizes, centers, scales)
    ]
    data = np.vstack(blocks)
    return data[:, 0].reshape(-1, 1), data[:, 1].reshape(-1, 1)


def gen_balance_cluster(n, seed=None):
    """Four Gaussian clusters with equal spread."""
    rng = np.random.default_rng(seed)
    n_each = n // 4
    sizes = [n_each] * 4
    sizes[0] += n - 4 * n_each

    centers = [[-5, -5], [-5, 5], [5, 5], [5, -5]]
    blocks = [
        rng.standard_normal((size, 2)) + np.asarray(center, dtype=float)
        for size, center in zip(sizes, centers)
    ]
    data = np.vstack(blocks)
    return data[:, 0].reshape(-1, 1), data[:, 1].reshape(-1, 1)


def gen_bimodal_conditional_gaussian(n, sigma_1=1.0, sigma_2=3.0, weight_1=0.75, seed=None):
    """2-D response drawn from a two-component Gaussian mixture whose means move with ``X``.

    ``X ~ U(0, 1)``; with probability ``weight_1`` the response is centred at
    ``(2X, 0)`` with scale ``sigma_1``, otherwise at ``(10 - 2X, 0)`` with scale
    ``sigma_2``.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, 1))
    Y = np.zeros((n, 2))

    first = rng.random(n) < weight_1
    mu1 = np.hstack([2.0 * X, np.zeros_like(X)])
    mu2 = np.hstack([10.0 - 2.0 * X, np.zeros_like(X)])

    Y[first] = rng.standard_normal((int(first.sum()), 2)) * sigma_1 + mu1[first]
    Y[~first] = rng.standard_normal((int((~first).sum()), 2)) * sigma_2 + mu2[~first]
    return X, Y


def generate_synthetic(name, n, noise=0.01, seed=None, sigma_1=1.0, sigma_2=3.0):
    """Generate one of :data:`SYNTHETIC_DATASETS`.

    Parameters
    ----------
    name : str
        Dataset key.
    n : int
        Number of observations.
    noise : float
        Observation noise for the scikit-learn manifolds (``s-shape``,
        ``circles``, ``moon``, ``roll``).
    seed : int or None
        Seed for the local random generator.
    sigma_1, sigma_2 : float
        Component scales, used by ``mix-gaussian`` only.

    Returns
    -------
    (X, Y) : tuple of np.ndarray
        Shapes ``(n, 1)`` and ``(n, d_y)``.
    """
    if name == "s-shape":
        pts, _ = make_s_curve(n_samples=n, noise=noise, random_state=seed)
        X, Y = pts[:, 0].reshape(-1, 1), pts[:, 2].reshape(-1, 1)
    elif name == "circles":
        pts, _ = make_circles(n_samples=n, factor=0.4, noise=noise, random_state=seed)
        X, Y = pts[:, 0].reshape(-1, 1), pts[:, 1].reshape(-1, 1)
    elif name == "moon":
        pts, _ = make_moons(n_samples=n, noise=noise, random_state=seed)
        X, Y = pts[:, 0].reshape(-1, 1), pts[:, 1].reshape(-1, 1)
    elif name == "roll":
        pts, _ = make_swiss_roll(n_samples=n, noise=noise, random_state=seed)
        X, Y = pts[:, 0].reshape(-1, 1), pts[:, 2].reshape(-1, 1)
    elif name == "spiral":
        X, Y = gen_two_spiral(n, seed=seed)
    elif name == "unbalance":
        X, Y = gen_unbalance_cluster(n, seed=seed)
    elif name == "balance":
        X, Y = gen_balance_cluster(n, seed=seed)
    elif name == "mix-gaussian":
        X, Y = gen_bimodal_conditional_gaussian(n, sigma_1=sigma_1, sigma_2=sigma_2, seed=seed)
    else:
        raise ValueError(
            f"Unknown dataset {name!r}. Available: {', '.join(SYNTHETIC_DATASETS)}"
        )

    X = np.asarray(X, dtype=float).reshape(n, -1)
    Y = np.asarray(Y, dtype=float).reshape(n, -1)
    return X, Y


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #

def train_pool_split(n, n_train, seed=0):
    """Split ``range(n)`` into a training set and an evaluation pool.

    The pool is later re-split into exploration / calibration / test many times,
    which lets a single trained model be reused across repetitions.
    """
    rng = np.random.default_rng(seed)
    idx_all = np.arange(n)
    idx_train = rng.choice(idx_all, size=n_train, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[idx_train] = False
    return np.sort(idx_train), np.sort(idx_all[mask])


def sample_cal_test_from_pool(idx_pool, n_calib, n_test, seed=0):
    """Draw disjoint calibration and test index sets out of the evaluation pool."""
    rng = np.random.default_rng(seed)
    need = n_calib + n_test
    if need > len(idx_pool):
        raise ValueError(f"n_calib + n_test = {need} exceeds pool size {len(idx_pool)}")
    chosen = rng.choice(idx_pool, size=need, replace=False)
    return chosen[:n_calib], chosen[n_calib:]
