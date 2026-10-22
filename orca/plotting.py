"""Stage 5 -- figures.

Publication-oriented matplotlib figures for the ORCA pipeline.  Every function
takes an optional ``ax`` so panels compose, and returns the axes it drew on.

Palette
-------
Categorical hues are taken in fixed slot order -- blue, orange, aqua -- from a
palette validated for colour-vision deficiency on all pairs in both light and
dark surfaces.  Do not cycle or generate further hues: with more than three
series, facet into small multiples instead.  Coverage outcomes use the reserved
status colours (never a categorical slot) and are always paired with a label, so
the distinction never rests on colour alone.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

__all__ = [
    "PALETTE",
    "METHOD_COLORS",
    "apply_style",
    "plot_dataset",
    "plot_prediction_sets_1d",
    "plot_radius_by_rank",
    "plot_coverage_efficiency",
]

#: Chart chrome, categorical slots (fixed order) and reserved status colours.
PALETTE = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series_1": "#2a78d6",   # blue
    "series_2": "#eb6834",   # orange
    "series_3": "#1baf7a",   # aqua
    "status_good": "#0ca30c",
    "status_critical": "#d03b3b",
}

#: Stable method -> hue map.  Colour follows the method, never its rank in a
#: sorted table, so filtering or reordering never repaints a series.
METHOD_COLORS = {
    "ORCA-MILP": PALETTE["series_1"],
    "ORCA-LP": PALETTE["series_1"],
    "ORCA-Greedy": PALETTE["series_1"],
    "ORCA": PALETTE["series_1"],
    "PCP": PALETTE["series_2"],
}


def apply_style():
    """Recessive, thin-mark matplotlib defaults used by every figure here."""
    plt.rcParams.update({
        "figure.facecolor": PALETTE["surface"],
        "axes.facecolor": PALETTE["surface"],
        "axes.edgecolor": PALETTE["axis"],
        "axes.linewidth": 0.8,
        "axes.labelcolor": PALETTE["ink_secondary"],
        "axes.titlecolor": PALETTE["ink"],
        "axes.titlesize": 11,
        "axes.titleweight": "medium",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",          # solid hairline; dashes are reserved for thresholds
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _new_ax(ax, figsize):
    if ax is None:
        apply_style()
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _despine(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_dataset(X, Y, ax=None, title=None, max_points=4000, seed=0, figsize=(5.0, 3.6)):
    """Scatter of a 1-D-response dataset.

    One series, so one hue and no legend -- the title names what is plotted.
    """
    ax = _new_ax(ax, figsize)
    X = np.asarray(X).reshape(len(X), -1)[:, 0]
    Y = np.asarray(Y).reshape(len(Y), -1)[:, 0]

    if len(X) > max_points:
        idx = np.random.default_rng(seed).choice(len(X), size=max_points, replace=False)
        X, Y = X[idx], Y[idx]

    ax.scatter(X, Y, s=6, alpha=0.45, linewidths=0, color=PALETTE["series_1"])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title:
        ax.set_title(title)
    _despine(ax)
    return ax


def plot_prediction_sets_1d(X_test, Y_test, Y_samples, radius, ax=None, method_name="ORCA",
                            color=None, max_points=300, seed=0, show_stats=True,
                            figsize=(5.0, 3.6)):
    """Prediction sets for a scalar response, drawn as unions of intervals.

    Each test point contributes one vertical bar per generated sample, spanning
    ``[Yhat_r - R_r, Yhat_r + R_r]``.  Where those bars separate, the prediction
    set is genuinely disconnected -- that is the multimodality ORCA is built to
    keep.

    Parameters
    ----------
    Y_samples : np.ndarray, shape (n_test, K) or (n_test, K, 1)
        Use the *same* ordering the radii were calibrated on: density-ranked
        samples for ORCA, raw samples for PCP.
    radius : float or array-like of shape (K,)
        Scalar for PCP, per-rank vector for ORCA.
    """
    ax = _new_ax(ax, figsize)
    color = color or METHOD_COLORS.get(method_name, PALETTE["series_1"])

    X = np.asarray(X_test).reshape(len(X_test), -1)[:, 0]
    Y = np.asarray(Y_test).reshape(-1)
    S = np.asarray(Y_samples, dtype=float)
    if S.ndim == 3:
        S = S[:, :, 0]
    K = S.shape[1]
    r = np.full(K, float(radius)) if np.ndim(radius) == 0 else np.asarray(radius, dtype=float)

    if len(X) > max_points:
        idx = np.random.default_rng(seed).choice(len(X), size=max_points, replace=False)
        X, Y, S = X[idx], Y[idx], S[idx]

    # One LineCollection instead of thousands of errorbar calls.
    lower, upper = S - r[None, :], S + r[None, :]
    segs = np.stack(
        [
            np.repeat(X, K),
            lower.ravel(),
            np.repeat(X, K),
            upper.ravel(),
        ],
        axis=1,
    ).reshape(-1, 2, 2)
    ax.add_collection(
        LineCollection(segs, colors=color, linewidths=1.1, alpha=0.30, capstyle="butt")
    )

    covered = np.any(np.abs(Y[:, None] - S) <= r[None, :], axis=1)
    ax.scatter(X[covered], Y[covered], s=7, linewidths=0,
               color=PALETTE["ink"], label="covered", zorder=3)
    if (~covered).any():
        ax.scatter(X[~covered], Y[~covered], s=13, linewidths=0,
                   color=PALETTE["status_critical"], marker="X",
                   label="not covered", zorder=4)

    ax.set_xlim(X.min() - 0.05 * np.ptp(X), X.max() + 0.05 * np.ptp(X))
    y_lo, y_hi = float(np.min(lower)), float(np.max(upper))
    pad = 0.05 * (y_hi - y_lo)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    title = method_name
    if show_stats:
        # Union length per point, merging the overlaps.
        from .metrics import region_size
        sizes = region_size(S, r, reference_data=Y, dimension_y=1)
        title = f"{method_name}  ·  coverage {covered.mean():.3f}  ·  mean size {sizes.mean():.3f}"
    ax.set_title(title)
    # These panels are dense to the edges, so the legend needs a surface backing
    # rather than the usual frameless treatment.
    ax.legend(loc="upper right", markerscale=1.4, frameon=True,
              facecolor=PALETTE["surface"], edgecolor="none", framealpha=0.85)
    _despine(ax)
    return ax


def plot_radius_by_rank(orca_radius, pcp_radius=None, ax=None, figsize=(5.0, 3.4),
                        title="Calibrated radius by sample rank"):
    """ORCA's per-rank radii against PCP's single radius.

    Rank 1 is the sample in the densest part of the generated cloud.  PCP is the
    flat line: one radius for every sample.  ORCA's profile is typically *spiky*
    rather than smoothly decaying -- the optimizer concentrates the budget on a
    few ranks that cover most exploration points and drives the rest toward zero,
    which is where the efficiency gain comes from.  A profile that is flat across
    ranks means the ranking bought nothing on this dataset.
    """
    ax = _new_ax(ax, figsize)
    r = np.asarray(orca_radius, dtype=float)
    ranks = np.arange(1, len(r) + 1)

    ax.step(ranks, r, where="mid", color=PALETTE["series_1"], linewidth=2.0, label="ORCA")
    ax.scatter(ranks, r, s=18, color=PALETTE["series_1"], zorder=3,
               edgecolors=PALETTE["surface"], linewidths=1.0)

    if pcp_radius is not None:
        ax.axhline(float(pcp_radius), color=PALETTE["series_2"], linewidth=2.0,
                   label="PCP (single radius)")

    ax.set_xlabel("sample rank  (1 = densest)")
    ax.set_ylabel("calibrated radius")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(title)
    ax.legend(loc="best")
    _despine(ax)
    return ax


def plot_coverage_efficiency(summary, alpha=0.1, axes=None, figsize=(6.6, 5.0),
                             method_col="method", coverage_col="coverage",
                             coverage_err_col=None, size_col="size_mean",
                             size_err_col=None, title=None, log_threshold=10.0):
    """Two stacked panels: marginal coverage, then prediction-set size.

    Both panels are dot plots rather than bars.  Coverage lives near 1 and a bar
    chart would need a truncated baseline, which exaggerates differences; a dot
    encodes position only, so the axis can start wherever the data does without
    misleading anyone.

    Methods are ordered by size, smallest (best) first.  Colour marks the method
    family -- ORCA blue, PCP orange -- and follows the method rather than its row
    in this sorted table, so re-sorting or dropping a method repaints nothing.
    The size axis switches to log only when the methods actually span more than
    ``log_threshold``-fold; forcing a log axis on a narrow range just makes the
    ticks unreadable.
    """
    import pandas as pd

    df = pd.DataFrame(summary).copy()
    for col in (method_col, coverage_col, size_col):
        if col not in df.columns:
            raise KeyError(f"summary is missing the column {col!r}")
    df = df.sort_values(size_col).reset_index(drop=True)

    if axes is None:
        apply_style()
        _, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                               gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.18})
    ax_cov, ax_eff = axes

    pos = np.arange(len(df))
    colors = [METHOD_COLORS.get(m, PALETTE["series_3"]) for m in df[method_col]]

    def _dots(ax, values, errors):
        if errors is not None:
            ax.errorbar(pos, values, yerr=errors, fmt="none",
                        ecolor=PALETTE["ink_secondary"], elinewidth=1.2, capsize=4,
                        zorder=2)
        ax.scatter(pos, values, s=70, c=colors, zorder=3,
                   edgecolors=PALETTE["surface"], linewidths=2.0)

    # -- coverage ----------------------------------------------------------- #
    cov = df[coverage_col].to_numpy(dtype=float)
    cov_err = df[coverage_err_col].to_numpy(dtype=float) \
        if coverage_err_col and coverage_err_col in df.columns else None
    _dots(ax_cov, cov, cov_err)

    # Dashed on purpose: this is the target threshold, not chart chrome.
    ax_cov.axhline(1 - alpha, color=PALETTE["ink_secondary"], linewidth=1.0,
                   linestyle=(0, (4, 3)), zorder=1)
    # Anchored left: the right edge is where the last method's error bar sits.
    ax_cov.annotate(f"nominal {1 - alpha:.2f}", xy=(0.0, 1 - alpha),
                    xycoords=("axes fraction", "data"), xytext=(4, 5),
                    textcoords="offset points", ha="left", va="bottom",
                    fontsize=8, color=PALETTE["ink_secondary"])

    spread = np.concatenate([cov, [1 - alpha]])
    if cov_err is not None:
        spread = np.concatenate([spread, cov - cov_err, cov + cov_err])
    pad = max(0.01, 0.25 * (spread.max() - spread.min()))
    ax_cov.set_ylim(spread.min() - pad, spread.max() + pad)
    ax_cov.set_ylabel("marginal coverage")
    ax_cov.grid(axis="x", visible=False)
    _despine(ax_cov)

    # -- prediction-set size ------------------------------------------------ #
    size = df[size_col].to_numpy(dtype=float)
    size_err = df[size_err_col].to_numpy(dtype=float) \
        if size_err_col and size_err_col in df.columns else None
    _dots(ax_eff, size, size_err)

    finite = size[np.isfinite(size) & (size > 0)]
    use_log = finite.size > 1 and (finite.max() / finite.min()) > log_threshold
    if use_log:
        ax_eff.set_yscale("log")
        ax_eff.set_ylabel("prediction-set size  (log, lower is better)")
    else:
        top = (size + size_err).max() if size_err is not None else size.max()
        ax_eff.set_ylim(0, top * 1.15)
        ax_eff.set_ylabel("prediction-set size  (lower is better)")

    ax_eff.set_xlim(-0.6, len(df) - 0.4)
    ax_eff.set_xticks(pos)
    ax_eff.set_xticklabels(df[method_col], rotation=18, ha="right")
    ax_eff.grid(axis="x", visible=False)
    _despine(ax_eff)

    # Direct-label the winner only; the axis carries the rest.
    best = int(np.argmin(size))
    ax_eff.annotate(f"{size[best]:.3g}", xy=(pos[best], size[best]),
                    xytext=(0, 12), textcoords="offset points", ha="center",
                    fontsize=9, color=PALETTE["ink"])

    # Colour marks the method family. The x labels already name every method, so
    # this legend explains the emphasis rather than carrying identity alone.
    families = []
    seen = set()
    for m, c in zip(df[method_col], colors):
        fam = "ORCA (ours)" if str(m).upper().startswith("ORCA") else str(m)
        if fam not in seen:
            seen.add(fam)
            families.append(Line2D([], [], marker="o", linestyle="none", markersize=7,
                                   markerfacecolor=c, markeredgecolor=PALETTE["surface"],
                                   markeredgewidth=1.5, label=fam))
    if len(families) > 1:
        ax_cov.legend(handles=families, loc="lower left", frameon=True,
                      facecolor=PALETTE["surface"], edgecolor="none", framealpha=0.85,
                      ncol=len(families), handletextpad=0.4, columnspacing=1.2)

    if title:
        ax_cov.set_title(title)
    return axes
