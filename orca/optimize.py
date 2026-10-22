"""Stage 3 -- optimized coverage allocation.

Given the exploration residual matrix ``E`` of shape ``(n1, K)`` -- row ``i`` is
one exploration point, column ``r`` is the rank-``r`` generated sample -- we look
for a vector of per-rank radii ``t = (t_1, ..., t_K)`` that minimises a proxy for
the size of the union of balls,

    minimise    sum_r t_r ** d
    subject to  (1/n1) * #{ i : min_r (E[i, r] - t_r) <= 0 } >= 1 - alpha,

where ``d`` is the response dimension.  Because the constraint only depends on
``t`` through the comparisons ``E[i, r] <= t_r``, an optimal ``t_r`` can always be
taken from the finite *level set* ``{E[i, r] : i}``, which turns the problem into
a mixed-integer linear program over level indices.

Three solvers are provided, all returning the same dictionary:

``solve_orca_milp``    exact MILP via OR-Tools CP-SAT (the paper's ORCA-MILP)
``solve_orca_lp``      LP relaxation + cutoff rounding  (ORCA-LP)
``solve_orca_greedy``  ratio-greedy over rank jumps + pruning (ORCA-Greedy)

None of them is required to be exact for ORCA to be valid: stage 3 only picks a
*direction* in radius space, and the split-conformal step in
:mod:`orca.conformal` rescales it.  The solvers differ in efficiency and cost.
"""

from __future__ import annotations

import heapq
import math
import time

import numpy as np
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

__all__ = [
    "build_levels",
    "build_levels_and_idx",
    "coverage_target",
    "coverage_count",
    "empirical_coverage",
    "greedy_warm_start",
    "tighten_thresholds",
    "solve_orca_milp",
    "solve_orca_lp",
    "solve_orca_greedy",
    "SOLVERS",
]


# --------------------------------------------------------------------------- #
# Level sets and feasibility bookkeeping
# --------------------------------------------------------------------------- #

def build_levels(E, with_zero=False):
    """Candidate thresholds per rank: the sorted unique residuals in that column.

    With ``with_zero=True`` a level of ``0.0`` is prepended, which lets the
    optimiser switch a rank off entirely (radius 0) rather than being forced to
    pay at least ``min_i E[i, r]``.
    """
    E = np.asarray(E, dtype=float)
    K = E.shape[1]
    if with_zero:
        return [np.unique(np.concatenate(([0.0], E[:, r]))) for r in range(K)]
    return [np.unique(E[:, r]) for r in range(K)]


def build_levels_and_idx(E, with_zero=False):
    """Levels plus the index map used by the MILP.

    ``idx_mat[i, r]`` is the smallest level index ``l`` with
    ``levels[r][l] >= E[i, r]``; point ``i`` is covered through rank ``r`` exactly
    when the chosen level index for rank ``r`` is at least ``idx_mat[i, r]``.
    """
    E = np.asarray(E, dtype=float)
    n, K = E.shape
    levels = [None] * K
    idx_mat = np.empty((n, K), dtype=np.int64)

    for r in range(K):
        col = E[:, r]
        uniq, inv = np.unique(col, return_inverse=True)
        if with_zero:
            lv = np.empty(len(uniq) + 1, dtype=float)
            lv[0] = 0.0
            lv[1:] = uniq
            levels[r] = lv
            idx_mat[:, r] = inv + 1
        else:
            levels[r] = uniq
            idx_mat[:, r] = inv

    return levels, idx_mat


def coverage_target(n, alpha):
    """Number of exploration points that must be covered.

    ``ceil((n + 1) * (1 - alpha))``, capped at ``n`` so the program stays
    feasible for small ``n`` or small ``alpha``.
    """
    return int(min(n, math.ceil((n + 1) * (1.0 - alpha))))


def coverage_count(E, thresholds):
    """How many rows of ``E`` are covered by at least one rank."""
    return int((np.asarray(E) <= np.asarray(thresholds)[None, :]).any(axis=1).sum())


def empirical_coverage(E, thresholds):
    """Fraction of rows of ``E`` covered by at least one rank."""
    return coverage_count(E, thresholds) / np.asarray(E).shape[0]


def _proxy_objective(thresholds, d):
    return float(np.sum(np.asarray(thresholds, dtype=float) ** d))


def _beta_hat(E, thresholds):
    """Per-rank empirical miscoverage ``1 - #{i : E[i,r] <= t_r} / (n + 1)``."""
    E = np.asarray(E)
    n, K = E.shape
    counts = (E <= np.asarray(thresholds)[None, :]).sum(axis=0)
    return 1.0 - counts / (n + 1)


def _result(E, thresholds, d, method, details=None):
    thresholds = np.asarray(thresholds, dtype=float)
    return {
        "thresholds": thresholds,
        "coverage": empirical_coverage(E, thresholds),
        "obj_value": _proxy_objective(thresholds, d),
        "beta_hat": _beta_hat(E, thresholds),
        "method": method,
        "details": details or {},
    }


# --------------------------------------------------------------------------- #
# Greedy warm start and threshold tightening
# --------------------------------------------------------------------------- #

def greedy_warm_start(E, alpha, d, with_zero=False):
    """Ratio-greedy construction plus a 2-exchange refinement.

    Starts from the baseline thresholds ``min_i E[i, r]`` and repeatedly promotes
    the rank whose next level buys newly covered points at the lowest cost per
    point, then tries single demote/promote swaps that keep the quota.

    Returns ``(thresholds, coverage, objective)``.
    """
    E = np.asarray(E, dtype=float)
    n, K = E.shape
    levels = build_levels(E, with_zero)
    cover = [(E[:, r][:, None] <= levels[r][None, :]) for r in range(K)]
    inc_cost = [np.diff(lv ** d, prepend=lv[0] ** d) for lv in levels]

    thresholds = np.array([lv[0] for lv in levels], dtype=float)
    covered = np.zeros(n, dtype=bool)
    for r in range(K):
        covered |= E[:, r] <= thresholds[r]

    target = coverage_target(n, alpha)

    heap, counter = [], 0
    for r in range(K):
        if len(levels[r]) > 1:
            gain = int((~covered & cover[r][:, 1]).sum())
            heapq.heappush(heap, (float(inc_cost[r][1]) / max(1, gain), counter, r, 1))
            counter += 1

    while covered.sum() < target and heap:
        ratio, _, r, ell = heapq.heappop(heap)
        gain = int((~covered & cover[r][:, ell]).sum())
        ratio_now = float(inc_cost[r][ell]) / max(1, gain)
        if abs(ratio_now - ratio) > 1e-12:
            heapq.heappush(heap, (ratio_now, counter, r, ell))
            counter += 1
            continue

        thresholds[r] = levels[r][ell]
        covered |= cover[r][:, ell]
        if ell + 1 < len(levels[r]):
            nxt = ell + 1
            g2 = int((~covered & cover[r][:, nxt]).sum())
            heapq.heappush(heap, (float(inc_cost[r][nxt]) / max(1, g2), counter, r, nxt))
            counter += 1

    # 2-exchange: demote one rank, promote another, keep it if the proxy drops.
    improved = True
    while improved:
        improved = False
        current_obj = _proxy_objective(thresholds, d)
        for r in range(K):
            lv_r = levels[r]
            idx_r = int(np.searchsorted(lv_r, thresholds[r], side="right")) - 1
            if idx_r <= 0:
                continue
            thr_r_old = thresholds[r]
            thresholds[r] = lv_r[idx_r - 1]

            for s in range(K):
                if s == r:
                    continue
                lv_s = levels[s]
                idx_s = int(np.searchsorted(lv_s, thresholds[s], side="right")) - 1
                if idx_s + 1 >= len(lv_s):
                    continue
                thr_s_old = thresholds[s]
                thresholds[s] = lv_s[idx_s + 1]

                if coverage_count(E, thresholds) >= target:
                    new_obj = _proxy_objective(thresholds, d)
                    if new_obj + 1e-12 < current_obj:
                        improved = True
                        current_obj = new_obj
                        break
                thresholds[s] = thr_s_old

            if not improved:
                thresholds[r] = thr_r_old
            else:
                break

    return thresholds, empirical_coverage(E, thresholds), _proxy_objective(thresholds, d)


def tighten_thresholds(E, alpha, thresholds, levels):
    """Demote each threshold as far as the coverage quota allows.

    Removes the discrete slack a solver leaves behind, so empirical coverage on
    the exploration set does not overshoot ``1 - alpha`` more than necessary.
    """
    E = np.asarray(E, dtype=float)
    n, K = E.shape
    target = coverage_target(n, alpha)
    thr = np.asarray(thresholds, dtype=float).copy()

    for r in range(K):
        lv = levels[r]
        while True:
            idx = int(np.searchsorted(lv, thr[r], side="right")) - 1
            if idx <= 0:
                break
            old = thr[r]
            thr[r] = lv[idx - 1]
            if coverage_count(E, thr) >= target:
                continue
            thr[r] = old
            break
    return thr


# --------------------------------------------------------------------------- #
# ORCA-MILP (CP-SAT)
# --------------------------------------------------------------------------- #

def _solve_milp_cpsat(E, alpha, d, time_limit, warm_start, with_zero,
                      levels, idx_mat, num_search_workers, relative_gap_limit,
                      cost_scale=10 ** 6):
    """Compressed CP-SAT model over per-rank level indices."""
    E = np.asarray(E, dtype=float)
    n, K = E.shape

    cost_int = [np.round((lv ** d) * cost_scale).astype(np.int64) for lv in levels]

    model = cp_model.CpModel()

    # j[r] = chosen level index for rank r; t[r] = its cost.
    j = [model.NewIntVar(0, len(levels[r]) - 1, f"j_{r}") for r in range(K)]
    t = []
    for r in range(K):
        c_r = cost_int[r]
        t_r = model.NewIntVar(int(c_r.min()), int(c_r.max()), f"t_{r}")
        model.AddElement(j[r], c_r.tolist(), t_r)
        t.append(t_r)

    # c[i][r] <=> point i is covered through rank r; y[i] <=> it is covered at all.
    c_ir = [[model.NewBoolVar(f"c_{i}_{r}") for r in range(K)] for i in range(n)]
    y = [model.NewBoolVar(f"y_{i}") for i in range(n)]

    for i in range(n):
        for r in range(K):
            idx_ir = int(idx_mat[i, r])
            if idx_ir == 0:
                model.Add(c_ir[i][r] == 1)  # covered at the baseline level
                continue
            model.Add(j[r] >= idx_ir).OnlyEnforceIf(c_ir[i][r])
            model.Add(j[r] <= idx_ir - 1).OnlyEnforceIf(c_ir[i][r].Not())
        model.Add(y[i] <= sum(c_ir[i]))

    model.Add(sum(y) >= coverage_target(n, alpha))
    model.Minimize(sum(t))

    if warm_start is not None:
        for r in range(K):
            lv = levels[r]
            hint = int(np.searchsorted(lv, warm_start[r], side="right")) - 1
            model.AddHint(j[r], int(np.clip(hint, 0, len(lv) - 1)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.relative_gap_limit = float(relative_gap_limit)
    solver.parameters.num_search_workers = int(num_search_workers)
    solver.parameters.cp_model_presolve = True
    solver.parameters.linearization_level = 0
    solver.parameters.enumerate_all_solutions = False

    t0 = time.time()
    status = solver.Solve(model)
    wrapper_time = time.time() - t0

    details = {
        "status": solver.StatusName(status),
        "wall_time": solver.WallTime(),
        "wrapper_time": wrapper_time,
        "best_bound": solver.BestObjectiveBound() / cost_scale,
        "n_levels_total": int(sum(len(lv) for lv in levels)),
    }

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # No incumbent: fall back to the warm start (or the baseline radii).
        thresholds = (
            np.asarray(warm_start, dtype=float)
            if warm_start is not None
            else np.array([levels[r][0] for r in range(K)], dtype=float)
        )
        details.update(objective_total=None, fallback=True)
        return thresholds, details

    thresholds = np.array([levels[r][solver.Value(j[r])] for r in range(K)], dtype=float)
    details.update(objective_total=solver.ObjectiveValue() / cost_scale, fallback=False)
    return thresholds, details


def solve_orca_milp(E, alpha, d=1, time_limit=180, with_zero=False, tighten=True,
                    num_search_workers=8, relative_gap_limit=0.01, verbose=False):
    """ORCA-MILP: exact coverage allocation via CP-SAT, warm-started by the greedy.

    Parameters
    ----------
    E : np.ndarray, shape (n1, K)
        Exploration residual matrix (ranked).
    alpha : float
        Target miscoverage.
    d : int
        Response dimension; the proxy objective is ``sum_r t_r ** d``.
    time_limit : float
        CP-SAT budget in seconds.  On timeout the best incumbent is returned;
        if there is none, the greedy warm start is.
    with_zero : bool
        Allow a rank to be switched off (radius 0).
    tighten : bool
        Run :func:`tighten_thresholds` on the solution.
    relative_gap_limit : float
        CP-SAT stops once the relative MIP gap is below this value.

    Returns
    -------
    dict with ``thresholds``, ``coverage``, ``obj_value``, ``beta_hat``,
    ``method`` and ``details`` (CP-SAT status, wall time, bound, ...).
    """
    E = np.asarray(E, dtype=float)
    if E.ndim != 2:
        raise ValueError(f"E must be 2-D (n1, K); got shape {E.shape}")

    t0 = time.time()
    warm, _, _ = greedy_warm_start(E, alpha, d, with_zero=with_zero)
    t_warm = time.time() - t0

    levels, idx_mat = build_levels_and_idx(E, with_zero=with_zero)
    thresholds, details = _solve_milp_cpsat(
        E, alpha, d, time_limit, warm, with_zero, levels, idx_mat,
        num_search_workers, relative_gap_limit,
    )
    details["warm_start_sec"] = t_warm

    if tighten:
        t1 = time.time()
        candidate = tighten_thresholds(E, alpha, thresholds, levels)
        details["tighten_sec"] = time.time() - t1
        if coverage_count(E, candidate) >= coverage_target(E.shape[0], alpha):
            thresholds = candidate

    if verbose:
        print(
            f"[ORCA-MILP] status={details['status']} "
            f"cpsat={details['wall_time']:.2f}s warm={t_warm:.2f}s "
            f"obj={_proxy_objective(thresholds, d):.5f}"
        )

    return _result(E, thresholds, d, "ORCA-MILP (CP-SAT)", details)


# --------------------------------------------------------------------------- #
# ORCA-LP (GLOP relaxation + cutoff rounding)
# --------------------------------------------------------------------------- #

def solve_orca_lp(E, alpha, d=1, with_zero=False, tighten=True):
    """ORCA-LP: relax the level indicators to ``[0, 1]``, then round by cutoff.

    The LP uses one variable ``z[r, l]`` per (rank, level) with monotonicity
    ``z[r, l] >= z[r, l+1]``, so ``z[r, .]`` is a fractional prefix indicator.
    Rounding picks a cutoff ``c`` and keeps the deepest level with ``z >= c``;
    we binary-search ``c`` for the largest value that still meets the quota,
    which is the cheapest feasible rounding along that family.
    """
    E = np.asarray(E, dtype=float)
    n, K = E.shape

    levels = build_levels(E, with_zero)
    inc_cost = [np.diff(lv ** d, prepend=lv[0] ** d) for lv in levels]

    # Smallest level index that covers (i, r).
    idx_mat = np.empty((n, K), dtype=np.int64)
    for r in range(K):
        idx_mat[:, r] = np.searchsorted(levels[r], E[:, r], side="left")

    baseline = (idx_mat == 0).any(axis=1).astype(np.int64)
    neighbours = [[] for _ in range(n)]
    for r in range(K):
        for i in np.where(idx_mat[:, r] > 0)[0]:
            neighbours[i].append((r, int(idx_mat[i, r])))

    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:  # pragma: no cover - OR-Tools build without GLOP
        raise RuntimeError("Could not create the GLOP linear solver.")

    z = {}
    for r in range(K):
        for ell in range(1, len(levels[r])):
            z[(r, ell)] = solver.NumVar(0.0, 1.0, f"z_{r}_{ell}")
    y = [solver.NumVar(0.0, 1.0, f"y_{i}") for i in range(n)]

    objective = solver.Objective()
    for r in range(K):
        for ell in range(1, len(levels[r])):
            objective.SetCoefficient(z[(r, ell)], float(inc_cost[r][ell]))
    objective.SetMinimization()

    for r in range(K):
        for ell in range(1, len(levels[r]) - 1):
            solver.Add(z[(r, ell)] >= z[(r, ell + 1)])

    for i in range(n):
        if neighbours[i]:
            solver.Add(sum(z[key] for key in neighbours[i]) + float(baseline[i]) >= y[i])
        else:
            solver.Add(y[i] == float(baseline[i]))

    target = coverage_target(n, alpha)
    solver.Add(sum(y) >= target)

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(f"LP relaxation failed with solver status {status}")

    z_sol = []
    for r in range(K):
        zr = np.zeros(len(levels[r]), dtype=float)
        zr[0] = 1.0  # the baseline level is always active
        for ell in range(1, len(levels[r])):
            zr[ell] = z[(r, ell)].solution_value()
        z_sol.append(zr)

    # Cutoff candidates: every level the LP actually switched on, plus 0.
    # Including the value 1.0 matters -- when the LP happens to be integral the
    # interior of (0, 1) is empty, and a grid of {0} alone would keep *every*
    # level and hand back the widest possible radii.
    active = np.concatenate([zr[1:] for zr in z_sol if len(zr) > 1]) if K else np.array([])
    active = active[active > 1e-9]
    c_grid = np.unique(np.concatenate([active, [0.0]]))

    def thresholds_at(c):
        thr = np.empty(K, dtype=float)
        for r in range(K):
            zr = z_sol[r]
            keep = np.nonzero(zr >= c - 1e-12)[0]
            thr[r] = levels[r][keep[-1] if keep.size else 0]
        return thr

    # Coverage is monotone decreasing in c, so binary search for the largest
    # feasible cutoff (i.e. the smallest feasible radii in this family).
    best_thr, best_obj = None, np.inf
    lo, hi = 0, len(c_grid) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        thr_c = thresholds_at(float(c_grid[mid]))
        if coverage_count(E, thr_c) >= target:
            obj_c = _proxy_objective(thr_c, d)
            if obj_c < best_obj - 1e-12:
                best_obj, best_thr = obj_c, thr_c
            lo = mid + 1
        else:
            hi = mid - 1

    if best_thr is None:  # no cutoff was feasible: fall back to the full levels
        best_thr = np.array([levels[r][-1] for r in range(K)], dtype=float)

    thresholds = best_thr
    if tighten:
        candidate = tighten_thresholds(E, alpha, thresholds, levels)
        if coverage_count(E, candidate) >= target:
            if _proxy_objective(candidate, d) <= _proxy_objective(thresholds, d) + 1e-12:
                thresholds = candidate

    details = {"lp_objective": solver.Objective().Value(), "n_cutoffs": int(c_grid.size)}
    return _result(E, thresholds, d, "ORCA-LP (relaxation + cutoff rounding)", details)


# --------------------------------------------------------------------------- #
# ORCA-Greedy (rank-jump set cover)
# --------------------------------------------------------------------------- #

def solve_orca_greedy(E, alpha, d=1, weights=None, with_zero=True, tighten=True):
    """ORCA-Greedy: weighted partial set cover over rank-level jumps.

    Reads the program as a set-cover instance: starting from the smallest radii,
    repeatedly take the jump ``(r, l)`` maximising newly covered exploration
    points per unit of added proxy volume, then prune radii that are no longer
    needed.  Much faster than the MILP and, in our runs, close in objective.
    """
    E = np.asarray(E, dtype=float)
    n, K = E.shape
    weights = np.ones(K, dtype=float) if weights is None else np.asarray(weights, dtype=float)

    levels = build_levels(E, with_zero)
    cover = [(E[:, r][:, None] <= levels[r][None, :]) for r in range(K)]

    idx = np.zeros(K, dtype=int)
    thr = np.array([levels[r][0] for r in range(K)], dtype=float)
    covered = np.zeros(n, dtype=bool)
    for r in range(K):
        covered |= cover[r][:, idx[r]]

    target = coverage_target(n, alpha)

    while covered.sum() < target:
        best_ratio, best_r, best_j = np.inf, None, None
        cur_cost = np.array([weights[r] * levels[r][idx[r]] ** d for r in range(K)])

        for r in range(K):
            for j in range(idx[r] + 1, len(levels[r])):
                gain = int((~covered & cover[r][:, j]).sum())
                if gain == 0:
                    continue
                ratio = float(weights[r] * levels[r][j] ** d - cur_cost[r]) / gain
                if ratio < best_ratio:
                    best_ratio, best_r, best_j = ratio, r, j

        if best_r is None:  # no jump can add coverage
            break

        idx[best_r] = best_j
        thr[best_r] = levels[best_r][best_j]
        covered |= cover[best_r][:, best_j]

    if tighten:
        changed = True
        while changed:
            changed = False
            for r in range(K):
                while idx[r] > 0:
                    cand_idx = idx.copy()
                    cand_idx[r] -= 1
                    cand_thr = np.array([levels[s][cand_idx[s]] for s in range(K)])
                    if coverage_count(E, cand_thr) >= target:
                        idx, thr = cand_idx, cand_thr
                        changed = True
                    else:
                        break

    details = {"weighted": weights is not None}
    return _result(E, thr, d, "ORCA-Greedy (rank-jump set cover)", details)


#: Solver name -> callable, used by ``conformal.ORCA(solver=...)``.
SOLVERS = {
    "milp": solve_orca_milp,
    "lp": solve_orca_lp,
    "greedy": solve_orca_greedy,
}
