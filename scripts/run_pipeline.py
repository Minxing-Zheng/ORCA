#!/usr/bin/env python3
"""End-to-end ORCA experiment: data -> model -> optimize -> calibrate -> evaluate.

The generative model is trained *once*; the evaluation pool is then re-split into
exploration / calibration / test ``--n-exp`` times, so repetition variance
reflects the conformal splits rather than retraining noise.

Example
-------
    python scripts/run_pipeline.py --dataset s-shape --n 4000 --n-exp 5
    python scripts/run_pipeline.py --dataset mix-gaussian --n 4000 --solvers greedy

Writes ``runs__<name>.csv`` (one row per method per repetition) and
``summary__<name>.csv`` (mean/std per method) into ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Keep BLAS single-threaded: the solvers are the parallel part.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orca import conformal, data, metrics, models, sampling  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the ORCA pipeline on a synthetic dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data
    p.add_argument("--dataset", default="s-shape", choices=list(data.SYNTHETIC_DATASETS))
    p.add_argument("--n", type=int, default=4000, help="total observations")
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-calib", type=int, default=1000,
                   help="conformal set per repetition; ORCA splits it in half")
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--noise", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-exp", type=int, default=5, help="calibration/test repetitions")

    # method
    p.add_argument("--alpha", type=float, default=0.1, help="target miscoverage")
    p.add_argument("--n-sample", type=int, default=20, help="K, generated samples per point")
    p.add_argument("--k-neighbor", type=int, default=None,
                   help="neighbourhood for the density-proxy ranking (default ceil(K/3))")
    p.add_argument("--solvers", nargs="+", default=["milp", "greedy"],
                   choices=["milp", "lp", "greedy"])
    p.add_argument("--milp-time-limit", type=float, default=120.0)
    p.add_argument("--with-zero", action="store_true",
                   help="let the optimizer switch a rank off (radius 0)")

    # model
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=512)

    # evaluation / io
    p.add_argument("--eff-grid", type=int, default=120,
                   help="Monte-Carlo resolution per dimension for multi-D set volume")
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "results"))
    p.add_argument("--tag", default="", help="optional suffix for the output filenames")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()

    # ---------------- stage 0: data ---------------------------------------- #
    X, Y = data.generate_synthetic(args.dataset, n=args.n, noise=args.noise, seed=args.seed)
    dimension_y = Y.shape[1]
    if args.n_train + args.n_calib + args.n_test > len(X):
        raise SystemExit(
            f"n_train + n_calib + n_test = "
            f"{args.n_train + args.n_calib + args.n_test} exceeds n = {len(X)}"
        )

    idx_train, idx_pool = data.train_pool_split(len(X), args.n_train, seed=args.seed)
    print(f"[data ] {args.dataset}: X{X.shape} Y{Y.shape} | "
          f"train={len(idx_train)} pool={len(idx_pool)}")

    # ---------------- stage 1: generative model ---------------------------- #
    t0 = time.time()
    model = models.QuantileNet(
        num_features=X.shape[1],
        num_hidden=args.hidden,
        dropout=args.dropout,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.seed,
        verbose=True,
    )
    if dimension_y != 1:
        raise SystemExit(
            "QuantileNet handles a scalar response only. For a multivariate Y, "
            "plug in your own sampler -- anything with .sample(X, n_sample, rng)."
        )
    model.fit(X[idx_train], Y[idx_train].ravel())
    train_sec = time.time() - t0
    print(f"[model] trained in {train_sec:.1f}s "
          f"(final pinball loss {model.loss_history[-1]:.5f})")

    # ---------------- stage 2: sample + rank once over the pool ------------ #
    t0 = time.time()
    rng = np.random.default_rng(args.seed + 7)
    mats = sampling.build_sample_matrices(
        model, X[idx_pool], Y[idx_pool], n_sample=args.n_sample,
        rng=rng, k_neighbor=args.k_neighbor,
    )
    sample_sec = time.time() - t0
    print(f"[sample] K={args.n_sample} residual matrices in {sample_sec:.1f}s")

    pool_pos = {int(g): i for i, g in enumerate(idx_pool)}
    E_rank_pool = mats["E"]              # ORCA operates on the ranked residuals
    E_raw_pool = mats["E_unranked"]      # PCP operates on the raw ones
    Yhat_rank_pool = mats["Y_hat_ranked"]
    Yhat_raw_pool = mats["Y_hat"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (f"{args.dataset}__K{args.n_sample}__cal{args.n_calib}__test{args.n_test}"
            f"__alpha{str(args.alpha).replace('.', 'p')}__seed{args.seed}"
            f"{'__' + args.tag if args.tag else ''}__{stamp}")
    runs_csv = out_dir / f"runs__{name}.csv"
    summary_csv = out_dir / f"summary__{name}.csv"

    rows = []
    for exp in range(args.n_exp):
        t_exp = time.time()
        split_seed = args.seed + 1000 + exp
        idx_cal, idx_test = data.sample_cal_test_from_pool(
            idx_pool, args.n_calib, args.n_test, seed=split_seed
        )
        cal_pos = np.array([pool_pos[int(i)] for i in idx_cal])
        test_pos = np.array([pool_pos[int(i)] for i in idx_test])

        E_cal_rank, E_test_rank = E_rank_pool[cal_pos], E_rank_pool[test_pos]
        E_cal_raw, E_test_raw = E_raw_pool[cal_pos], E_raw_pool[test_pos]
        Yhat_test_rank = Yhat_rank_pool[test_pos]
        Yhat_test_raw = Yhat_raw_pool[test_pos]
        X_test = X[idx_test]

        # ORCA splits the conformal set: exploration half optimizes, calibration
        # half rescales. PCP keeps the whole set, which favours the baseline.
        expl_idx, cal_idx = conformal.split_exploration_calibration(
            args.n_calib, seed=split_seed + 99
        )
        E1, E2 = E_cal_rank[expl_idx], E_cal_rank[cal_idx]

        def record(method, radius, E_test, Y_samples, runtime, extra=None):
            covered = np.any(E_test <= np.atleast_2d(radius), axis=1)
            sizes = metrics.region_size(
                Y_samples, radius, reference_data=Y, dimension_y=dimension_y,
                n_samples_per_dim=args.eff_grid, seed=args.seed,
            )
            stats = metrics.summarize(covered, sizes)
            wsc, *_ = metrics.worst_slab_coverage(X_test, covered, delta=0.2, M=100,
                                                  seed=split_seed)
            row = dict(run=exp, split_seed=split_seed, method=method,
                       coverage=stats["coverage"], wsc_coverage=wsc,
                       size_mean=stats["size_mean"], size_median=stats["size_median"],
                       size_mean_covered=stats["size_mean_covered"],
                       runtime_sec=runtime, status="ok", error="")
            row.update(extra or {})
            rows.append(row)
            print(f"  [run {exp}] {method:<14} coverage={stats['coverage']:.3f} "
                  f"size={stats['size_mean']:.4f} ({runtime:.1f}s)")

        # ---- PCP baseline ------------------------------------------------- #
        t0 = time.time()
        pcp = conformal.PCP(alpha=args.alpha, dimension_y=dimension_y)
        pcp.calibrate(E_cal_raw)
        record("PCP", pcp.radius, E_test_raw, Yhat_test_raw, time.time() - t0,
               extra={"scale_q": np.nan, "proxy_obj": float(args.n_sample * pcp.radius ** dimension_y)})

        # ---- ORCA variants ------------------------------------------------ #
        for solver in args.solvers:
            label = {"milp": "ORCA-MILP", "lp": "ORCA-LP", "greedy": "ORCA-Greedy"}[solver]
            kwargs = {"with_zero": args.with_zero}
            if solver == "milp":
                kwargs["time_limit"] = args.milp_time_limit
            elif solver == "greedy":
                kwargs["with_zero"] = True  # the greedy needs a zero level to start from

            t0 = time.time()
            try:
                model_orca = conformal.ORCA(alpha=args.alpha, dimension_y=dimension_y,
                                            solver=solver, **kwargs)
                model_orca.fit_calibrate(E1, E2)
                record(label, model_orca.radius, E_test_rank, Yhat_test_rank,
                       time.time() - t0,
                       extra={"scale_q": model_orca.scale,
                              "proxy_obj": float(np.sum(model_orca.radius ** dimension_y)),
                              "solver_status": str(model_orca.solver_result["details"]
                                                   .get("status", ""))})
            except Exception as exc:  # keep the sweep alive, record the failure
                rows.append(dict(run=exp, split_seed=split_seed, method=label,
                                 coverage=np.nan, wsc_coverage=np.nan, size_mean=np.nan,
                                 size_median=np.nan, size_mean_covered=np.nan,
                                 runtime_sec=time.time() - t0, status="fail",
                                 error=repr(exc)))
                print(f"  [run {exp}] {label:<14} FAILED: {exc!r}")

        print(f"  [run {exp}] done in {time.time() - t_exp:.1f}s")

    runs = pd.DataFrame(rows)
    runs.to_csv(runs_csv, index=False)

    ok = runs[runs.status == "ok"]
    summary = (
        ok.groupby("method", as_index=False)
        .agg(n_runs=("coverage", "size"),
             coverage=("coverage", "mean"), coverage_std=("coverage", "std"),
             wsc_coverage=("wsc_coverage", "mean"),
             size_mean=("size_mean", "mean"), size_std=("size_mean", "std"),
             runtime_sec=("runtime_sec", "mean"))
        .sort_values("size_mean")
    )
    # Standard error of the mean is what the figure's bars should show.
    summary["coverage_sem"] = summary["coverage_std"] / np.sqrt(summary["n_runs"].clip(lower=1))
    summary["size_sem"] = summary["size_std"] / np.sqrt(summary["n_runs"].clip(lower=1))
    summary.to_csv(summary_csv, index=False)

    with open(out_dir / f"config__{name}.json", "w") as fh:
        json.dump({**vars(args), "train_sec": train_sec, "sample_sec": sample_sec}, fh, indent=2)

    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n[done] {time.time() - t_start:.1f}s total")
    print(f"[done] runs    -> {runs_csv}")
    print(f"[done] summary -> {summary_csv}")
    return runs, summary


if __name__ == "__main__":
    main()
