#!/usr/bin/env python3
"""Build the comparison figure from a ``summary__*.csv`` written by run_pipeline.py.

Example
-------
    python scripts/make_figures.py --summary results/summary__s-shape__*.csv
    python scripts/make_figures.py            # picks the newest summary in results/
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write files without needing a display
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orca import plotting  # noqa: E402


def _readable_title(stem):
    """Turn ``summary__s-shape__K20__cal1000__...`` into a human title."""
    parts = stem.replace("summary__", "").split("__")
    dataset = parts[0] if parts else stem
    bits = {p[0]: p for p in parts[1:] if p}
    keep = [p for p in parts[1:] if p.startswith(("K", "cal", "test", "alpha"))]
    detail = ", ".join(keep[:4]).replace("alpha0p", "alpha 0.")
    return f"{dataset}" + (f"  ({detail})" if detail else "")


def parse_args(argv=None):
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Render ORCA comparison figures.")
    p.add_argument("--summary", default=None,
                   help="path to a summary CSV (default: newest in results/)")
    p.add_argument("--out-dir", default=str(root / "figures"))
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--title", default=None)
    p.add_argument("--formats", nargs="+", default=["png", "pdf"])
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]

    if args.summary:
        matches = sorted(glob.glob(args.summary))
        if not matches:
            raise SystemExit(f"no file matches {args.summary!r}")
        summary_path = Path(matches[-1])
    else:
        matches = sorted((root / "results").glob("summary__*.csv"))
        if not matches:
            raise SystemExit("no summary CSV in results/ -- run scripts/run_pipeline.py first")
        summary_path = matches[-1]

    df = pd.read_csv(summary_path)
    print(f"[read] {summary_path.name}  ({len(df)} methods)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plotting.apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.25]})
    plotting.plot_coverage_efficiency(
        df, alpha=args.alpha, axes=axes,
        coverage_err_col="coverage_sem" if "coverage_sem" in df.columns else None,
        size_err_col="size_sem" if "size_sem" in df.columns else None,
        title=args.title or _readable_title(summary_path.stem),
    )
    fig.tight_layout()

    stem = summary_path.stem.replace("summary__", "coverage_efficiency__")
    for ext in args.formats:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path)
        print(f"[write] {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
