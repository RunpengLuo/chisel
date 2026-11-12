#!/usr/bin/env python
"""Benchmark CHISEL against the bin-cell HMM over simulated classes and their replicates.

Author: Runpeng Luo (2026-11-12)

Dependencies:
    As compare_chisel.py: Python 3 with numpy, pandas, scikit-learn, matplotlib and cnplot;
    the chisel-py27 env for CHISEL's Caller.py and Cloner.py and for hmm_caller.py.

Usage:
    python run_benchmark.py [--outdir DIR] [--replicates 2] [--workers 4] [--jobs 4]
                            [--classes 1.2:10M 3.3:30M ...] [--py27 PATH] [--plot-only]
        --outdir     # output directory (default: bin_cell_hmm/test/out/benchmark)
        --classes    # <shared>.<private>:<length>M (default: simulation.SIM_CLASSES)

Inputs:
    None; every sample is drawn by simulation.class_config.
Parameters:
    replicates: replicates per class, sample ids ..._random0 to ..._random<n-1>.
    workers: samples run at once; workers * jobs should not exceed the CPU count.
    jobs: parallel jobs of Caller.py and hmm_caller.py within one sample.
    K, stay, maxcn: HMM states (default: number of distinct true states), self-transition
        probability, grid maximum copy number.
    plot_only: redraw from the per-sample metrics already written.
Outputs:
    <sample_id>/: compare_chisel.run_sample's outputs of the sample, and its metrics.tsv.
    benchmark_metrics.tsv: every sample's rows, sample_id first, plus bp_ratio, the method's
        breakpoints over the truth's.
    benchmark_medians.tsv: each plotted method's median per class.
    benchmark_metrics.pdf, benchmark_metrics.png: plot_simulation.plot_simulation_metrics of
        the methods of PLOTTED.
Notes:
    A sample whose metrics.tsv exists is not run again, so an interrupted benchmark resumes.
    Each sample is simulated by class_config, seeded by its sample id; the methods take the
    replicate index as their seed.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path[:0] = [str(PKG), str(HERE)]

from compare_chisel import (  # noqa: E402
    DEFAULT_PY27,
    log,
    run_sample,
    score_sample,
    write_metrics,
)
from plot_simulation import get_class_medians, plot_simulation_metrics  # noqa: E402
from simulation import SIM_CLASSES, class_config, sample_id, simulate  # noqa: E402

PLOTTED = (
    "chisel_caller_bic",
    "chisel_cloner_bic",
    "hmm_grid",
    "hmm_chisel_caller_bic",
    "hmm_chisel_cloner_bic",
)
"""The methods the metric page compares, in drawing order."""


def parse_class(text):
    """Parse ``<shared>.<private>:<length>M`` into a class tuple.

    Args:
        text: e.g. ``1.2:10M``.

    Returns:
        (shared, private, length in bp).
    """
    counts, size = text.split(":")
    shared, private = (int(x) for x in counts.split("."))
    return shared, private, int(float(size.rstrip("Mm")) * 1_000_000)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outdir", type=Path, default=HERE / "out" / "benchmark")
    p.add_argument("--replicates", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--classes", nargs="+", type=parse_class, default=list(SIM_CLASSES))
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--stay", type=float, default=0.99)
    p.add_argument("--maxcn", type=int, default=6)
    p.add_argument("--py27", type=Path, default=DEFAULT_PY27)
    p.add_argument("--plot-only", action="store_true")
    return p.parse_args()


def run_one(task):
    """Simulate, run and score one sample unless its metrics.tsv exists.

    Args:
        task: (class tuple, replicate, outdir, py27, jobs, K, stay, maxcn).

    Returns:
        (sample id, "done" or "skipped").
    """
    (shared, private, size), rep, outdir, py27, jobs, k, stay, maxcn = task
    sid = sample_id(shared, private, size, rep)
    metrics = outdir / sid / "metrics.tsv"
    if metrics.exists():
        return sid, "skipped"
    sim = simulate(class_config(shared, private, size, rep))
    res, _ = run_sample(sim, outdir / sid, str(py27), jobs, rep, k, stay, maxcn)
    write_metrics(score_sample(sim, res), metrics)
    return sid, "done"


def collect(outdir, classes, replicates):
    """Concatenate the samples' metrics and add bp_ratio, breakpoints over the truth's.

    Args:
        outdir: Benchmark directory.
        classes: Class tuples.
        replicates: Replicates per class.

    Returns:
        DataFrame, one row per sample and method, ``sample_id`` first.
    """
    tables = []
    for shared, private, size in classes:
        for rep in range(replicates):
            sid = sample_id(shared, private, size, rep)
            path = outdir / sid / "metrics.tsv"
            if path.exists():
                table = pd.read_csv(path, sep="\t")
                truth = float(table.loc[table["method"] == "truth", "bp_cn"].iloc[0])
                table["bp_ratio"] = table["bp_cn"] / truth if truth else float("nan")
                tables.append(table.assign(sample_id=sid))
    table = pd.concat(tables, ignore_index=True)
    return table[["sample_id", *[c for c in table if c != "sample_id"]]]


def main():
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        tasks = [
            (c, rep, outdir, args.py27, args.jobs, args.K, args.stay, args.maxcn)
            for c in args.classes
            for rep in range(args.replicates)
        ]
        log(f"{len(tasks)} samples, {args.workers} at once, {args.jobs} jobs each")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, t) for t in tasks]
            for n, future in enumerate(as_completed(futures), 1):
                sid, status = future.result()
                log(f"[{n}/{len(tasks)}] {sid}: {status}")

    table = collect(outdir, args.classes, args.replicates)
    table.to_csv(outdir / "benchmark_metrics.tsv", sep="\t", index=False)
    get_class_medians(table, list(PLOTTED)).to_csv(
        outdir / "benchmark_medians.tsv", sep="\t", index=False, float_format="%.4f"
    )
    figure = plot_simulation_metrics(table, list(PLOTTED))
    for ext in ("pdf", "png"):
        figure.savefig(
            outdir / f"benchmark_metrics.{ext}", dpi=300, bbox_inches="tight"
        )
    log(f"{len(table['sample_id'].unique())} samples; outputs in {outdir}")
    for name in (
        "benchmark_metrics.tsv",
        "benchmark_medians.tsv",
        "benchmark_metrics.pdf",
    ):
        print(os.path.join(outdir, name))


if __name__ == "__main__":
    main()
