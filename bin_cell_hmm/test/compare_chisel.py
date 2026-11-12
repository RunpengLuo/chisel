#!/usr/bin/env python
"""Compare CHISEL's global clustering and calling with the bin-cell HMM on simulated data.

Author: Runpeng Luo (2026-11-11)

Dependencies:
    Python 3 with numpy, scikit-learn, matplotlib (bin_cell_hmm/hmm.py, simulation.py), and
    cnplot >= 0.1.1b3 (https://github.com/raphael-group/cnplot) for ab_heatmaps.pdf.
    Python 2.7 env with CHISEL's requirements, for src/chisel/Caller.py, src/chisel/Cloner.py
    and bin_cell_hmm/hmm_caller.py.

Usage:
    python compare_chisel.py [--outdir DIR] [--seed 0] [--n-cells 120] [--n-chroms 4]
                             [--bins-per-chrom 50] [--cell-event-rate 0.5]
                             [--n-wgd-clones 0] [--K K] [--stay 0.99] [--maxcn 6]
                             [--jobs 4] [--py27 PATH] [--cn-palette cnamaste]
        --outdir   # output directory (default: bin_cell_hmm/test/out/sim_seed<seed>[_wgd<n>])

Inputs:
    None; data are drawn from simulation.SimConfig with the CLI overrides.
Parameters:
    seed: simulation and method seed.
    n_cells, n_chroms, bins_per_chrom, cell_event_rate, n_wgd_clones: simulation overrides.
    K: HMM states (default: number of distinct true states of each HMM variant).
    stay, maxcn: HMM self-transition probability and grid maximum copy number.
    jobs: parallel jobs of CHISEL Caller.
    py27: python2.7 executable of the CHISEL env (default: $CHISEL_PY27, else
        ~/.local/miniconda3/envs/chisel-py27/bin/python).
    cn_palette: cnplot (a, b) palette of ab_heatmaps.pdf, "cnamaste" or "hatchet".
Outputs:
    combo.tsv, truth.tsv: simulated CHISEL input and true states.
    The suffix _bic marks output whose base ploidy per cell is the one Caller.calling's BIC
    selected (-P 4: ploidy 2 or 4).
    chisel/calls_bic.tsv, chisel/caller.log: CHISEL Caller output (CLUSTER, CN_STATE), log.
    chisel/calls_cloned_bic.tsv, chisel/mapping_bic.tsv, chisel/cloner.log: CHISEL Cloner.
    hmm/calls_bic.tsv, hmm/hmm.npz, hmm/caller.log: hmm_caller.py (hmmclustering on CHISEL's
        data matrix with the phased BAF, then CHISEL's Caller.calling per cell), run in py2.7.
    hmm/ploidy_bic.tsv: CELL, CLONE (true), PLOIDY, the BIC-selected base ploidy per cell.
    hmm/calls_cloned_bic.tsv, hmm/mapping_bic.tsv, hmm/cloner.log: CHISEL Cloner on
        hmm/calls_bic.tsv.
    Rows of metrics.tsv: chisel_caller_bic, chisel_cloner_bic; hmm_grid (HMM states labeled
        by the nearest haplotype-specific grid point at ploidy base 2); hmm_chisel_caller_bic
        and hmm_chisel_cloner_bic (HMM states labeled per cell by Caller.calling, then
        Cloner).
    metrics.tsv: per method (and the truth as reference), sensitivity, specificity,
        N-CNA recall, CNA recall, cell-event recall, clone ARI and breakpoint counts
        (bin_cell_hmm/metrics.py).
    cn_heatmaps.png, error_maps.png, rdr_baf_states.png: figures.
    ab_heatmaps.pdf: (a, b) state per cell and bin, truth and every method, drawn with cnplot
        in the --cn-palette palette (plots.py).
    rdr_baf_scatter.pdf: cnplot (BAF, RDR) scatters, one dot per (cell, bin), a page per
        coloring (truth, then every method), each page labelling only its own states.
Notes:
    CHISEL runs unmodified as `python2.7 src/chisel/Caller.py combo.tsv -j J --seed S`, the
    command chisel_calling issues. Cloner rewrites its input file, so it runs on a copy.
    ARI: Hubert and Arabie (1985), https://doi.org/10.1007/BF01908075
"""

import argparse
from collections import Counter
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import adjusted_rand_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parent
sys.path.insert(0, str(PKG))

from hmm import cn_grid  # noqa: E402
from metrics import breakpoints, phase_free, supervised_metrics  # noqa: E402
from cnplot import CN_PALETTES  # noqa: E402
from plots import bins_table, plot_rdr_baf_scatters, plot_state_heatmaps  # noqa: E402
from simulation import SimConfig, simulate, write_combo, write_truth  # noqa: E402

DEFAULT_PY27 = Path(
    os.environ.get(
        "CHISEL_PY27",
        Path.home()
        / ".local"
        / "miniconda3"
        / "envs"
        / "chisel-py27"
        / "bin"
        / "python",
    )
)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID_INK = "#c3c2b7"
SLOT_1 = "#2a78d6"
SLOT_2 = "#eb6834"
CN_MAX = 6


def log(msg):
    print(f"[compare] {msg}", file=sys.stderr, flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-cells", type=int, default=SimConfig.n_cells)
    p.add_argument("--n-chroms", type=int, default=SimConfig.n_chroms)
    p.add_argument("--bins-per-chrom", type=int, default=SimConfig.bins_per_chrom)
    p.add_argument("--cell-event-rate", type=float, default=SimConfig.cell_event_rate)
    p.add_argument("--n-wgd-clones", type=int, default=SimConfig.n_wgd_clones)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--stay", type=float, default=0.99)
    p.add_argument("--maxcn", type=int, default=6)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--py27", type=Path, default=DEFAULT_PY27)
    p.add_argument("--cn-palette", choices=CN_PALETTES, default="cnamaste")
    return p.parse_args()


def run(cmd, log_path, stdout_path=None):
    """Run a command, stdout to ``stdout_path`` (or discarded) and stderr to ``log_path``.

    ``PYTHONDONTWRITEBYTECODE`` is set: Python 2 validates ``.pyc`` by source mtime at
    1-second resolution, so a ``.pyc`` written next to a just-edited module can go stale.
    """
    cmd = [str(c) for c in cmd]
    log("run: " + " ".join(cmd))
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    start = time.time()
    with open(log_path, "w") as err, open(stdout_path or os.devnull, "w") as out:
        subprocess.run(cmd, stdout=out, stderr=err, check=True, env=env)
    return time.time() - start


def read_calls(path, col, sim):
    """Read a CHISEL calls file into (n_cells, n_bins) a, b and the per-bin CLUSTER.

    Args:
        path: calls.tsv from Caller (state in column 11) or Cloner (corrected state in 12).
        col: 0-based column of the "a|b" state.
        sim: Simulated data, which fixes the bin and cell order.

    Returns:
        a, b: (n_cells, n_bins) int arrays.
        cluster: (n_bins,) int array.
    """
    bin_idx = {(c, int(s)): t for t, (c, s) in enumerate(zip(sim.chroms, sim.starts))}
    cell_idx = {c: e for e, c in enumerate(sim.cells)}
    a = np.full(sim.a.shape, -1)
    b = np.full(sim.a.shape, -1)
    cluster = np.full(len(sim.chroms), -1)
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            t, e = bin_idx[(p[0], int(p[1]))], cell_idx[p[3]]
            a[e, t], b[e, t] = (int(x) for x in p[col].split("|"))
            cluster[t] = int(p[10])
    assert (a >= 0).all() and (cluster >= 0).all(), f"{path} misses some (bin, cell)"
    return a, b, cluster


def allele(a, b):
    return np.maximum(a, b), np.minimum(a, b)


def evaluate(name, a_hat, b_hat, labels, sim, seconds, clone_ari=float("nan")):
    """Score one method against the true states with ``metrics.py``.

    Args:
        name: Method name.
        a_hat, b_hat: (n_cells, n_bins) called copies.
        labels: (n_cells, n_bins) the method's own labels: CHISEL's bin cluster broadcast
            to every cell, or the HMM state.
        sim: Simulated data.
        seconds: Wall time of the method.
        clone_ari: ARI of the method's cell clones against the true clones.

    Returns:
        dict with sensitivity, specificity, exact_altered (N-CNA recall), state_altered
        (CNA recall), exact_cell_event (exact phase-free pair on cell-specific events),
        clone_ari, boundaries (adjacent same-chromosome (bin, cell) pairs), bp_cn and
        bp_label (pairs whose phase-free call or own label changes) with their rates
        over boundaries, n_labels and seconds.
    """
    call = phase_free(a_hat, b_hat)
    true = phase_free(sim.a, sim.b)
    ev = sim.cell_event
    bp_cn, n_bound = breakpoints(call, sim.chroms)
    bp_label, _ = breakpoints(labels, sim.chroms)
    row = {"method": name, **supervised_metrics(sim.a, sim.b, a_hat, b_hat)}
    row.update(
        exact_cell_event=float(np.mean(call[ev] == true[ev]))
        if ev.any()
        else float("nan"),
        clone_ari=clone_ari,
        boundaries=n_bound,
        bp_cn=bp_cn,
        bp_cn_rate=bp_cn / n_bound,
        bp_label=bp_label,
        bp_label_rate=bp_label / n_bound,
        n_labels=len(np.unique(labels)),
        seconds=seconds,
    )
    return row


def read_clone_ari(mapping, sim):
    """ARI of CHISEL Cloner's clones against the true ones, over cells with a clone.

    Args:
        mapping: Cloner's mapping.tsv (CELL CLUSTER CLONE), CLONE ``None`` for noisy cells.
        sim: Simulated data.

    Returns:
        ARI over the cells Cloner assigns to a clone.
    """
    clone = dict(
        line.split("\t")[0::2]
        for line in open(mapping).read().splitlines()
        if not line.startswith("#")
    )
    kept = [e for e, c in enumerate(sim.cells) if clone[c] != "None"]
    return adjusted_rand_score(sim.clone[kept], [clone[sim.cells[e]] for e in kept])


def write_metrics(rows, path):
    keys = list(rows[0])
    with open(path, "w") as out:
        out.write("\t".join(keys) + "\n")
        for r in rows:
            out.write(
                "\t".join(
                    f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k])
                    for k in keys
                )
                + "\n"
            )


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_INK)
    ax.tick_params(colors=INK_2, labelsize=8)


def _chrom_lines(ax, sim, n_rows):
    edges = [t for t in range(1, len(sim.chroms)) if sim.chroms[t] != sim.chroms[t - 1]]
    ax.vlines(np.array(edges) - 0.5, -0.5, n_rows - 0.5, color=INK, linewidth=0.6)
    mids = [np.mean(np.flatnonzero(sim.chroms == c)) for c in dict.fromkeys(sim.chroms)]
    ax.set_xticks(mids, list(dict.fromkeys(sim.chroms)))


def plot_heatmaps(panels, sim, order, path):
    """Total copy number per (cell, bin), cells grouped by clone."""
    ramp = LinearSegmentedColormap.from_list(
        "cn",
        [
            (0.0, "#184f95"),
            (2 / CN_MAX, "#f0efec"),
            (3 / CN_MAX, "#f4a3a0"),
            (1.0, "#7a1f1f"),
        ],
    )
    cmap = ListedColormap([ramp(v / CN_MAX) for v in range(CN_MAX + 1)])
    norm = BoundaryNorm(np.arange(-0.5, CN_MAX + 1.5), cmap.N)
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(10, 2.3 * len(panels)),
        sharex=True,
        facecolor=SURFACE,
        layout="constrained",
    )
    for ax, (title, total) in zip(axes, panels):
        im = ax.imshow(
            np.minimum(total[order], CN_MAX),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )
        _style(ax)
        _chrom_lines(ax, sim, len(order))
        ax.set_ylabel("cells", color=INK_2, fontsize=8)
        ax.set_title(title, loc="left", color=INK, fontsize=9)
    cbar = fig.colorbar(im, ax=axes, ticks=range(CN_MAX + 1), shrink=0.5, aspect=30)
    cbar.ax.set_yticklabels([str(v) for v in range(CN_MAX)] + [f"{CN_MAX}+"])
    cbar.set_label("total copy number", color=INK_2, fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_errors(panels, sim, order, path):
    """Allele-specific mismatches per (cell, bin); first panel marks cell-specific events."""
    cmap = ListedColormap([SURFACE, INK_2])
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(10, 2.0 * len(panels)), sharex=True, facecolor=SURFACE
    )
    for ax, (title, mask) in zip(axes, panels):
        ax.imshow(
            mask[order].astype(int),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
        )
        _style(ax)
        _chrom_lines(ax, sim, len(order))
        ax.set_ylabel("cells", color=INK_2, fontsize=8)
        ax.set_title(title, loc="left", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_states(points, grid_states, grid_means, init_means, means, path, seed):
    """(RDR, BAF) points with the grid, the initial and the fitted HMM state means."""
    rng = np.random.default_rng(seed)
    points = points[np.isfinite(points).all(axis=1)]
    pts = points[rng.choice(len(points), min(len(points), 6000), replace=False)]
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    _style(ax)
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=4,
        color=GRID_INK,
        alpha=0.5,
        linewidths=0,
        label="(bin, cell) points",
    )
    ax.scatter(
        grid_means[:, 0],
        grid_means[:, 1],
        s=30,
        facecolors="none",
        edgecolors=INK_2,
        linewidths=0.8,
    )
    for (a, b), (x, y) in zip(grid_states, grid_means):
        ax.annotate(
            f"{a},{b}",
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color=INK_2,
        )
    for (x0, y0), (x1, y1) in zip(init_means, means):
        ax.plot([x0, x1], [y0, y1], color=INK_2, linewidth=0.8)
    ax.scatter(
        init_means[:, 0],
        init_means[:, 1],
        s=64,
        color=SLOT_1,
        edgecolors=SURFACE,
        linewidths=2,
        label="init (grid)",
    )
    ax.scatter(
        means[:, 0],
        means[:, 1],
        s=64,
        color=SLOT_2,
        edgecolors=SURFACE,
        linewidths=2,
        label="fitted mean",
    )
    ax.set_xlabel("RDR", color=INK_2, fontsize=9)
    ax.set_ylabel("BAF", color=INK_2, fontsize=9)
    ax.set_xlim(0, max(3.2, float(np.quantile(pts[:, 0], 0.999))))
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


METHODS = (
    "chisel_caller_bic",
    "chisel_cloner_bic",
    "hmm_grid",
    "hmm_chisel_caller_bic",
    "hmm_chisel_cloner_bic",
)
"""Rows of metrics.tsv after the truth, in order (docs/benchmark.md, methods compared)."""


def run_sample(sim, outdir, py27, jobs=4, seed=0, k=None, stay=0.99, maxcn=6):
    """Run CHISEL and the bin-cell HMM on one simulated dataset.

    Writes combo.tsv and truth.tsv; runs CHISEL Caller.py then Cloner.py, and hmm_caller.py
    then Cloner.py on its calls, all in Python 2.7.

    Args:
        sim: Simulated data.
        outdir: Output directory, created if missing.
        py27: python2.7 executable of the CHISEL env.
        jobs: Parallel jobs of Caller.py and hmm_caller.py.
        seed: Seed of every method.
        k: HMM states, or None for the number of distinct true haplotype-specific states.
        stay: HMM self-transition probability.
        maxcn: HMM grid maximum copy number.

    Returns:
        (res, hm): ``res`` maps each method of ``METHODS`` it ran to a dict with ``a``,
        ``b`` (n_cells, n_bins) calls, ``labels`` (n_cells, n_bins), ``seconds`` and, for
        the Cloner rows, ``clone_ari``; ``hm`` is hmm_caller.py's npz as a dict.
    """
    outdir = Path(outdir)
    (outdir / "chisel").mkdir(parents=True, exist_ok=True)
    (outdir / "hmm").mkdir(exist_ok=True)
    combo = outdir / "combo.tsv"
    write_combo(sim, combo)
    write_truth(sim, outdir / "truth.tsv")
    k = k or len(np.unique(sim.a * 100 + sim.b))
    log(f"{outdir.name}: K {k}")
    src = REPO / "src" / "chisel"

    calls = outdir / "chisel" / "calls_bic.tsv"
    t_caller = run(
        [py27, src / "Caller.py", combo, "-j", jobs, "--seed", seed],
        outdir / "chisel" / "caller.log",
        calls,
    )
    cloned = outdir / "chisel" / "calls_cloned_bic.tsv"
    shutil.copy(calls, cloned)
    t_cloner = run(
        [py27, src / "Cloner.py", cloned, "--seed", seed],
        outdir / "chisel" / "cloner.log",
        outdir / "chisel" / "mapping_bic.tsv",
    )

    npz = outdir / "hmm" / "hmm.npz"
    hmm_calls = outdir / "hmm" / "calls_bic.tsv"
    t_hmm = run(
        [py27, PKG / "hmm_caller.py", combo, "--hmm-states", k]
        + ["--hmm-stay", stay, "--hmm-maxcn", maxcn, "--hmm-npz", npz]
        + ["-j", jobs, "--seed", seed],
        outdir / "hmm" / "caller.log",
        hmm_calls,
    )
    hmm_cloned = outdir / "hmm" / "calls_cloned_bic.tsv"
    shutil.copy(hmm_calls, hmm_cloned)
    t_hmm_cloner = run(
        [py27, src / "Cloner.py", hmm_cloned, "--seed", seed],
        outdir / "hmm" / "cloner.log",
        outdir / "hmm" / "mapping_bic.tsv",
    )
    ploidy = np.load(npz)["ploidy"]
    with open(outdir / "hmm" / "ploidy_bic.tsv", "w") as out:
        out.write("CELL\tCLONE\tPLOIDY\n")
        for cell, clone, p in zip(sim.cells, sim.clone, ploidy):
            out.write(f"{cell}\t{clone}\t{int(p)}\n")

    seconds = {
        "chisel_caller_bic": t_caller,
        "chisel_cloner_bic": t_caller + t_cloner,
        "hmm_grid": t_hmm,
        "hmm_chisel_caller_bic": t_hmm,
        "hmm_chisel_cloner_bic": t_hmm + t_hmm_cloner,
    }
    res, hm = load_results(sim, outdir)
    for name, value in seconds.items():
        res[name]["seconds"] = value
    return res, hm


def load_results(sim, outdir):
    """Read the calls ``run_sample`` wrote for one sample, without running anything.

    Args:
        sim: The simulated data the sample was run on.
        outdir: The sample's output directory.

    Returns:
        (res, hm) as ``run_sample`` returns them, less the ``seconds`` of each method.
    """
    outdir = Path(outdir)
    hm = dict(np.load(outdir / "hmm" / "hmm.npz"))
    assert (hm["chroms"].astype(str) == sim.chroms).all()
    assert (hm["starts"] == sim.starts).all()
    assert (hm["cells"].astype(str) == sim.cells).all()
    members = hm["members"].T
    cn_m = hm["state_cn"][members]
    a_c, b_c, clus = read_calls(outdir / "chisel" / "calls_bic.tsv", 11, sim)
    a_k, b_k, _ = read_calls(outdir / "chisel" / "calls_cloned_bic.tsv", 12, sim)
    a_h, b_h, _ = read_calls(outdir / "hmm" / "calls_bic.tsv", 11, sim)
    a_hk, b_hk, _ = read_calls(outdir / "hmm" / "calls_cloned_bic.tsv", 12, sim)
    clus_cells = np.broadcast_to(clus, sim.a.shape)
    res = {
        "chisel_caller_bic": dict(a=a_c, b=b_c, labels=clus_cells),
        "chisel_cloner_bic": dict(
            a=a_k,
            b=b_k,
            labels=clus_cells,
            clone_ari=read_clone_ari(outdir / "chisel" / "mapping_bic.tsv", sim),
        ),
        "hmm_grid": dict(a=cn_m[..., 0], b=cn_m[..., 1], labels=members),
        "hmm_chisel_caller_bic": dict(a=a_h, b=b_h, labels=members),
        "hmm_chisel_cloner_bic": dict(
            a=a_hk,
            b=b_hk,
            labels=members,
            clone_ari=read_clone_ari(outdir / "hmm" / "mapping_bic.tsv", sim),
        ),
    }
    return res, hm


def score_sample(sim, res):
    """Score the truth (reference row) and every method of ``res`` in ``METHODS`` order.

    Args:
        sim: Simulated data.
        res: ``run_sample``'s methods.

    Returns:
        List of ``evaluate`` rows.
    """
    rows = [
        evaluate("truth", sim.a, sim.b, phase_free(sim.a, sim.b), sim, float("nan"))
    ]
    for name in METHODS:
        if name in res:
            r = res[name]
            rows.append(
                evaluate(
                    name,
                    r["a"],
                    r["b"],
                    r["labels"],
                    sim,
                    r["seconds"],
                    r.get("clone_ari", float("nan")),
                )
            )
    return rows


SHOWN = (
    ("CHISEL Caller (CN_STATE)", "chisel_caller_bic"),
    ("CHISEL Cloner (CORRECTED_HAP_CN)", "chisel_cloner_bic"),
    ("bin-cell HMM, grid labels", "hmm_grid"),
    ("bin-cell HMM + CHISEL Caller.calling", "hmm_chisel_caller_bic"),
    ("bin-cell HMM + CHISEL Caller.calling + Cloner", "hmm_chisel_cloner_bic"),
)
"""The methods each sample's figures draw after the truth, as (title, method)."""


def plot_ab_heatmaps(sim, res, path, cn_palette="cnamaste"):
    """Draw the (a, b) state of every cell and bin, the truth and each method of ``SHOWN``,
    cells grouped by true clone, with cnplot (``plots.plot_state_heatmaps``).

    Args:
        sim: Simulated data.
        res: ``run_sample``'s or ``load_results``' methods.
        path: Output image path.
        cn_palette: cnplot (a, b) palette.
    """
    rows = np.lexsort((sim.cells, sim.clone))[::-1]
    plot_state_heatmaps(
        [("truth", sim.a[rows], sim.b[rows])]
        + [(t, res[m]["a"][rows], res[m]["b"][rows]) for t, m in SHOWN],
        bins_table(sim.chroms, sim.starts, sim.ends),
        np.where(
            sim.clone[rows] == 0, "normal", [f"clone{c}" for c in sim.clone[rows]]
        ),
        path,
        cn_palette=cn_palette,
    )


def plot_rdr_baf(sim, res, path, cn_palette="cnamaste"):
    """Write the (BAF, RDR) scatters of one sample with cnplot, one dot per (cell, bin): a page
    colored by the truth, then one per method of ``SHOWN``, each page labelling only its own
    states (``plots.plot_rdr_baf_scatters``).

    Args:
        sim: Simulated data.
        res: ``run_sample``'s or ``load_results``' methods.
        path: Output PDF path.
        cn_palette: cnplot (a, b) palette.
    """
    baf = np.where(sim.a_count + sim.b_count > 0, sim.baf, np.nan)
    plot_rdr_baf_scatters(
        sim.rdr,
        baf,
        [("truth", sim.a, sim.b)] + [(t, res[m]["a"], res[m]["b"]) for t, m in SHOWN],
        path,
        cn_palette,
    )


def plot_sample(sim, res, hm, outdir, maxcn=6, seed=0, cn_palette="cnamaste"):
    """Write one sample's figures: cn_heatmaps.png, error_maps.png, ab_heatmaps.pdf,
    rdr_baf_scatter.pdf and rdr_baf_states.png.

    Args:
        sim: Simulated data.
        res: ``run_sample``'s methods.
        hm: ``run_sample``'s hmm_caller.py npz.
        outdir: Output directory.
        maxcn: HMM grid maximum copy number.
        seed: Seed of the scatter's subsample.
        cn_palette: cnplot (a, b) palette of ab_heatmaps.pdf.
    """
    outdir = Path(outdir)
    tmaj, tmin = allele(sim.a, sim.b)
    shown = SHOWN
    order = np.lexsort((sim.cells, sim.clone))
    plot_heatmaps(
        [("truth", sim.a + sim.b)] + [(t, res[m]["a"] + res[m]["b"]) for t, m in shown],
        sim,
        order,
        outdir / "cn_heatmaps.png",
    )

    def wrong(m):
        pmaj, pmin = allele(res[m]["a"], res[m]["b"])
        return (pmaj != tmaj) | (pmin != tmin)

    plot_errors(
        [("truth: cell-specific events (cell differs from its clone)", sim.cell_event)]
        + [(f"{t}: allele-specific mismatches", wrong(m)) for t, m in shown],
        sim,
        order,
        outdir / "error_maps.png",
    )
    plot_ab_heatmaps(sim, res, outdir / "ab_heatmaps.pdf", cn_palette)
    plot_rdr_baf(sim, res, outdir / "rdr_baf_scatter.pdf", cn_palette)
    data = hm["data"]
    points = np.column_stack([data[:, 0::2].ravel(), data[:, 1::2].ravel()])
    grid_states, grid_means = cn_grid(maxcn, 2.0)
    init_idx = [
        int(np.flatnonzero((grid_states == s).all(1))[0]) for s in hm["init_states"]
    ]
    plot_states(
        points,
        grid_states,
        grid_means,
        grid_means[init_idx],
        hm["means"],
        outdir / "rdr_baf_states.png",
        seed,
    )


def main():
    args = parse_args()
    name = f"sim_seed{args.seed}" + (
        f"_wgd{args.n_wgd_clones}" if args.n_wgd_clones else ""
    )
    outdir = (args.outdir or HERE / "out" / name).resolve()
    py27 = str(args.py27)
    if shutil.which(py27) is None:
        raise SystemExit(f"python2.7 for CHISEL not found: {py27}")

    cfg = SimConfig(
        n_cells=args.n_cells,
        n_chroms=args.n_chroms,
        bins_per_chrom=args.bins_per_chrom,
        cell_event_rate=args.cell_event_rate,
        n_wgd_clones=args.n_wgd_clones,
        seed=args.seed,
    )
    log(f"simulate: {cfg}")
    sim = simulate(cfg)
    for c in np.unique(sim.clone):
        sel = sim.clone == c
        log(
            f"clone {c}: {sel.sum()} cells, mean total CN {np.mean(sim.a[sel] + sim.b[sel]):.3f}"
        )
    log(f"cell-specific (bin, cell) entries: {int(sim.cell_event.sum())}")

    res, hm = run_sample(
        sim, outdir, py27, args.jobs, args.seed, args.K, args.stay, args.maxcn
    )
    ploidy = dict(zip(hm["cells"].astype(str), hm["ploidy"]))
    for c in np.unique(sim.clone):
        called = Counter(int(ploidy[e]) for e in sim.cells[sim.clone == c])
        log(f"clone {c}: HMM + Caller.calling base ploidy counts {dict(called)}")

    write_metrics(score_sample(sim, res), outdir / "metrics.tsv")
    print((outdir / "metrics.tsv").read_text(), end="")
    plot_sample(sim, res, hm, outdir, args.maxcn, args.seed, args.cn_palette)
    log(f"outputs in {outdir}")


if __name__ == "__main__":
    main()
