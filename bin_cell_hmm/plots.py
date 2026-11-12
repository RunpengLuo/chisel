"""Figures of per-(bin, cell) copy-number states, drawn with cnplot.

cnplot (https://github.com/raphael-group/cnplot) draws both figures:
- ``plot_state_heatmaps``: cell-by-bin heatmaps; ``GenomeAxis`` places the bins on a genome
  axis and ``plot_heatmap`` draws each (cells, bins) matrix.
- ``plot_rdr_baf_scatters``: (BAF, RDR) scatters, one dot per (cell, bin), BAF unmirrored on
  ``BAF_LIM``; ``plot_scatter_2d`` draws each page, with landmarks of expected states.

The (a, b) colors come from ``get_cn_cmap(cn_palette)``, which keys every state under both
orders, so a state is drawn phase-free: (2, 1) and (1, 2) share a color. States the palette
does not list take its default color. The ``"cnamaste"`` palette needs cnplot 0.1.1b3 or
later. Within a heatmap figure, the states of every panel are coded on one shared colormap,
so a color means the same state in every panel.
"""

import os
import tempfile
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cnplot import BAF_LIM, GenomeAxis, get_cn_cmap, plot_heatmap, plot_scatter_2d
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


def bins_table(chroms, starts, ends):
    """Return the bins as cnplot's coordinate table.

    Args:
        chroms: (n_bins,) chromosome of each bin.
        starts: (n_bins,) start of each bin.
        ends: (n_bins,) end of each bin.

    Returns:
        DataFrame with columns "#CHR", "START", "END", one row per bin.
    """
    return pd.DataFrame({"#CHR": chroms, "START": starts, "END": ends})


def genome_axis(coords_df):
    """Build a cnplot GenomeAxis over the chromosomes of ``coords_df``.

    Each chromosome ends at the end of its last bin, in order of first appearance; every
    chromosome is kept (no default exclusion of chrX, chrY, chrM).

    Args:
        coords_df: Bins with "#CHR", "START", "END".

    Returns:
        GenomeAxis with one full-length region per chromosome.
    """
    sizes = coords_df.groupby("#CHR", sort=False)["END"].max()
    with tempfile.NamedTemporaryFile("w", suffix=".sizes", delete=False) as fh:
        for chrom, size in sizes.items():
            fh.write(f"{chrom}\t{int(size)}\n")
    try:
        return GenomeAxis(None, fh.name, excluded_chroms=())
    finally:
        os.remove(fh.name)


def encode_states(calls, cn_palette="cnamaste"):
    """Code the phase-free (major, minor) states of several call matrices on one colormap.

    Args:
        calls: List of (a, b) pairs of (n_cells, n_bins) int arrays.
        cn_palette: One of ``cnplot.CN_PALETTES``.

    Returns:
        codes: List of (n_cells, n_bins) int arrays, the index of each entry's state.
        cmap: ListedColormap with one color per state.
        norm: BoundaryNorm mapping index i to color i.
        states: (major, minor) of each index, in the palette's legend order (by total copy
            number), then the states the palette does not list.
    """
    style, tcn_states = get_cn_cmap(cn_palette)
    present = {
        (int(x), int(y))
        for a, b in calls
        for x, y in zip(np.maximum(a, b).ravel(), np.minimum(a, b).ravel())
    }
    listed = [(x, y) for t in sorted(tcn_states) for x, y in tcn_states[t] if x >= y]
    states = [s for s in listed if s in present]
    states += sorted(present - set(states), key=lambda s: (sum(s), s[0]))
    keys = np.array([x * 1000 + y for x, y in states])
    by_key = np.argsort(keys)
    codes = []
    for a, b in calls:
        key = np.maximum(a, b) * 1000 + np.minimum(a, b)
        codes.append(by_key[np.searchsorted(keys[by_key], key)])
    cmap = ListedColormap([style.get(s, style["default"]) for s in states])
    norm = BoundaryNorm(np.arange(-0.5, len(states) + 0.5), len(states))
    return codes, cmap, norm, states


def state_legend(ax, states, cn_palette="cnamaste", max_cols=9):
    """Draw a key of the (major, minor) states just below an axes.

    The states the palette does not list share its default color and one entry.

    Args:
        ax: Axes the key hangs under, typically the bottom panel.
        states: (major, minor) states in key order.
        cn_palette: One of ``cnplot.CN_PALETTES``.
        max_cols: Entries per row; longer keys wrap to more rows.
    """
    handles = state_handles(states, cn_palette)
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=min(max_cols, len(handles)),
        frameon=False,
        title="(a, b), phase-free",
        handlelength=1.2,
        columnspacing=0.8,
    )


def plot_state_heatmaps(
    panels,
    coords_df,
    row_labels,
    path,
    cn_palette="cnamaste",
    panel_height=2.4,
    width=14.0,
):
    """Draw one cell-by-bin (a, b) heatmap per panel, stacked, with a shared key.

    Args:
        panels: List of (title, a, b), a and b (n_cells, n_bins) int arrays whose rows are
            in drawing order, bottom to top.
        coords_df: Bins with "#CHR", "START", "END", one row per matrix column.
        row_labels: (n_cells,) label per row; each contiguous run is boxed and ticked.
        path: Output image path.
        cn_palette: One of ``cnplot.CN_PALETTES``.
        panel_height: Height of each panel in inches.
        width: Figure width in inches.
    """
    axis = genome_axis(coords_df)
    codes, cmap, norm, states = encode_states(
        [(a, b) for _, a, b in panels], cn_palette
    )
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(width, panel_height * len(panels)), squeeze=False
    )
    fig.subplots_adjust(hspace=0.45)
    for i, (ax, (title, _, _), code) in enumerate(zip(axes[:, 0], panels, codes)):
        plot_heatmap(
            ax,
            code,
            coords_df,
            axis,
            row_labels=np.asarray(row_labels),
            plot_chrname=i == 0,
            title=title,
            cmap=cmap,
            norm=norm,
        )
    state_legend(axes[-1, 0], states, cn_palette)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def state_handles(states, cn_palette="cnamaste"):
    """Return key patches of (major, minor) states, the unlisted ones as one "other" entry.

    Args:
        states: (major, minor) states in key order.
        cn_palette: One of ``cnplot.CN_PALETTES``.

    Returns:
        List of Patch.
    """
    style, _ = get_cn_cmap(cn_palette)
    handles = [
        Patch(
            facecolor=style[s],
            edgecolor="black",
            linewidth=0.5,
            label=f"({s[0]},{s[1]})",
        )
        for s in states
        if s in style
    ]
    if any(s not in style for s in states):
        handles.append(
            Patch(
                facecolor=style["default"],
                edgecolor="black",
                linewidth=0.5,
                label="other",
            )
        )
    return handles


def plot_rdr_baf_scatters(
    rdr, baf, pages, path, landmarks=None, cn_palette="cnamaste", ylim=None
):
    """Write a PDF of (BAF, RDR) scatters, one page per coloring, one dot per (cell, bin).

    Each page is cnplot's ``plot_scatter_2d``: BAF on x over ``BAF_LIM`` (unmirrored, as
    given), RDR on y, the dots colored by that page's (a, b) calls, a marginal density per
    state, and ``landmarks`` drawn on every page. A key of the page's states sits right of
    the plot.

    Args:
        rdr: (n_cells, n_bins) RDR.
        baf: (n_cells, n_bins) BAF, B / (A + B); non-finite entries are left out.
        pages: List of (title, a, b), a and b (n_cells, n_bins) int calls coloring the dots.
        path: Output PDF path.
        landmarks: cnplot ``expected_df`` with ``cn_<name>`` ("a|b"), ``exp_BAF``,
            ``exp_RDR`` and optional ``is_balanced``, or None for no landmarks.
        cn_palette: One of ``cnplot.CN_PALETTES``.
        ylim: RDR limits; None takes (0, max(3, 99.9th percentile of RDR)).
    """
    style, tcn_states = get_cn_cmap(cn_palette)
    listed = [(x, y) for t in sorted(tcn_states) for x, y in tcn_states[t] if x >= y]
    obs = pd.DataFrame({"BAF": np.ravel(baf), "RDR": np.ravel(rdr)})
    if ylim is None:
        top = np.nanquantile(obs["RDR"].to_numpy(dtype=float), 0.999)
        ylim = (0.0, max(3.0, float(top)))
    with PdfPages(path) as pdf:
        for title, a, b in pages:
            a, b = np.ravel(a), np.ravel(b)
            labels = np.array([f"{x}|{y}" for x, y in zip(a, b)])
            palette = {
                lab: style.get(tuple(int(v) for v in lab.split("|")), style["default"])
                for lab in np.unique(labels)
            }
            with warnings.catch_warnings():
                # NB: seaborn skips the density of a state whose points share one value
                warnings.filterwarnings("ignore", message="Dataset has 0 variance")
                grid = plot_scatter_2d(
                    obs,
                    "BAF",
                    "RDR",
                    expected_df=landmarks,
                    hue=labels,
                    palette=palette,
                    xlim=BAF_LIM,
                    ylim=ylim,
                    xlabel="BAF",
                    ylabel="RDR",
                    title=title,
                )
            present = {(int(max(x, y)), int(min(x, y))) for x, y in zip(a, b)}
            states = [s for s in listed if s in present]
            states += sorted(present - set(states), key=lambda s: (sum(s), s[0]))
            grid.figure.legend(
                handles=state_handles(states, cn_palette),
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                frameon=False,
                title="(a, b), phase-free",
            )
            pdf.savefig(grid.figure, bbox_inches="tight", dpi=200)
            plt.close(grid.figure)
