"""Cell-by-bin heatmaps of (a, b) copy-number states, drawn with cnplot.

cnplot (https://github.com/raphael-group/cnplot) places the bins on a genome axis
(``GenomeAxis``) and draws a (cells, bins) matrix with ``plot_heatmap``. The (a, b) colors
come from ``get_cn_cmap(cn_palette)``, which keys every state under both orders, so a state
is drawn phase-free: (2, 1) and (1, 2) share a color. States the palette does not list take
its default color. The ``"cnamaste"`` palette needs cnplot 0.1.1b3 or later.

The states of every panel of a figure are coded on one shared colormap, so a color means
the same state in every panel.
"""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cnplot import GenomeAxis, get_cn_cmap, plot_heatmap
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
