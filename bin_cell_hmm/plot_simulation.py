"""The benchmark page: each method's metrics per simulated class, a box per method.

Copied from cna_maste_reproducibility ``scripts/plots/simulation.py`` (commit dce58a9):
``SAMPLE_ID``, ``JITTER``, ``FLOOR_TICKS``, ``compact_size``, ``sim_groups``,
``group_order``, ``get_complement``, ``set_floor_axis``, ``not_computed``, ``draw_groups``,
``draw_panels`` and ``plot_simulation_metrics``, with ``get_class_medians`` reduced to one
scoring level. The functions they import from that repository's ``scripts.plots.style``,
``scripts.plots.layout`` and from cnamaste's ``figure_style`` and ``plot.layout`` are
replaced by the local stand-ins below, with the same values: ``GAP`` 0.1 in, ``LETTER``
0.16 in, ``PANEL_ROW`` 2.3 in, a page 6.48 in wide, 8 pt STIX serif text, ``METHOD_ALPHA``
0.4, ``GROUP_WIDTH`` 0.8.

The page is 2 x 2 (``METRIC_PANELS``): N-CNA recall, per (bin, cell); 1 - normal
specificity on a log axis, a run at specificity 1 drawn at a floor ticked 0.00, as the
source's solver page draws it (``get_complement``, ``set_floor_axis``); the breakpoint rate
on a log axis, the truth's median per class a dashed line; and the clone ARI. Columns are
those of ``bin_cell_hmm/metrics.py`` that ``test/compare_chisel.py`` writes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator

SAMPLE_ID = re.compile(
    r"^numcnas(?P<numcnas>\d+(?:\.\d+)*)_cnasize(?P<cnasize>[\d.]+(?:[eE][+-]?\d+)?)"
    r"_ploidy(?P<ploidy>\d+(?:\.\d+)?)_random(?P<random>\d+)$"
)
"""A simulated sample's id, `numcnas<a.b>_cnasize<s>_ploidy<p>_random<r>`."""

JITTER = 0.1
"""Half-width of the points' horizontal jitter, in group positions."""

METHOD_COLOURS = {
    "chisel_caller_bic": "#2a78d6",
    "chisel_cloner_bic": "#eb6834",
    "hmm_grid": "#1baf7a",
    "hmm_chisel_caller_bic": "#eda100",
    "hmm_chisel_cloner_bic": "#e87ba4",
}
"""Each method's colour, the dataviz reference categorical slots 1-5 in drawing order (their
adjacent pairs validated for colour-vision deficiency)."""

METHOD_NAMES = {
    "chisel_caller_bic": "CHISEL Caller",
    "chisel_cloner_bic": "CHISEL Cloner",
    "hmm_grid": "HMM, grid labels",
    "hmm_chisel_caller_bic": "HMM + Caller.calling",
    "hmm_chisel_cloner_bic": "HMM + Caller.calling + Cloner",
}
"""How the page names each method."""

COMPLEMENT = "1 - specificity"
"""The column of 1 - normal specificity, drawn on a log axis (`get_complement`)."""

METRIC_PANELS = (
    ("exact_altered", r"$\mathbb{N}$-CNA recall"),
    (COMPLEMENT, r"1 $-$ normal specificity"),
    ("bp_cn_rate", "Breakpoint rate"),
    ("clone_ari", "Clone ARI"),
)
"""The panels, row by row, as `(column, label)`."""

LOG_PANELS = ("bp_cn_rate",)
"""Panels drawn on log axes ticked at 1, 2 and 5 per decade."""

TRUTH = "truth"
"""The method whose breakpoint rate is drawn as the reference."""

MEDIAN_COLUMNS = (
    "exact_altered",
    "exact_altered_hap",
    "specificity",
    "bp_cn_rate",
    "clone_ari",
    "sensitivity",
    "state_altered",
    "exact_cell_event",
    "bp_ratio",
    "seconds",
)
"""The columns `get_class_medians` summarizes, the page's first."""

FLOOR_TICKS = {1e-3: r"$10^{-3}$", 1e-2: r"$10^{-2}$", 1e-1: r"$10^{-1}$", 1.0: "1.00"}
"""A log rate panel's ticks above its floor, by value (`set_floor_axis`)."""

METHOD_ALPHA = 0.4
GROUP_WIDTH = 0.8
PANEL_ROW = 2.3
METRIC_AXES_HEIGHT = 1.43
GAP = 0.1
LETTER = 0.16
PAGE_WIDTH = 8.5 - 2 * 1.01
FONT_SIZE = 8.0
LOG_TICKS = 12


@contextmanager
def page_style():
    """Set the page's face for the block, as cnamaste's `figure_style.page_style`.

    Yields:
        The font size.
    """
    with mpl.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
        }
    ):
        yield FONT_SIZE


def get_method_colour(method: str) -> str:
    """Get a method's colour, grey for a method not in `METHOD_COLOURS`."""
    return METHOD_COLOURS.get(method, "#808080")


def get_method_order(table: pd.DataFrame, methods: list[str] | None) -> list[str]:
    """Get the methods to draw: `methods`, or every method of `METHOD_COLOURS` in `table`."""
    if methods:
        return list(methods)
    found = set(table["method"])
    return [method for method in METHOD_COLOURS if method in found]


def get_method_key(methods: list[str]) -> list[Patch]:
    """Get the methods' key: a patch per method in its colour at `METHOD_ALPHA`."""
    return [
        Patch(
            facecolor=to_rgba(get_method_colour(m), METHOD_ALPHA),
            edgecolor=to_rgba(get_method_colour(m), METHOD_ALPHA),
            label=METHOD_NAMES.get(m, m),
        )
        for m in methods
    ]


def set_text_sizes(figure: Figure, size: float) -> None:
    """Set every text on `figure` to `size`."""
    for text in figure.findobj(Text):
        text.set_fontsize(size)


def get_panel_rows(
    rows: int, columns: int = 2, sharex: bool = False
) -> tuple[Figure, np.ndarray]:
    """Get a page of `rows` rows of `columns` panels, `PANEL_ROW` a row, constrained layout,
    each panel padded `GAP` left and right and `LETTER` above and below."""
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(PAGE_WIDTH, PANEL_ROW * rows),
        layout="constrained",
        squeeze=False,
        sharex=sharex,
    )
    figure.get_layout_engine().set(w_pad=GAP, h_pad=LETTER)
    return figure, axes


def set_axes_height(figure: Figure, ax: Any, height: float) -> None:
    """Set the page height so `ax` is `height` inches tall; drawn twice."""
    figure.canvas.draw()
    drawn = ax.get_position().height * figure.get_figheight()
    figure.set_figheight(
        figure.get_figheight() + ax.get_gridspec().nrows * (height - drawn)
    )
    figure.canvas.draw()


def add_panel_letters(
    figure: Figure, axes: list[Any], size: float, titles: list[str] | None = None
) -> None:
    """Letter each panel (a), (b), ... at its top left, left of its tick labels; a panel's
    title is its letter's label."""
    renderer = figure.canvas.get_renderer()
    width = figure.get_size_inches()[0] * figure.dpi
    for ax, name, title in zip(axes, "abcdefghijkl", titles or [""] * len(axes)):
        x0 = ax.get_tightbbox(renderer).x0 / width
        figure.text(
            x0,
            ax.get_position().y1,
            f"({name})",
            label=title,
            va="bottom",
            ha="left",
            fontsize=size,
        )


def compact_size(size: float) -> str:
    """Get a CNA size in bases as `10M`, `50M`, `1.5M`, `500K` (`qa.plot.compact_size`).

    Args:
        size: the size in bases.

    Returns:
        The size with a K, M or G suffix where at least a thousand.
    """
    for scale, unit in ((1e9, "G"), (1e6, "M"), (1e3, "K")):
        if size >= scale:
            return f"{size / scale:g}{unit}"
    return f"{size:g}"


def sim_groups(sample_ids: Iterable[str]) -> pd.DataFrame:
    """Get simulated sample ids' parameters and group; an id that does not parse is dropped
    (`qa.plot.sim_groups`).

    Args:
        sample_ids: ids such as `numcnas1.2_cnasize5e7_ploidy2_random0`, repeats allowed.

    Returns:
        One row per parsed id, in input order: `sample_id`; `numcnas` the string (`"1.2"`);
        `cnasize`, `ploidy` floats; `random` int; `group` the id less `_random<r>`; `label`
        `(1,2, 50M)`.
    """
    rows = []
    for sample_id in sample_ids:
        found = SAMPLE_ID.match(str(sample_id))
        if found is None:
            continue
        numcnas, cnasize = found["numcnas"], float(found["cnasize"])
        rows.append(
            {
                "sample_id": str(sample_id),
                "numcnas": numcnas,
                "cnasize": cnasize,
                "ploidy": float(found["ploidy"]),
                "random": int(found["random"]),
                "group": str(sample_id)[: found.start("random") - len("_random")],
                "label": f"({numcnas.replace('.', ',')}, {compact_size(cnasize)})",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "sample_id",
            "numcnas",
            "cnasize",
            "ploidy",
            "random",
            "group",
            "label",
        ],
    )


def group_order(groups: pd.DataFrame) -> list[str]:
    """Get the groups in drawing order: `cnasize` largest first, then the sum of `numcnas`'
    components largest first, then by name (`qa.plot.group_order`).

    Args:
        groups: `sim_groups`' result.

    Returns:
        Each group once.
    """
    unique = groups.drop_duplicates("group").assign(
        total=lambda d: d["numcnas"].map(lambda n: sum(int(p) for p in n.split(".")))
    )
    return list(
        unique.sort_values(
            ["cnasize", "total", "group"], ascending=[False, False, True]
        )["group"]
    )


def get_class_medians(
    table: pd.DataFrame, methods: list[str] | None = None
) -> pd.DataFrame:
    """Get each method's median per simulated class over its replicates, NaN skipped.

    Args:
        table: the benchmark table, `sample_id`, `method` and `MEDIAN_COLUMNS`' columns.
        methods: the methods, in order, or None (`get_method_order`).

    Returns:
        One row per simulated class and method, classes in `group_order`: `sim_id`, `method`,
        `#replicates`, then the medians of the `MEDIAN_COLUMNS` in `table`.
    """
    groups = sim_groups(table["sample_id"].unique())
    columns = [c for c in MEDIAN_COLUMNS if c in table]
    rows = table.merge(groups[["sample_id", "group"]], on="sample_id").rename(
        columns={"group": "sim_id"}
    )
    rows = rows[rows["method"].isin(get_method_order(table, methods))]
    by = rows.groupby(["sim_id", "method"])
    medians = by[columns].median().join(by.size().rename("#replicates")).reset_index()
    ranks = {"sim_id": group_order(groups), "method": get_method_order(table, methods)}
    medians = medians.sort_values(
        ["sim_id", "method"], key=lambda s: s.map(ranks[str(s.name)].index)
    )
    return medians[["sim_id", "method", "#replicates", *columns]].reset_index(drop=True)


def get_complement(values: pd.Series) -> tuple[pd.Series, float]:
    """Get 1 - value of rates, and the floor a rate of 1 is drawn at on a log axis, where 0
    has no place: the power of 10 a decade under the lowest positive complement's decade, 0.1
    where none is positive.

    Args:
        values: the rates, at most 1, NaN where not computed.

    Returns:
        The complements, 0 at the floor, NaN kept; and the floor.
    """
    complement = 1 - pd.to_numeric(values, errors="coerce")
    positive = complement[complement > 0]
    floor = (
        float(10.0 ** (np.floor(np.log10(positive.min())) - 1))
        if len(positive)
        else 0.1
    )
    return complement.mask(complement == 0, floor), floor


def set_floor_axis(ax: Any, floor: float) -> None:
    """Set a rate panel's y axis to a log axis from `floor` to 1, as rates are drawn on
    [0, 1]: from half the floor to 2, the floor ticked `0.00` and `FLOOR_TICKS` above it, no
    minor ticks.

    Args:
        ax: the panel, drawn.
        floor: its floor (`get_complement`).
    """
    ax.set_yscale("log")
    ax.set_ylim(floor / 2, 2)
    ticks = {
        floor: "0.00",
        **{value: label for value, label in FLOOR_TICKS.items() if value > floor},
    }
    ax.set_yticks(list(ticks), list(ticks.values()))
    ax.yaxis.set_minor_locator(NullLocator())


def not_computed(ax: Any) -> None:
    """Write "not computed" across the middle of `ax` (`qa.plot.not_computed`)."""
    ax.text(
        0.5,
        0.5,
        "not computed",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="0.4",
    )
    ax.set_yticks([])


def draw_groups(
    ax: Any,
    values: list[np.ndarray],
    seed: int,
    offset: float,
    width: float,
    box: str,
    whisker: str,
) -> None:
    """Draw a box per non-empty group at its position `0, 1, ...` plus `offset`, its samples
    as jittered points (`qa.plot.draw_groups`).

    Args:
        ax: the panel.
        values: each group's samples, NaN dropped.
        seed: the jitter's generator seed.
        offset: the boxes' offset in group positions.
        width: the boxes' width in group positions.
        box: the box colour.
        whisker: the whisker colour.
    """
    rng = np.random.default_rng(seed)
    drawn = [k for k, v in enumerate(values) if v.size]
    if not drawn:
        return
    ax.boxplot(
        [values[k] for k in drawn],
        positions=[k + offset for k in drawn],
        widths=width,
        showfliers=False,
        patch_artist=True,
        boxprops={
            "facecolor": box,
            "alpha": METHOD_ALPHA,
            "edgecolor": whisker,
            "linewidth": 0.5,
        },
        whiskerprops={"color": whisker, "linewidth": 0.5},
        capprops={"color": whisker, "linewidth": 0.5},
        medianprops={"color": "black", "linewidth": 0.8},
        manage_ticks=False,
    )
    for k in drawn:
        x = (
            k
            + offset
            + rng.uniform(
                -JITTER * width / 0.6, JITTER * width / 0.6, size=values[k].size
            )
        )
        ax.scatter(
            x,
            values[k],
            s=3,
            color="black",
            alpha=0.7,
            edgecolors="black",
            linewidths=0.2,
            zorder=3,
        )


def draw_panels(
    axes: Any,
    table: pd.DataFrame,
    methods: list[str],
    panels: tuple[tuple[str, str], ...],
    series: str = "method",
    colours: dict[str, str] | None = None,
    floors: dict[str, float] | None = None,
) -> None:
    """Draw one column of `table` per panel, a box per simulated class and series, the classes
    along x.

    Args:
        axes: the panels, as many as `panels`.
        table: the simulation table, `sample_id`, `series` and `panels`' columns.
        methods: the series, in order, side by side in each class.
        panels: `(column, label)` per panel.
        series: the column naming each row's series, `method` by default.
        colours: each series' colour, or None for its method's (`get_method_colour`).
        floors: a log column's floor (`get_complement`), its panel's axis set by
            `set_floor_axis`, or None.
    """
    groups = sim_groups(table["sample_id"].unique())
    order, labels = (
        group_order(groups),
        dict(zip(groups["group"], groups["label"], strict=True)),
    )
    table = table.merge(groups[["sample_id", "group"]], on="sample_id")
    width = GROUP_WIDTH / len(methods)
    for k, (ax, (column, label)) in enumerate(zip(axes, panels, strict=True)):
        ax.set_ylabel(label)
        ax.set_xlim(-0.5, max(len(order), 1) - 0.5)
        floor = (floors or {}).get(column)
        drawn = False
        for i, method in enumerate(methods):
            rows = table[table[series] == method]
            found = (
                pd.to_numeric(rows[column], errors="coerce")
                if column in rows
                else pd.Series(dtype=float)
            )
            values = [found[rows["group"] == g].dropna().to_numpy(float) for g in order]
            if any(v.size for v in values):
                colour = (
                    get_method_colour(method) if colours is None else colours[method]
                )
                offset = (i - (len(methods) - 1) / 2) * width
                draw_groups(
                    ax,
                    values,
                    seed=k,
                    offset=offset,
                    width=0.9 * width,
                    box=colour,
                    whisker=colour,
                )
                drawn = True
        if not drawn:
            not_computed(ax)
        elif floor is not None:
            set_floor_axis(ax, floor)
        elif column in LOG_PANELS:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(
                LogLocator(subs=(1.0, 2.0, 5.0), numticks=LOG_TICKS)
            )
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticks(
            range(len(order)), [labels[g] for g in order], rotation=45, ha="right"
        )
        ax.set_xlabel("Simulated class")


def plot_simulation_metrics(
    table: pd.DataFrame,
    methods: list[str] | None = None,
    columns: tuple[str, ...] | None = None,
) -> Figure:
    """Draw each method's metrics per simulated class (`METRIC_PANELS`), those of `columns`
    alone where given, two a row, the classes named under the last row alone; keyed under the
    panels, three entries a line; each panel's letter labelled by its metric
    (`add_panel_letters`); the page as tall as `METRIC_AXES_HEIGHT` a panel makes it.
    Normal specificity is drawn as 1 - specificity on a log axis (`get_complement`), and the
    breakpoint panel carries the median rate of the `TRUTH` rows per class as a dashed line.

    Args:
        table: the benchmark table, a row per sample and method, the truth's rows included.
        methods: the methods to draw, in order, or None (`get_method_order`).
        columns: the metrics drawn, an even number of `METRIC_PANELS`' columns, or None for
            all of them.

    Returns:
        The figure, panels lettered (a), (b), ... in `METRIC_PANELS`' order.
    """
    methods = get_method_order(table, methods)
    panels = tuple(
        panel for panel in METRIC_PANELS if columns is None or panel[0] in columns
    )
    complement, floor = get_complement(table["specificity"])
    table = table.assign(**{COMPLEMENT: complement})
    with page_style() as size:
        figure, axes = get_panel_rows(len(panels) // 2, sharex=True)
        figure.get_layout_engine().set(h_pad=LETTER / 2, hspace=0.0)
        draw_panels(list(axes.flat), table, methods, panels, floors={COMPLEMENT: floor})
        key = get_method_key(methods)
        truth = table[table["method"] == TRUTH]
        if len(truth) and "bp_cn_rate" in dict(panels):
            ax = list(axes.flat)[[c for c, _ in panels].index("bp_cn_rate")]
            groups = sim_groups(table["sample_id"].unique())
            truth = truth.merge(groups[["sample_id", "group"]], on="sample_id")
            medians = truth.groupby("group")["bp_cn_rate"].median()
            for k, group in enumerate(group_order(groups)):
                if group in medians:
                    ax.hlines(
                        medians[group],
                        k - 0.45,
                        k + 0.45,
                        colors="0.3",
                        linestyles="--",
                        linewidth=0.8,
                    )
            key.append(
                Line2D(
                    [],
                    [],
                    color="0.3",
                    linestyle="--",
                    linewidth=0.8,
                    label="Truth (median)",
                )
            )
        for ax in axes.flat:
            ax.set_xlabel("")
        figure.legend(
            handles=key,
            loc="outside lower center",
            ncol=3,
            frameon=False,
            borderaxespad=0,
            fontsize=size,
            handlelength=1.6,
            columnspacing=1.2,
        )
        set_text_sizes(figure, size)
        set_axes_height(figure, axes[0, 0], METRIC_AXES_HEIGHT)
        add_panel_letters(
            figure,
            list(axes.flat),
            size,
            [f"{label}, simulations" for _, label in panels],
        )
    return figure
