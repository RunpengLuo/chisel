"""Simulate CHISEL-style single-cell RDR/BAF data from a clonal haplotype-specific copy-number model.

Copy numbers:
    A diploid normal clone (clone 0), a tumor ancestor with ``n_clonal_events`` segment events,
    and ``n_clones`` tumor clones that each add ``n_subclonal_events`` (star phylogeny). An event
    sets a contiguous run of bins on one chromosome to a random (a, b) with
    1 <= a + b <= max_cn and (a, b) != (1, 1). The last ``n_wgd_clones`` tumor clones double
    the ancestor, (a, b) -> (2a, 2b) (whole-genome doubling), before their subclonal events.
    Each cell copies its clone profile and adds Poisson(``cell_event_rate``) cell-specific
    events. With ``overlap`` False, the clonal and subclonal events of one lineage are placed
    on bins no earlier event of that lineage covers, so each event stays visible.

Benchmark classes:
    ``SIM_CLASSES`` lists (shared CNAs, private CNAs per clone, CNA length), from easy to hard.
    ``class_config`` gives one replicate of a class, every CNA exactly the class length,
    placed without overlap; ``sample_id`` names it as cna_maste's simulations do,
    ``numcnas<shared>.<private>_cnasize<length>_ploidy2_random<replicate>``.

Read counts:
    A per-bin bias g_b ~ LogNormal(0, bin_bias_sd) is shared by the normal sample and the cells.
    Normal count ~ Poisson(normal_reads_per_bin * g_b). Cell e has depth d_e ~ LogNormal(0,
    depth_sd); its count in bin b is Gamma-Poisson with mean
    reads_per_bin * d_e * g_b * C_be / mean_b(C_be) and squared coefficient of variation
    ``read_dispersion`` on the Gamma factor. RDR is computed as in CHISEL ``RDREstimator``:
    (count / normal) * (total_normal / total_cell).

Allele counts:
    SNP-covering reads n_be ~ Poisson(snp_reads_per_bin * d_e * C_be / mean_b(C_be)),
    B ~ Binomial(n_be, (1 - seq_error) * b / C + seq_error * a / C), A = n_be - B.
    BAF = B / (A + B), and 0.5 when A + B = 0 (CHISEL ``Combiner`` convention). Haplotype B is
    the same haplotype in every bin, i.e., phasing is error-free.
"""

import zlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimConfig:
    """Parameters of the simulation; see the module docstring for the generative model."""

    n_cells: int = 120
    n_chroms: int = 4
    bins_per_chrom: int = 50
    bin_size: int = 5_000_000
    n_clones: int = 3
    n_wgd_clones: int = 0
    normal_frac: float = 0.1
    n_clonal_events: int = 6
    n_subclonal_events: int = 3
    event_bins: tuple = (5, 25)
    overlap: bool = True
    cell_event_rate: float = 0.5
    cell_event_bins: tuple = (2, 6)
    max_cn: int = 5
    reads_per_bin: float = 800.0
    normal_reads_per_bin: float = 2000.0
    read_dispersion: float = 0.02
    bin_bias_sd: float = 0.1
    depth_sd: float = 0.2
    snp_reads_per_bin: float = 40.0
    seq_error: float = 0.01
    seed: int = 0


@dataclass
class SimData:
    """Simulated data; matrices are (n_cells, n_bins), bin vectors are (n_bins,)."""

    chroms: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    cells: np.ndarray
    clone: np.ndarray
    a: np.ndarray
    b: np.ndarray
    cell_event: np.ndarray
    normal: np.ndarray
    count: np.ndarray
    rdr: np.ndarray
    a_count: np.ndarray
    b_count: np.ndarray
    baf: np.ndarray


def _event_states(max_cn):
    return [
        (a, t - a)
        for t in range(1, max_cn + 1)
        for a in range(t + 1)
        if (a, t - a) != (1, 1)
    ]


def _add_events(profile, n_events, length_range, states, cfg, rng, occupied=None):
    out = profile.copy()
    for _ in range(n_events):
        length = min(
            int(rng.integers(length_range[0], length_range[1] + 1)), cfg.bins_per_chrom
        )
        for _ in range(10_000):
            start = int(rng.integers(cfg.n_chroms)) * cfg.bins_per_chrom
            start += int(rng.integers(0, cfg.bins_per_chrom - length + 1))
            if occupied is None or not occupied[start : start + length].any():
                break
        else:
            raise ValueError(
                "No free stretch for a non-overlapping event; fewer or shorter CNAs"
            )
        out[start : start + length] = states[int(rng.integers(len(states)))]
        if occupied is not None:
            occupied[start : start + length] = True
    return out


def simulate(cfg=SimConfig()):
    """Draw one dataset from the generative model.

    Args:
        cfg: Simulation parameters.

    Returns:
        SimData with true states (a, b), counts, RDR and BAF.

    Raises:
        ValueError: If ``n_wgd_clones`` exceeds ``n_clones``.
    """
    if not 0 <= cfg.n_wgd_clones <= cfg.n_clones:
        raise ValueError("n_wgd_clones must be in [0, n_clones]")
    rng = np.random.default_rng(cfg.seed)
    n_bins = cfg.n_chroms * cfg.bins_per_chrom
    chroms = np.repeat([f"chr{c + 1}" for c in range(cfg.n_chroms)], cfg.bins_per_chrom)
    starts = np.tile(np.arange(cfg.bins_per_chrom) * cfg.bin_size, cfg.n_chroms)
    states = _event_states(cfg.max_cn)

    diploid = np.ones((n_bins, 2), dtype=int)
    used = None if cfg.overlap else np.zeros(n_bins, dtype=bool)
    ancestor = _add_events(
        diploid, cfg.n_clonal_events, cfg.event_bins, states, cfg, rng, used
    )
    first_wgd = cfg.n_clones - cfg.n_wgd_clones
    clones = [diploid] + [
        _add_events(
            ancestor * (2 if i >= first_wgd else 1),
            cfg.n_subclonal_events,
            cfg.event_bins,
            states,
            cfg,
            rng,
            None if used is None else used.copy(),
        )
        for i in range(cfg.n_clones)
    ]
    n_normal = int(round(cfg.normal_frac * cfg.n_cells))
    clone = np.concatenate(
        [
            np.zeros(n_normal, dtype=int),
            rng.integers(1, cfg.n_clones + 1, cfg.n_cells - n_normal),
        ]
    )
    profiles = np.stack(
        [
            _add_events(
                clones[c],
                int(rng.poisson(cfg.cell_event_rate)),
                cfg.cell_event_bins,
                states,
                cfg,
                rng,
            )
            for c in clone
        ]
    )
    a, b = profiles[..., 0], profiles[..., 1]
    clone_profiles = np.stack([clones[c] for c in clone])
    cell_event = (profiles != clone_profiles).any(axis=-1)

    total_cn = a + b
    rel = total_cn / total_cn.mean(axis=1, keepdims=True)
    depth = rng.lognormal(0.0, cfg.depth_sd, cfg.n_cells)[:, None]
    bias = rng.lognormal(0.0, cfg.bin_bias_sd, n_bins)
    normal = rng.poisson(cfg.normal_reads_per_bin * bias)
    mean_count = cfg.reads_per_bin * depth * bias * rel
    if cfg.read_dispersion > 0:
        shape = 1.0 / cfg.read_dispersion
        mean_count = mean_count * rng.gamma(shape, 1.0 / shape, size=mean_count.shape)
    count = rng.poisson(mean_count)
    ratio = np.divide(count, normal, out=np.zeros(count.shape), where=normal > 0)
    rdr = ratio * (normal.sum() / count.sum(axis=1, keepdims=True))

    n_snp = rng.poisson(cfg.snp_reads_per_bin * depth * rel)
    frac_b = np.divide(b, total_cn, out=np.full(b.shape, 0.5), where=total_cn > 0)
    b_count = rng.binomial(
        n_snp, (1.0 - cfg.seq_error) * frac_b + cfg.seq_error * (1.0 - frac_b)
    )
    a_count = n_snp - b_count
    baf = np.divide(b_count, n_snp, out=np.full(b.shape, 0.5), where=n_snp > 0)

    return SimData(
        chroms=chroms,
        starts=starts,
        ends=starts + cfg.bin_size,
        cells=np.array([f"cell{i:04d}" for i in range(cfg.n_cells)]),
        clone=clone,
        a=a,
        b=b,
        cell_event=cell_event,
        normal=normal,
        count=count,
        rdr=rdr,
        a_count=a_count,
        b_count=b_count,
        baf=baf,
    )


SIM_CLASSES = (
    (1, 2, 10_000_000),
    (1, 2, 30_000_000),
    (1, 2, 50_000_000),
    (3, 3, 10_000_000),
    (3, 3, 30_000_000),
    (3, 3, 50_000_000),
    (6, 3, 10_000_000),
    (6, 3, 30_000_000),
    (6, 3, 50_000_000),
)
"""Benchmark classes: (shared CNAs, private CNAs per tumor clone, CNA length in bp)."""


def sample_id(n_shared, n_private, cna_size, replicate):
    """Name one replicate of a class as cna_maste's simulations do.

    Args:
        n_shared: Shared (clonal) CNAs.
        n_private: Private (subclonal) CNAs per tumor clone.
        cna_size: CNA length in bp.
        replicate: Replicate index.

    Returns:
        ``numcnas<shared>.<private>_cnasize<m>e<k>_ploidy2_random<replicate>``, e.g.
        ``numcnas1.2_cnasize1e7_ploidy2_random0``.
    """
    digits = str(int(cna_size))
    zeros = len(digits) - len(digits.rstrip("0"))
    size = f"{digits[: len(digits) - zeros] or 0}e{zeros}"
    return f"numcnas{n_shared}.{n_private}_cnasize{size}_ploidy2_random{replicate}"


def class_config(n_shared, n_private, cna_size, replicate, **overrides):
    """Return the SimConfig of one replicate of a benchmark class.

    Every shared and private CNA is exactly ``cna_size`` bp, placed without overlap within its
    lineage. The seed is the CRC32 of ``sample_id``, so a replicate is reproducible on its own.

    Args:
        n_shared: Shared (clonal) CNAs.
        n_private: Private (subclonal) CNAs per tumor clone.
        cna_size: CNA length in bp, a multiple of the bin size.
        replicate: Replicate index.
        **overrides: Other SimConfig fields; ``n_wgd_clones`` is not allowed, as the class id
            states ploidy 2.

    Returns:
        SimConfig.

    Raises:
        ValueError: If ``cna_size`` is not a multiple of the bin size, or WGD is requested.
    """
    if overrides.get("n_wgd_clones"):
        raise ValueError("benchmark classes are ploidy 2; n_wgd_clones is not allowed")
    bin_size = overrides.get("bin_size", SimConfig.bin_size)
    if cna_size % bin_size:
        raise ValueError(
            f"cna_size {cna_size} is not a multiple of bin_size {bin_size}"
        )
    length = cna_size // bin_size
    seed = zlib.crc32(sample_id(n_shared, n_private, cna_size, replicate).encode())
    fields = {
        "n_clonal_events": n_shared,
        "n_subclonal_events": n_private,
        "event_bins": (length, length),
        "overlap": False,
        "seed": seed,
    }
    return SimConfig(**{**fields, **overrides})


def write_combo(sim, path):
    """Write the data in CHISEL ``combo.tsv`` format (input of ``Caller.py``).

    Columns: CHR START END CELL NORMAL COUNT RDR A_COUNT B_COUNT BAF, one row per (bin, cell).

    Args:
        sim: Simulated data.
        path: Output path.
    """
    with open(path, "w") as out:
        out.write("#CHR\tSTART\tEND\tCELL\tNORMAL\tCOUNT\tRDR\tA_COUNT\tB_COUNT\tBAF\n")
        for t in range(len(sim.chroms)):
            head = f"{sim.chroms[t]}\t{sim.starts[t]}\t{sim.ends[t]}"
            for e, cell in enumerate(sim.cells):
                out.write(
                    f"{head}\t{cell}\t{sim.normal[t]}\t{sim.count[e, t]}\t{sim.rdr[e, t]:.10g}\t"
                    f"{sim.a_count[e, t]}\t{sim.b_count[e, t]}\t{sim.baf[e, t]:.10g}\n"
                )


def write_truth(sim, path):
    """Write the true states, one row per (bin, cell).

    Columns: CHR START END CELL CLONE A B CELL_EVENT (1 if the cell differs from its clone).

    Args:
        sim: Simulated data.
        path: Output path.
    """
    with open(path, "w") as out:
        out.write("#CHR\tSTART\tEND\tCELL\tCLONE\tA\tB\tCELL_EVENT\n")
        for t in range(len(sim.chroms)):
            head = f"{sim.chroms[t]}\t{sim.starts[t]}\t{sim.ends[t]}"
            for e, cell in enumerate(sim.cells):
                out.write(
                    f"{head}\t{cell}\t{sim.clone[e]}\t{sim.a[e, t]}\t{sim.b[e, t]}\t{int(sim.cell_event[e, t])}\n"
                )
