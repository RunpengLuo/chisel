# Benchmark: CHISEL vs the bin-cell HMM on simulated data

Everything under `bin_cell_hmm/test/out/` is generated and gitignored.

## Methods compared

Each sample is scored per (bin, cell) for these rows of its `metrics.tsv`, after the truth's reference row ([`compare_chisel.METHODS`](../test/compare_chisel.py)). The suffix `_bic` marks calls whose base ploidy per cell is the one `Caller.calling`'s BIC selected (`-P 4`: ploidy 2 or 4); `hmm/ploidy_bic.tsv` lists that choice per cell for the HMM run.

| Row | Clustering | Copy numbers |
|---|---|---|
| `chisel_caller_bic` | `kclustering` | `Caller.calling` (`CN_STATE`) |
| `chisel_cloner_bic` | `kclustering` | `Cloner` (`CORRECTED_HAP_CN`), CHISEL's final output |
| `hmm_grid` | `hmmclustering` on CHISEL's mirrored data matrix | each state's nearest grid point, base ploidy 2 for every cell |
| `hmm_phased_grid` | `BinCellHMM` on unmirrored BAF, haplotype-specific grid | as `hmm_grid` |
| `hmm_chisel_caller_bic` | `hmmclustering` (via `hmm_caller.py`) | `Caller.calling` per cell |
| `hmm_chisel_cloner_bic` | `hmmclustering` (via `hmm_caller.py`) | `Cloner` on `hmm_chisel_caller_bic`'s calls |

By default K is the number of distinct true states (oracle). `Cloner` gives every cell of a clone the clone's majority state per bin (`Cloner.profiling`), so it removes cell-specific CNAs by construction.

## Metrics

The metrics are in [metrics.py](../metrics.py). The supervised ones follow cna_maste's simulation benchmark (`qa/metrics.py`, `benchmark/scripts/metrics/supervised_metrics.py`), with one (bin, cell) as the unit and AB pairs compared phase-free.

| Column | Definition |
|---|---|
| `sensitivity` | 1 - (truly altered called (1, 1)) / (truly altered) |
| `specificity` | 1 - (truly (1, 1) called altered) / (truly (1, 1)) |
| `exact_altered` | N-CNA recall: (truly altered called their true AB pair) / (truly altered) |
| `state_altered` | CNA recall: (truly altered called their true CNA state, one of neu, del, amp, cnloh, bamp, bdel) / (truly altered) |
| `exact_cell_event` | (cell-specific-event entries called their true AB pair) / (cell-specific-event entries) |
| `clone_ari` | ARI of `Cloner`'s clones against the true clones, over cells with a clone |
| `boundaries` | adjacent bin pairs on the same chromosome, summed over cells: `n_cells * sum_chrom (n_bins_chrom - 1)` |
| `bp_cn`, `bp_cn_rate` | boundaries whose phase-free called pair changes, and their share of `boundaries` |
| `bp_label`, `bp_label_rate` | as `bp_cn`, for the method's own labels: CHISEL's bin cluster (identical in every cell), the HMM state |
| `bp_ratio` | (benchmark table) `bp_cn` over the truth's `bp_cn` |
| `seconds` | wall time; a Cloner row includes its Caller run |

## One sample: `test/compare_chisel.py`

In `test/out/sim_seed<seed>[_wgd<n>]/` it writes:
- the simulated `combo.tsv` and `truth.tsv`;
- CHISEL's `chisel/calls_bic.tsv`, `calls_cloned_bic.tsv`, `mapping_bic.tsv`;
- the HMM's `hmm/calls_bic.tsv`, `calls_cloned_bic.tsv`, `mapping_bic.tsv`, `ploidy_bic.tsv`, `hmm.npz`;
- `metrics.tsv`;
- four figures: `cn_heatmaps.png` (total copy number), `error_maps.png` (allele-specific mismatches), `ab_heatmaps.pdf` ((a, b) states drawn with cnplot's `--cn-palette`, `cnamaste` by default) and `rdr_baf_states.png` (the HMM's states in (RDR, BAF)).

`--n-wgd-clones N` makes the last N tumor clones whole-genome doubled. The default simulation is 120 cells (12 normal) x 200 bins (4 chromosomes x 50 bins of 5 Mb); see [simulation.md](simulation.md).

## Benchmark: `test/run_benchmark.py`

It simulates each class of `simulation.SIM_CLASSES` (shared CNAs, private CNAs per clone, CNA length), from easy to hard, with `--replicates` replicates each:

| Class | Shared CNAs | Private CNAs per clone | CNA length |
|---|---|---|---|
| (1,2, 10M), (1,2, 30M), (1,2, 50M) | 1 | 2 | 10, 30, 50 Mb |
| (3,3, 10M), (3,3, 30M), (3,3, 50M) | 3 | 3 | 10, 30, 50 Mb |
| (6,3, 10M), (6,3, 30M), (6,3, 50M) | 6 | 3 | 10, 30, 50 Mb |

A sample is named `numcnas<shared>.<private>_cnasize<length>_ploidy2_random<replicate>` and seeded by that name, so each replicate is an independent draw, reproducible on its own. Within a lineage, the CNAs are placed without overlap.

In `test/out/benchmark/` it writes:
- one directory per sample: the outputs of one sample above except `cn_heatmaps.png`, `error_maps.png` and `rdr_baf_states.png`, plus `ab_heatmaps.pdf`, the (a, b) state of every cell and bin for the truth and each plotted method (`--cn-palette`, `cnamaste` by default);
- `benchmark_metrics.tsv`, every sample and method;
- `benchmark_medians.tsv`, each method's median per class;
- `benchmark_metrics.pdf`/`.png`, the metric page. Per class it has a box per method for `chisel_caller_bic`, `chisel_cloner_bic`, `hmm_grid`, `hmm_chisel_caller_bic` and `hmm_chisel_cloner_bic`, in a 2 x 2 grid ([plot_simulation.py](../plot_simulation.py)):
  - N-CNA recall (`exact_altered`, per (bin, cell));
  - 1 - normal specificity, on a log axis, a run at specificity 1 drawn at a floor ticked 0.00;
  - the breakpoint rate (`bp_cn_rate`), on a log axis, with the truth's median per class as a dashed line;
  - the clone ARI, of the Cloner rows only.

A sample whose `metrics.tsv` exists is skipped, so an interrupted run resumes where it stopped; `--plot-only` runs no sample and redraws the page, plus the `ab_heatmaps.pdf` of any finished sample without one (read back from its calls on disk). Keep `--workers` times `--jobs` within the CPU count.
