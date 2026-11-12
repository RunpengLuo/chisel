# Interface: a swap-in for CHISEL's global clustering

`hmm.hmmclustering` is a drop-in counterpart of CHISEL's global clustering, `Clusterizer.kclustering`. It takes the same RDR/BAF data-matrix layout, with the phased BAF in place of CHISEL's mirrored one, but every cell gets its own Markov chain along the genome over K emission states shared by all cells. It returns a state per (bin, cell) rather than a cluster per bin. CHISEL's own `Caller.calling` then turns those states into copy numbers per cell.

## `kclustering` and `hmmclustering`

| | `Clusterizer.kclustering` | `hmm.hmmclustering` |
|---|---|---|
| Data | `data`: `(n_bins, 2 * n_cells)`, columns `[RDR_e1, mBAF_e1, RDR_e2, ...]` from `Caller.format_combo`, the BAF mirrored | the same layout with the phased BAF, `[RDR_e1, BAF_e1, ...]`, from [`hmm_caller.format_phased`](../hmm_caller.py#L95); NaN where a cell has no SNP-covering read |
| Extra input | - | `chroms`: chromosome of each row of `data` |
| Hyper-parameters | `restarts, threshold, seed, lord, j, LB, UB` | `K, stay, maxcn, base, restarts, maxiter, tol, varfloor, seed` |
| Output | `members`: `(n_bins,)` cluster per bin | `members`: `(n_bins, n_cells)` state per (bin, cell); `members[mapb[b], x]` is bin `b` in the x-th sorted cell |

## `hmm_caller.py`: CHISEL's Caller with the HMM

The code downstream in `Caller` (`est_clusters`, `call`, `infer`) indexes `members[mapb[b]]` as one cluster per bin. [hmm_caller.py](../hmm_caller.py) is `Caller.main` with that glue made per cell. Its copy-number calls come from CHISEL's own `Caller.calling`, unchanged:

| Step | `Caller.main` | `hmm_caller.py` |
|---|---|---|
| read | `read_combo`, `format_combo` | `read_combo`; `format_phased`: `format_combo`'s rows and cells, phased BAF |
| cluster | `kclustering` | `hmmclustering` |
| summarize clusters | `est_clusters`: cluster c = bins with `members[mapb[b]] == c` | `cell_clusters`: cluster c of cell e = bins with `members[mapb[b], x_e] == c`; same RDR, BAF and size formulas |
| call | `call`: Pool of `init` + `calling` | `call_cells`: Pool of the same `init` + `calling` |
| orient, phase | `infer`, `phase` | `orient`: by default (`--hmm-orient state`) by the fitted BAF of the (bin, cell)'s HMM state, no `phase`, as the BAF is phased; with `--hmm-orient bin`, `infer`'s rule per (bin, cell), then the same `phase` |

`Caller.calling(e)` reads only `clusters[c][e]`, `counts[e][c]` and `rdrs[e][c]`, so a cluster membership that differs between cells needs no change to it. Per cell, it finds the largest balanced cluster (`identify_base`), scales RDR for each even base ploidy up to `-P`, and picks the ploidy by BIC ([hmm.tex](hmm.tex), Sec. 6.2).

```sh
$CHISEL_PY27 bin_cell_hmm/hmm_caller.py combo.tsv --hmm-states 11 -j 12 --seed 0 > calls_bic.tsv
$CHISEL_PY27 src/chisel/Cloner.py calls_bic.tsv > mapping_bic.tsv   # unchanged CHISEL Cloner
```

`--hmm-*` options go to `hmmclustering`; every other option is `Caller.py`'s.

> [!IMPORTANT]
> `hmm.py` must stay importable by CHISEL's runtime (Python 2.7, numpy 1.16). Do not use f-strings, `@`, type hints or `numpy.random.default_rng` in it. `pytest/test_hmm.py::test_hmm_python27_matches_python3` checks this. `hmm_caller.py` imports CHISEL's `Caller`, so it runs under Python 2.7 only. Every other module and the scripts under `test/` are Python 3.

## Files

| Path | Content |
|---|---|
| [hmm.py](../hmm.py) | model: grid, constrained k-means, forward-backward, Viterbi, `BinCellHMM`, `hmmclustering` |
| [hmm_caller.py](../hmm_caller.py) | Python 2.7: CHISEL's `Caller` with `hmmclustering` in place of `kclustering` |
| [simulation.py](../simulation.py) | clonal haplotype-specific copy-number simulator writing CHISEL `combo.tsv`; benchmark classes ([simulation.md](simulation.md)) |
| [metrics.py](../metrics.py) | scores of per-(bin, cell) calls: sensitivity, specificity, N-CNA recall, CNA recall, breakpoints |
| [plots.py](../plots.py) | cell-by-bin (a, b) state heatmaps drawn with cnplot |
| [plot_simulation.py](../plot_simulation.py) | the benchmark's metric page, copied from cna_maste_reproducibility `scripts/plots/simulation.py` |
| [test/compare_chisel.py](../test/compare_chisel.py) | one sample: simulate, run CHISEL and the HMM, score, plot |
| [test/run_benchmark.py](../test/run_benchmark.py) | every class and replicate, the benchmark table and metric page |
| [pytest/](../pytest/) | unit tests |
| [hmm.tex](hmm.tex), [hmm.md](hmm.md) | the full model, and its map to the code |
| [benchmark.md](benchmark.md) | methods compared, metrics, outputs of one sample and of the benchmark |
| [simulation.md](simulation.md) | the simulation's steps, linked to the code |
