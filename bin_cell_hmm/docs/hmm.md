# Bin-cell HMM: model

The full model is written in [hmm.tex](hmm.tex). Build `hmm.pdf` in this folder with `latexmk -pdf hmm.tex && latexmk -c`. This page summarizes the model, then maps each part to its section in `hmm.tex` and to the code that implements it.

> [!NOTE]
> Code links point to line numbers, so they drift when the code is edited. The function names are the stable reference.

## Summary

For cell `e` and bin `t`, the observation is the 2-dim point `x[e,t] = (RDR, BAF)`. The BAF is taken as phased: `B / (A + B)` with B the same haplotype in every bin and cell, on [0, 1], not mirrored, and no emission mean is constrained to BAF <= 0.5.

| Component | Definition |
|---|---|
| Hidden state | `z[e,t]` in `{1..K}`; the K states are shared by all cells |
| Start | `P(z[e,first bin of a chromosome] = k) = 1/K`, fixed |
| Transition | `P(z[e,t] = l \| z[e,t-1] = k) = stay` if `k = l`, else `(1 - stay) / (K - 1)`, fixed and shared by all cells ([`make_transmat`](../hmm.py#L43), same construction as hatchet-long-read `cluster_bins/hmm/hmm_transitions.py`) |
| Emission | `x[e,t] \| z = k ~ N(mu_k, diag(var_k))`; a non-finite value is marginalized out |
| Initialization | constrained k-means: K distinct centers chosen from the integer grid `(a, b) -> ((a + b) / base, b / (a + b))` with `a + b <= maxcn`, both `(a, b)` and `(b, a)`; initial variances are the pooled within-cluster variances |
| Learning | Baum-Welch on `mu_k` and `var_k` only (`var_k >= varfloor`); start and transitions stay fixed |
| Decoding | Viterbi per cell and chromosome; `BinCellHMM.state_cn()` labels each state with the grid point nearest to its fitted mean |
| Per-cell labels | `hmm_caller.py` labels each cell's states with CHISEL's `Caller.calling`: base cluster, RDR scale per even base ploidy, ploidy chosen by BIC; each call oriented by its state's fitted BAF ([interface.md](interface.md)) |

Each chromosome is processed for all cells at once, so forward-backward and Viterbi are vectorized over cells. Chains restart at chromosome boundaries.

## Limitations

- K is an input. CHISEL chooses its number of clusters by binary search on the clustering objective (`--threshold`).
- One set of emission means for all cells assumes a common RDR scale. A cell whose mean ploidy differs (e.g., after whole-genome doubling) shifts the RDR of every (a, b), and the HMM spends states on these shifted copies. `hmm_caller.py` rescales each cell only after clustering, through `Caller.calling`.
- The base ploidy is not identifiable from RDR and BAF alone: (a, b) at one scale and (2a, 2b) at half that scale predict the same values. Only `Caller.calling`'s BIC penalty separates them.
- The Gaussian BAF emission ignores the number of SNP-covering reads behind each BAF.
- The BAF is taken as phased. Phase switches between bins are not modeled: CHISEL's `Combiner` makes B each unbalanced bin's pseudo-bulk minor allele, so B can switch haplotype between adjacent bins, and each switch reads as a change between `(a, b)` and `(b, a)` states.

## Map to hmm.tex and the code

### Data

| Part | hmm.tex | Code |
|---|---|---|
| RDR `r` | Sec. 1, eq. (1) ([L30](hmm.tex#L30)) | CHISEL [`RDREstimator.main`](../../src/chisel/RDREstimator.py#L126-L127) |
| BAF `beta` and mirrored BAF | Sec. 1, eq. (2) ([L44](hmm.tex#L44)) | CHISEL [`Combiner.main`](../../src/chisel/Combiner.py#L149), [`Caller.read_combo`](../../src/chisel/Caller.py#L154) (`mirroredBAF`) |
| Data matrix `D` (T x 2N) | Sec. 1 ([L52](hmm.tex#L52)) | CHISEL [`Caller.format_combo`](../../src/chisel/Caller.py#L167) (mirrored BAF); [`format_phased`](../hmm_caller.py#L95) (the same layout, phased BAF); [`split_data`](../hmm.py#L239) |
| Missing values `m` | Sec. 1 | [`BinCellHMM._prepare`](../hmm.py#L342) |
| Chromosome blocks | Sec. 1 | [`chrom_blocks`](../hmm.py#L260) |

### Model

| Part | hmm.tex | Code |
|---|---|---|
| Hidden states shared by all cells | Sec. 2.1 ([L65](hmm.tex#L65)) | [`BinCellHMM`](../hmm.py#L281) |
| Start `pi = 1/K`, transitions `A` | Sec. 2.2, eq. (3) ([L71](hmm.tex#L71)) | [`make_transmat`](../hmm.py#L43), same construction as hatchet-long-read `src/hatchet/cluster_bins/hmm/hmm_transitions.py`; set in [`_init_params`](../hmm.py#L354) |
| Emission, missing values marginalized | Sec. 2.3, eq. (4) ([L87](hmm.tex#L87)) | [`log_emission`](../hmm.py#L158) |
| Likelihood | Sec. 2.4, eq. (5) ([L96](hmm.tex#L96)) | [`forward_backward`](../hmm.py#L206) (returned log-likelihood) |

### Initialization

| Part | hmm.tex | Code |
|---|---|---|
| Copy-number grid `g(a, b)` | Sec. 3.1, eq. (6) ([L107](hmm.tex#L107)) | [`cn_grid`](../hmm.py#L63) |
| Constrained k-means on the grid | Sec. 3.2, eq. (7) ([L121](hmm.tex#L121)) | [`constrained_kmeans`](../hmm.py#L103), [`_snap_centers`](../hmm.py#L90) |
| Initial means and tied variances | Sec. 3.3, eq. (8) ([L142](hmm.tex#L142)) | [`BinCellHMM._init_params`](../hmm.py#L354) |

### Baum-Welch

| Part | hmm.tex | Code |
|---|---|---|
| Scaled forward-backward, posteriors, log-likelihood | Sec. 4.1, eqs. (9)-(11) ([L156](hmm.tex#L156)) | [`forward_backward`](../hmm.py#L175) |
| Sufficient statistics | Sec. 4.2 ([L179](hmm.tex#L179)) | [`BinCellHMM._e_step`](../hmm.py#L378) |
| Mean and floored variance update | Sec. 4.2, eq. (14) ([L196](hmm.tex#L196)) | [`BinCellHMM._m_step`](../hmm.py#L395) |
| Stopping rule | Sec. 4.3 ([L203](hmm.tex#L203)) | [`BinCellHMM.fit`](../hmm.py#L403) |

### Decoding and labels

| Part | hmm.tex | Code |
|---|---|---|
| Viterbi path | Sec. 5 ([L209](hmm.tex#L209)) | [`viterbi`](../hmm.py#L210), [`BinCellHMM.predict`](../hmm.py#L438); posteriors [`predict_proba`](../hmm.py#L458) |
| `members` (T x N) | Sec. 5 | [`hmmclustering`](../hmm.py#L489) |
| Grid labels (nearest grid point) | Sec. 6.1 ([L223](hmm.tex#L223)) | [`BinCellHMM.state_cn`](../hmm.py#L480) |
| Per-cell clusters `C_k,e` and summaries | Sec. 6.2 ([L232](hmm.tex#L232)) | [`cell_clusters`](../hmm_caller.py#L120); CHISEL [`Caller.est_clusters`](../../src/chisel/Caller.py#L172) |
| Base cluster | Sec. 6.2, base cluster ([L246](hmm.tex#L246)) | CHISEL [`Caller.identify_base`](../../src/chisel/Caller.py#L322), [`Combiner.EM`](../../src/chisel/Combiner.py#L326) |
| Scale per base ploidy | Sec. 6.2, scale ([L260](hmm.tex#L260)) | CHISEL [`Caller.calling`](../../src/chisel/Caller.py#L267) |
| Variance floors | Sec. 6.2, variance floors ([L264](hmm.tex#L264)) | CHISEL [`Caller.calling`](../../src/chisel/Caller.py#L276-L285) |
| State of each cluster | Sec. 6.2, state of each cluster ([L274](hmm.tex#L274)) | CHISEL [`Caller.calling`](../../src/chisel/Caller.py#L287-L304) |
| Base ploidy by BIC | Sec. 6.2, eq. (22) ([L287](hmm.tex#L287)) | CHISEL [`Caller.calling`](../../src/chisel/Caller.py#L306-L317) |
| Calling every cell | Sec. 6.2 | [`call_cells`](../hmm_caller.py#L174): `Pool` of CHISEL [`Caller.init`](../../src/chisel/Caller.py#L209) + [`Caller.calling`](../../src/chisel/Caller.py#L259) |
| Orientation and phasing | Sec. 6.2, orientation and phasing ([L302](hmm.tex#L302)) | [`orient`](../hmm_caller.py#L203) (by the HMM state's fitted BAF; with `--hmm-orient bin`, the rule of CHISEL [`Caller.infer`](../../src/chisel/Caller.py#L348)); CHISEL [`Caller.phase`](../../src/chisel/Caller.py#L354), [`Caller.phasing`](../../src/chisel/Caller.py#L370) |

### Interface and parameters

| Part | hmm.tex | Code |
|---|---|---|
| `hmmclustering` vs `kclustering` | Sec. 7 ([L317](hmm.tex#L317)) | [`hmmclustering`](../hmm.py#L489); CHISEL [`Clusterizer.kclustering`](../../src/chisel/Clusterizer.py#L19) |
| CHISEL Caller with the HMM | Sec. 6.2, 7 | [`hmm_caller.main`](../hmm_caller.py#L231) |
| Hyper-parameters and defaults | Sec. 8 ([L332](hmm.tex#L332)) | [`BinCellHMM.__init__`](../hmm.py#L309), [`hmm_caller.parse_args`](../hmm_caller.py#L74); CHISEL `Caller.parse_args` |
| Complexity | Sec. 9 ([L357](hmm.tex#L357)) | [`forward_backward`](../hmm.py#L175), [`viterbi`](../hmm.py#L210) |
| Limitations | Sec. 10 ([L364](hmm.tex#L364)) | - |
