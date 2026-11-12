# Bin-cell HMM: model

The full model is written in [hmm.tex](hmm.tex). Build `hmm.pdf` in this folder with `latexmk -pdf hmm.tex && latexmk -c`. This page summarizes the model, then maps each part to its section in `hmm.tex` and to the code that implements it.

> [!NOTE]
> Code links point to line numbers, so they drift when the code is edited. The function names are the stable reference.

## Summary

For cell `e` and bin `t`, the observation is the 2-dim point `x[e,t] = (RDR, BAF)`.

| Component | Definition |
|---|---|
| Hidden state | `z[e,t]` in `{1..K}`; the K states are shared by all cells |
| Start | `P(z[e,first bin of a chromosome] = k) = 1/K`, fixed |
| Transition | `P(z[e,t] = l \| z[e,t-1] = k) = stay` if `k = l`, else `(1 - stay) / (K - 1)`, fixed and shared by all cells ([`make_transmat`](../hmm.py#L38), same construction as hatchet-long-read `cluster_bins/hmm/hmm_transitions.py`) |
| Emission | `x[e,t] \| z = k ~ N(mu_k, diag(var_k))`; a non-finite value is marginalized out |
| Initialization | constrained k-means: K distinct centers chosen from the integer grid `(a, b) -> ((a + b) / base, b / (a + b))` with `a + b <= maxcn` (and `a >= b` when `mirrored`); initial variances are the pooled within-cluster variances |
| Learning | Baum-Welch on `mu_k` and `var_k` only (`var_k >= varfloor`); start and transitions stay fixed |
| Decoding | Viterbi per cell and chromosome; `BinCellHMM.state_cn()` labels each state with the grid point nearest to its fitted mean |
| Per-cell labels | `hmm_caller.py` labels each cell's states with CHISEL's `Caller.calling`: base cluster, RDR scale per even base ploidy, ploidy chosen by BIC ([interface.md](interface.md)) |

Each chromosome is processed for all cells at once, so forward-backward and Viterbi are vectorized over cells. Chains restart at chromosome boundaries.

## Limitations

- K is an input. CHISEL chooses its number of clusters by binary search on the clustering objective (`--threshold`).
- One set of emission means for all cells assumes a common RDR scale. A cell whose mean ploidy differs (e.g., after whole-genome doubling) shifts the RDR of every (a, b), and the HMM spends states on these shifted copies. `hmm_caller.py` rescales each cell only after clustering, through `Caller.calling`.
- The base ploidy is not identifiable from RDR and BAF alone: (a, b) at one scale and (2a, 2b) at half that scale predict the same values. Only `Caller.calling`'s BIC penalty separates them.
- The Gaussian BAF emission ignores the number of SNP-covering reads behind each BAF.
- With mirrored BAF the states are allele-specific; haplotype-specific states need the unmirrored BAF and consistent phasing.

## Map to hmm.tex and the code

### Data

| Part | hmm.tex | Code |
|---|---|---|
| RDR `r` | Sec. 1, eq. (1) ([L30](hmm.tex#L30)) | CHISEL [`RDREstimator.main`](../../src/chisel/RDREstimator.py#L126-L127) |
| BAF `beta` and mirrored BAF | Sec. 1, eq. (2) ([L44](hmm.tex#L44)) | CHISEL [`Combiner.main`](../../src/chisel/Combiner.py#L149), [`Caller.read_combo`](../../src/chisel/Caller.py#L154) (`mirroredBAF`) |
| Data matrix `D` (T x 2N) | Sec. 1 ([L50](hmm.tex#L50)) | CHISEL [`Caller.format_combo`](../../src/chisel/Caller.py#L167); [`split_data`](../hmm.py#L236) |
| Missing values `m` | Sec. 1 | [`BinCellHMM._prepare`](../hmm.py#L342) |
| Chromosome blocks | Sec. 1 | [`chrom_blocks`](../hmm.py#L257) |

### Model

| Part | hmm.tex | Code |
|---|---|---|
| Hidden states shared by all cells | Sec. 2.1 ([L63](hmm.tex#L63)) | [`BinCellHMM`](../hmm.py#L278) |
| Start `pi = 1/K`, transitions `A` | Sec. 2.2, eq. (3) ([L69](hmm.tex#L69)) | [`make_transmat`](../hmm.py#L38), same construction as hatchet-long-read `src/hatchet/cluster_bins/hmm/hmm_transitions.py`; set in [`_init_params`](../hmm.py#L354) |
| Emission, missing values marginalized | Sec. 2.3, eq. (4) ([L85](hmm.tex#L85)) | [`log_emission`](../hmm.py#L155) |
| Likelihood | Sec. 2.4, eq. (5) ([L94](hmm.tex#L94)) | [`forward_backward`](../hmm.py#L203) (returned log-likelihood) |

### Initialization

| Part | hmm.tex | Code |
|---|---|---|
| Copy-number grid `g(a, b)` | Sec. 3.1, eq. (6) ([L105](hmm.tex#L105)) | [`cn_grid`](../hmm.py#L58) |
| Constrained k-means on the grid | Sec. 3.2, eq. (7) ([L118](hmm.tex#L118)) | [`constrained_kmeans`](../hmm.py#L100), [`_snap_centers`](../hmm.py#L87) |
| Initial means and tied variances | Sec. 3.3, eq. (8) ([L139](hmm.tex#L139)) | [`BinCellHMM._init_params`](../hmm.py#L354) |

### Baum-Welch

| Part | hmm.tex | Code |
|---|---|---|
| Scaled forward-backward, posteriors, log-likelihood | Sec. 4.1, eqs. (9)-(11) ([L153](hmm.tex#L153)) | [`forward_backward`](../hmm.py#L172) |
| Sufficient statistics | Sec. 4.2 ([L176](hmm.tex#L176)) | [`BinCellHMM._e_step`](../hmm.py#L380) |
| Mean and floored variance update | Sec. 4.2, eq. (14) ([L193](hmm.tex#L193)) | [`BinCellHMM._m_step`](../hmm.py#L397) |
| Stopping rule | Sec. 4.3 ([L200](hmm.tex#L200)) | [`BinCellHMM.fit`](../hmm.py#L405) |

### Decoding and labels

| Part | hmm.tex | Code |
|---|---|---|
| Viterbi path | Sec. 5 ([L206](hmm.tex#L206)) | [`viterbi`](../hmm.py#L207), [`BinCellHMM.predict`](../hmm.py#L440); posteriors [`predict_proba`](../hmm.py#L460) |
| `members` (T x N) | Sec. 5 | [`hmmclustering`](../hmm.py#L491) |
| Grid labels (nearest grid point) | Sec. 6.1 ([L220](hmm.tex#L220)) | [`BinCellHMM.state_cn`](../hmm.py#L482) |
| Per-cell clusters `C_k,e` and summaries | Sec. 6.2 ([L229](hmm.tex#L229)) | [`cell_clusters`](../hmm_caller.py#L83); CHISEL [`Caller.est_clusters`](../../src/chisel/Caller.py#L172) |
| Base cluster | Sec. 6.2, base cluster ([L243](hmm.tex#L243)) | CHISEL [`Caller.identify_base`](../../src/chisel/Caller.py#L322), [`Combiner.EM`](../../src/chisel/Combiner.py#L326) |
| Scale per base ploidy | Sec. 6.2, scale ([L257](hmm.tex#L257)) | CHISEL [`Caller.calling`](../../src/chisel/Caller.py#L267) |
| Variance floors | Sec. 6.2, variance floors ([L261](hmm.tex#L261)) | CHISEL [`Caller.calling`](../../src/chisel/Caller.py#L276-L285) |
| State of each cluster | Sec. 6.2, state of each cluster ([L271](hmm.tex#L271)) | CHISEL [`Caller.calling`](../../src/chisel/Caller.py#L287-L304) |
| Base ploidy by BIC | Sec. 6.2, eq. (22) ([L284](hmm.tex#L284)) | CHISEL [`Caller.calling`](../../src/chisel/Caller.py#L306-L317) |
| Calling every cell | Sec. 6.2 | [`call_cells`](../hmm_caller.py#L137): `Pool` of CHISEL [`Caller.init`](../../src/chisel/Caller.py#L209) + [`Caller.calling`](../../src/chisel/Caller.py#L259) |
| Orientation and phasing | Sec. 6.2, orientation and phasing ([L299](hmm.tex#L299)) | [`orient`](../hmm_caller.py#L166) (rule of CHISEL [`Caller.infer`](../../src/chisel/Caller.py#L348)); CHISEL [`Caller.phase`](../../src/chisel/Caller.py#L354), [`Caller.phasing`](../../src/chisel/Caller.py#L370) |

### Interface and parameters

| Part | hmm.tex | Code |
|---|---|---|
| `hmmclustering` vs `kclustering` | Sec. 7 ([L307](hmm.tex#L307)) | [`hmmclustering`](../hmm.py#L491); CHISEL [`Clusterizer.kclustering`](../../src/chisel/Clusterizer.py#L19) |
| CHISEL Caller with the HMM | Sec. 6.2, 7 | [`hmm_caller.main`](../hmm_caller.py#L188) |
| Hyper-parameters and defaults | Sec. 8 ([L322](hmm.tex#L322)) | [`BinCellHMM.__init__`](../hmm.py#L307), [`hmm_caller.parse_args`](../hmm_caller.py#L63); CHISEL `Caller.parse_args` |
| Complexity | Sec. 9 ([L348](hmm.tex#L348)) | [`forward_backward`](../hmm.py#L172), [`viterbi`](../hmm.py#L207) |
| Limitations | Sec. 10 ([L355](hmm.tex#L355)) | - |
