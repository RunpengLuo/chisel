# Simulation

[`simulate`](../simulation.py#L118) draws one dataset from a clonal haplotype-specific copy-number model. Its parameters and defaults are the fields of [`SimConfig`](../simulation.py#L41); its output is a [`SimData`](../simulation.py#L69).

## Copy numbers

- Genome: `n_chroms` chromosomes of `bins_per_chrom` bins of `bin_size` bp, named `chr1`, `chr2`, ... ([L134-L135](../simulation.py#L134-L135)).
- Event states: every `(a, b)` with `1 <= a + b <= max_cn` except `(1, 1)` ([`_event_states`](../simulation.py#L88)).
- One event ([`_add_events`](../simulation.py#L97)):
  - a length, uniform in `event_bins`, capped at the chromosome ([L100](../simulation.py#L100));
  - a chromosome, uniform, and a start, uniform where the event fits ([L104-L105](../simulation.py#L104-L105));
  - with `overlap` False, the chromosome and start are drawn again until no earlier event of the lineage covers those bins ([L106](../simulation.py#L106), [L114](../simulation.py#L114));
  - a state, uniform among the event states, overwriting the bins ([L112](../simulation.py#L112)).
- Clone 0 is normal: `(1, 1)` in every bin ([L138](../simulation.py#L138)).
- Ancestor: the normal profile plus `n_clonal_events` events ([L139-L142](../simulation.py#L139-L142)).
- Tumor clones `1..n_clones`: the ancestor plus `n_subclonal_events` events each, a star phylogeny ([L143-L155](../simulation.py#L143-L155)).
  - The last `n_wgd_clones` clones double the ancestor first, `(a, b) -> (2a, 2b)`, i.e. whole-genome doubling ([L146](../simulation.py#L146)).
- Cells: `round(normal_frac * n_cells)` normal cells; every other cell picks a tumor clone uniformly ([L156-L162](../simulation.py#L156-L162)).
- Each cell copies its clone's profile and adds `Poisson(cell_event_rate)` events of `cell_event_bins` bins, overlap allowed ([L163-L175](../simulation.py#L163-L175)).
- `cell_event` marks the (bin, cell) entries that differ from the cell's clone ([L177-L178](../simulation.py#L177-L178)).

## Read counts and RDR

- Relative copy number `rel = C / mean_b(C)` per cell, `C = a + b` ([L180-L181](../simulation.py#L180-L181)).
- Cell depth `d_e ~ LogNormal(0, depth_sd)` ([L182](../simulation.py#L182)).
- Bin bias `g_b ~ LogNormal(0, bin_bias_sd)`, shared by the normal and the cells ([L183](../simulation.py#L183)).
- Normal count `~ Poisson(normal_reads_per_bin * g_b)` ([L184](../simulation.py#L184)).
- Cell count `~ Poisson(reads_per_bin * d_e * g_b * rel * G)`, with `G ~ Gamma(1/read_dispersion, read_dispersion)` (mean 1) when `read_dispersion > 0` ([L185-L189](../simulation.py#L185-L189)).
- RDR `= (count / normal) * (sum_b normal / sum_b count)`, as CHISEL [`RDREstimator`](../../src/chisel/RDREstimator.py#L126-L127) computes it; 0 where the normal count is 0 ([L190-L191](../simulation.py#L190-L191)).

## Allele counts and BAF

- SNP-covering reads `n ~ Poisson(snp_reads_per_bin * d_e * rel)` ([L193](../simulation.py#L193)).
- `B ~ Binomial(n, (1 - seq_error) * b/C + seq_error * a/C)`, and `A = n - B` ([L194-L198](../simulation.py#L194-L198)).
- BAF `= B / (A + B)`, 0.5 where `A + B = 0`, as CHISEL [`Combiner`](../../src/chisel/Combiner.py#L149) writes it ([L199](../simulation.py#L199)).
- Haplotype B is the same haplotype in every bin, i.e. phasing is error-free. Counts are simulated per bin, so CHISEL's SNP-level steps (`BAFEstimator`, the block phasing of `Combiner`) are skipped.

## Benchmark classes

- [`SIM_CLASSES`](../simulation.py#L219): (shared CNAs, private CNAs per tumor clone, CNA length), (1,2), (3,3) and (6,3) CNAs at 10, 30 and 50 Mb each.
- [`class_config`](../simulation.py#L252): one replicate of a class. It sets `n_clonal_events` to the shared CNAs, `n_subclonal_events` to the private ones and `event_bins` to the class length in bins, with `overlap` False; every other field keeps its default unless overridden.
- [`sample_id`](../simulation.py#L233): `numcnas<shared>.<private>_cnasize<length>_ploidy2_random<replicate>`, e.g. `numcnas1.2_cnasize1e7_ploidy2_random0`.
- Seed: the CRC32 of the sample id ([L280](../simulation.py#L280)), so replicates of a class are independent draws and each is reproducible on its own.

## Output

- [`write_combo`](../simulation.py#L291): CHISEL `combo.tsv`, `CHR START END CELL NORMAL COUNT RDR A_COUNT B_COUNT BAF`, the input of `Caller.py` and `hmm_caller.py`.
- [`write_truth`](../simulation.py#L311): `truth.tsv`, `CHR START END CELL CLONE A B CELL_EVENT`.
- Cells are named `cell0000`, `cell0001`, ..., so name order is index order, as CHISEL sorts cells by name.
