# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CHISEL infers allele- and haplotype-specific copy numbers in single cells from low-coverage scDNA-seq (Zaccaria & Raphael, Nat Biotechnol 2021, https://doi.org/10.1038/s41587-020-0661-6). Package version is in `setup.py` (`version='1.2'`).

> [!IMPORTANT]
> The code is **Python 2.7 only** (`python_requires='==2.7.*'`). It uses print statements, `xrange`, `unichr`, tuple-unpacking lambdas (`lambda (e, cns, p): ...`) and implicit relative imports (`from Utils import *`). Ruff cannot parse these files (it reports syntax errors), so do not run `ruff format`/`ruff check` on the existing modules or "modernize" them in passing.

## Environment and commands

External tools must be on `PATH`: `samtools`, `bcftools`, `awk`. `chisel_prep` also needs `bwa`. The `*nonormal*` and `chisel_rdr` commands need `art_illumina` and `bwa`, plus a `bwa index` of the reference, to simulate and align a normal sample. CI uses samtools/bcftools 1.7. The bin drivers launch each step as `python2.7 <src/chisel/Step.py>`, so a `python2.7` executable must also be on `PATH`.

```sh
# install (bioconda package, from README)
conda create -n chisel chisel
# install from source for development (extras_require 'dev' = pytest, mock)
pip install -e ".[dev]"

# all unit tests (what tox / CI runs)
pytest tests/pytests
# single test
pytest tests/pytests/test_call.py::test_call

# end-to-end regression: runs every demo, diffs stdout/PNGs/TSVs against
# tests/*.chk and tarballs from github.com/raphael-group/chisel-data
bash tests/allchecks.sh
```

Unit-test data:
- `test_call`, `test_combine` and `test_clone` run offline on the fixtures in `tests/pytests/data/output/` (`rdr.tsv`, `baf.tsv`, `combo.tsv`, `calls.tsv`, `total.tsv`).
- `test_rdr` and `test_baf` need `normal.bam`, `cells.bam`, `phases.tsv` and an indexed `hg19.fa` (+ `.fai`, `.dict`) in `$TEST_DIRECTORY` (default `tests/pytests/data/input/`, gitignored). Fetch them with `python3 -m zenodo_get 10.5281/zenodo.3950299`; see `.github/workflows/main.yml` for the full CI setup, including the hg19 download.
- Every test compares the **md5 hash** of the step's whole stdout output, run with `-j 1` and a fixed seed. Any intended change to the output of `Combiner`, `Caller` or `Cloner` (values, column order, float formatting, row order) breaks the hash. Confirm the change is intended, then update the expected hash in the test.

## Architecture

### Two layers: bin drivers -> step modules via subprocess

- `src/chisel/bin/chisel_*.py` are the console entry points registered in `setup.py`. They validate arguments, create the run-directory layout, and then **shell out** to the step modules with `runcmd(cmd, outdir, out=...)` (`Utils.py`). They do not import the step logic.
- `runcmd` writes the child's **stdout** to the step's output TSV and tees **stderr** to the terminal and to `<outdir>/log`, with progress-bar lines and ANSI codes stripped. A step module must therefore print only data to stdout; all logging (`Utils.log`, `ProgressBar`) goes to stderr.
- Each step module in `src/chisel/` (`RDREstimator`, `BAFEstimator`, `Combiner`, `Caller`, `Cloner`) exposes `main(args=None, stdout_file=None)`. Called with an argv list it can run in-process (this is what the pytests do), and `stdout_file` redirects the output table to a file. `Plotter.py` is the exception: `parse_args()` reads `sys.argv`, and it writes PNGs into the cwd (drivers `chdir` into `plots/` first).

### Pipeline and run-directory layout

`chisel` runs these steps; each subfolder of `--rundir` holds one output plus `log`:

| Step | Module | Output |
|------|--------|--------|
| Read counts per bin (samtools + inline awk) and cell selection by `--minreads` | `RDREstimator.py` | `rdr/rdr.tsv`, `rdr/total.tsv` |
| Germline het-SNP allele counts per cell | `BAFEstimator.py` | `baf/baf.tsv` |
| Join RDR + BAF per bin | `Combiner.py` | `combo/combo.tsv` |
| Global clustering, ploidy selection, copy-number calls | `Caller.py` (+ `Clusterizer.py`) | `calls/calls.tsv` |
| Cell clustering into clones | `Cloner.py` | `clones/mapping.tsv`, and **rewrites `calls/calls.tsv` in place** |
| Plots | `Plotter.py` | `plots/*.png` |

Driver variants are compositions of these steps:
- `chisel_preprocess` / `chisel_rdr` stop after RDR/BAF.
- `chisel_combocall` aggregates the `rdr/` and `baf/` folders of several preprocessing runs, then runs Combiner -> Caller -> Cloner -> Plotter.
- `chisel_calling` / `chisel_cloning` / `chisel_plotting` restart from an intermediate file.
- The `*nonormal*` variants first simulate a matched normal with `art` (`simulate_sequencing`, written to `data/simulatednormal.bam`) and pass `--minimumsnps/--missingsnps` to Combiner.
- When adding a flag that must reach a step module, update every driver that builds that step's command line. The `chisel_*combocall` drivers have already drifted from the others at least once (see the recent commits adding `--reference`).

### TSV contracts (tab-separated, `#` header lines skipped by readers)

- `rdr.tsv`: `CHR START END CELL NORMAL_COUNT COUNT RDR` (7 cols; `Combiner.read_rdr` asserts this).
- `baf.tsv`: `CHR POS CELL A_COUNT B_COUNT`.
- `combo.tsv`: `CHR START END CELL NORMAL_COUNT COUNT RDR A_COUNT B_COUNT BAF`.
- `calls.tsv` (from `Caller`): the 10 combo columns + `CLUSTER CN_STATE`, where `CN_STATE` is `a|b`. After `Cloner` the header becomes `... CLUSTER HAP_CN CORRECTED_HAP_CN`. `doc/chisel.md` is stale here: it lists 11 columns with a dash-separated state.

### Algorithmic core

- **Combiner.combine** (per bin, parallel): groups SNPs into haplotype blocks of `--blocksize` (default 50kb). With phase correction on, it z-tests for phasing switch errors inside each block. It then estimates the bin's BAF with a binomial-mixture `EM` (also reused by `Caller.identify_base`). If `|beta - 0.5|` reaches a bootstrapped threshold, blocks are oriented so B is the minor allele; otherwise the bin is "balanced" and blocks are randomly flipped.
- **Clusterizer.kclustering**: clusters bins globally on the matrix `[RDR, mirroredBAF]` of every cell concatenated (rows = bins). It uses K-medians (`--lorder 1`, default) or K-means with k-means++-style seeding and `--restarts` random restarts. K is chosen by **binary search** between `LB` and `UB` on the normalized objective with tolerance `--threshold`.
- **Caller.calling** (per cell, parallel):
  1. `identify_base` finds the largest balanced cluster.
  2. For each even ploidy 2..`--maxploidy`, it scales RDR to total copy number.
  3. Each cluster gets the best `(A, B)` state under Gaussian log-likelihoods of per-bin RDR and BAF.
  4. The ploidy is selected by BIC, weighted by `--sensitivity` and filtered by `--maxhomodel`.

  States are assigned **per cluster**; there is no smoothing between adjacent bins along the genome.
- **Caller.phasing** (per chromosome): the only sequential pass along the genome. A 2-state DP (keep vs. swap the `(A, B)` order) minimizes the Hamming distance between consecutive bins summed over all cells, then backtracks.
- **Cloner**: hierarchical clustering of cells on haplotype-specific states (`--maxdiff`, `--linkage`), keeps clusters of at least `--minsize` cells, refines, and writes the per-clone consensus profile as `CORRECTED_HAP_CN`.

### Conventions in the code

- Parallelism uses `multiprocessing.Pool(initializer=init_*, initargs=...)`. The initializer stores shared inputs in **module-level globals**, and worker functions read those globals. Follow this pattern rather than passing large objects per job. Clusterizer shares numpy arrays through `mp.Array` buffers.
- Bin drivers set `OMP_NUM_THREADS` etc. to 1 before importing numpy so that BLAS threads do not oversubscribe the process pool.
- Reproducibility: `--seed` seeds numpy in each module's `parse_args`/`combo`. Per-job seeds are drawn from that RNG and reseeded inside workers (`combine`, `run_kclustering`).
- Bins are keyed as `(chrom, start, end)` tuples and ordered with `Utils.orderchrs` (numeric, then X=23, Y=24, M=25).
- `src/chisel/count.awk`, `src/chisel/bin.awk` and `src/chisel/bin/count.awk` are not read by any code. `BAFEstimator.mkcount` regenerates `count.awk` in the run directory at runtime, and `RDREstimator` builds its awk program inline. `Mutator.py` is not wired into any entry point.
- `Clusterizer.kclustering` calls `taskset`, which exists only on Linux. On macOS the call fails harmlessly.

## bin_cell_hmm/ (bin-cell HMM, Python 3 subproject)

A per-(bin, cell) alternative to `Clusterizer.kclustering`. `bin_cell_hmm/README.md` holds only install and run commands; the documentation is in `bin_cell_hmm/docs/` (`hmm.md`, `interface.md`, `benchmark.md`, `simulation.md`, `hmm.tex`).

- `hmm.py` is a swap-in for CHISEL, so it must stay importable under Python 2.7 / numpy 1.16. Don't use f-strings, `@`, type hints or `np.random.default_rng`. Use `np.true_divide` for integer arrays: `np.divide` floors them under Python 2 even with `from __future__ import division`. `pytest/test_hmm.py::test_hmm_python27_matches_python3` enforces this. `hmm_caller.py` (CHISEL's `Caller` with `hmmclustering`; it imports `Caller`) is Python 2.7 only. `simulation.py`, `metrics.py` and `test/*.py` are Python 3.
- CHISEL's runtime for the comparison is the conda env `chisel-py27` (`osx-64` under Rosetta on Apple Silicon; the create command is in `bin_cell_hmm/README.md`). Override its path with `CHISEL_PY27`.
- Run Python 2.7 with `PYTHONDONTWRITEBYTECODE=1`. Python 2 writes `.pyc` next to the source and validates it by mtime at 1-second resolution, so an edit made within the same second runs stale bytecode.
- Ruff applies to `bin_cell_hmm/` only.
- `docs/hmm.md` and `docs/simulation.md` link to line numbers of `hmm.py`, `hmm_caller.py`, `simulation.py`, `docs/hmm.tex` and CHISEL's `Caller.py`. Update the links when those files change. Build `docs/hmm.pdf` with `latexmk -pdf hmm.tex && latexmk -c` in `docs/`; the PDF and LaTeX intermediates are gitignored.

```sh
python -m pytest bin_cell_hmm/pytest
python bin_cell_hmm/test/compare_chisel.py --seed 0 --jobs 12   # outputs: bin_cell_hmm/test/out/ (gitignored)
```

## Git remotes

`origin` is the personal fork `RunpengLuo/chisel`; `upstream` is the canonical `raphael-group/chisel`. Push feature branches to `origin` only; they reach `upstream` through pull requests.
