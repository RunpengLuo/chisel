# bin_cell_hmm: local-global clustering of (bin, cell) copy-number states

```sh
# clone this branch
git clone -b feat/bin_cell_hmm https://github.com/RunpengLuo/chisel.git chisel && cd chisel

# CHISEL's runtime, Python 2.7 (Apple Silicon: osx-64 under Rosetta; on Linux or Intel macOS drop CONDA_SUBDIR)
CONDA_SUBDIR=osx-64 mamba create -y -n chisel-py27 -c conda-forge "python=2.7" \
  "numpy>=1.16.1" "scipy>=1.2.1" "statsmodels<=0.10.1" pandas matplotlib "seaborn>=0.7.1"

# the HMM, simulation, metrics and plots, Python 3, with cnplot
mamba create -y -n bin-cell-hmm -c conda-forge "python>=3.10" numpy scipy pandas scikit-learn matplotlib pytest
conda activate bin-cell-hmm
pip install git+https://github.com/raphael-group/cnplot.git
export CHISEL_PY27=$(conda run -n chisel-py27 python -c "import sys; print(sys.executable)")

# unit tests, the Python 2.7 ones through $CHISEL_PY27
python -m pytest bin_cell_hmm/pytest

# one simulated sample: CHISEL vs the HMM, metrics and figures -> bin_cell_hmm/test/out/sim_seed0/
python bin_cell_hmm/test/compare_chisel.py --seed 0 --jobs 8

# benchmark, 9 classes x 2 replicates; metric page -> bin_cell_hmm/test/out/benchmark/benchmark_metrics.pdf
python bin_cell_hmm/test/run_benchmark.py --replicates 2 --workers 4 --jobs 4

# documentation: bin_cell_hmm/docs/ (hmm.md, interface.md, benchmark.md, simulation.md, hmm.tex)
```
