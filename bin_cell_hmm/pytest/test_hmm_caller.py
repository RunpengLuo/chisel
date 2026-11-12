import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from simulation import SimConfig, simulate, write_combo

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
PY27 = Path(
    os.environ.get(
        "CHISEL_PY27", Path.home() / ".local/miniconda3/envs/chisel-py27/bin/python"
    )
)


@pytest.mark.skipif(
    not PY27.exists(), reason="CHISEL python2.7 env not found (set CHISEL_PY27)"
)
def test_hmm_caller_output_feeds_cloner(tmp_path):
    sim = simulate(SimConfig(n_cells=20, n_chroms=2, bins_per_chrom=20, seed=3))
    write_combo(sim, tmp_path / "combo.tsv")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    cmd = [
        str(PY27),
        str(PKG / "hmm_caller.py"),
        str(tmp_path / "combo.tsv"),
        "--hmm-states",
        "4",
    ]
    cmd += ["--hmm-npz", str(tmp_path / "hmm.npz"), "-j", "2", "--seed", "0"]
    with open(tmp_path / "calls.tsv", "w") as out:
        result = subprocess.run(
            cmd, stdout=out, stderr=subprocess.PIPE, text=True, env=env
        )
    assert result.returncode == 0, result.stderr
    rows = [line.rstrip("\n").split("\t") for line in open(tmp_path / "calls.tsv")]
    assert rows[0][-2:] == ["CLUSTER", "CN_STATE"]
    assert len(rows) == 1 + 20 * 40 and all(len(r) == 12 for r in rows)
    assert all(len(r[11].split("|")) == 2 for r in rows[1:])
    model = np.load(tmp_path / "hmm.npz")
    assert model["members"].shape == (40, 20) and model["ploidy"].shape == (20,)

    shutil.copy(tmp_path / "calls.tsv", tmp_path / "cloned.tsv")
    cloner = [
        str(PY27),
        str(REPO / "src" / "chisel" / "Cloner.py"),
        str(tmp_path / "cloned.tsv"),
        "-s",
        "2",
    ]
    result = subprocess.run(cloner, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("#CELL\tCLUSTER\tCLONE")
