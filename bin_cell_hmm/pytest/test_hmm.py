import itertools
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

import hmm

PKG = Path(__file__).resolve().parents[1]
PY27 = Path(
    os.environ.get(
        "CHISEL_PY27", Path.home() / ".local/miniconda3/envs/chisel-py27/bin/python"
    )
)


def _brute_force(log_emit, startprob, transmat):
    N, T, K = log_emit.shape
    paths = np.array(list(itertools.product(range(K), repeat=T)))
    logp = np.log(startprob)[paths[:, 0]][None] + log_emit[:, np.arange(T), paths].sum(
        -1
    )
    logp = logp + np.log(transmat)[paths[:, :-1], paths[:, 1:]].sum(-1)[None]
    loglik = logsumexp(logp, axis=1)
    weight = np.exp(logp - loglik[:, None])
    gamma = np.stack(
        [[weight[:, paths[:, t] == k].sum(1) for k in range(K)] for t in range(T)]
    ).transpose(2, 0, 1)
    return loglik, gamma, paths[logp.argmax(1)], logp.max(1)


def _sample_hmm(n_cells, chroms, means, sd, stay, rng):
    K = len(means)
    transmat = hmm.make_transmat(stay, K)
    z = np.empty((n_cells, len(chroms)), dtype=int)
    for t in range(len(chroms)):
        if t == 0 or chroms[t] != chroms[t - 1]:
            z[:, t] = rng.integers(K, size=n_cells)
        else:
            u = rng.random(n_cells)[:, None]
            z[:, t] = (u > np.cumsum(transmat[z[:, t - 1]], axis=1)).sum(1)
    X = means[z] + rng.normal(0.0, sd, size=z.shape + (2,))
    return z, X[..., 0], X[..., 1]


def test_make_transmat_rows_and_entries():
    A = hmm.make_transmat(0.9, 4)
    np.testing.assert_allclose(A.sum(1), 1.0)
    np.testing.assert_allclose(np.diag(A), 0.9)
    np.testing.assert_allclose(A[0, 1], 0.1 / 3)
    np.testing.assert_array_equal(hmm.make_transmat(0.9, 1), [[1.0]])


def test_cn_grid_mirrored_and_phased():
    states, means = hmm.cn_grid(4, base=2.0, mirrored=True)
    assert (states[:, 0] >= states[:, 1]).all()
    assert (means[:, 1] <= 0.5).all()
    np.testing.assert_allclose(means[(states == (0, 0)).all(1)], [[0.0, 0.5]])
    np.testing.assert_allclose(means[(states == (2, 1)).all(1)], [[1.5, 1 / 3]])
    full, _ = hmm.cn_grid(4, mirrored=False)
    assert len(full) == sum(t + 1 for t in range(5))


def test_forward_backward_and_viterbi_match_brute_force():
    rng = np.random.default_rng(1)
    log_emit = rng.normal(size=(3, 5, 3)) * 2
    startprob = np.array([0.2, 0.5, 0.3])
    transmat = hmm.make_transmat(0.7, 3)
    loglik, gamma, best_path, best_logp = _brute_force(log_emit, startprob, transmat)
    g, ll = hmm.forward_backward(log_emit, startprob, transmat)
    np.testing.assert_allclose(ll, loglik, rtol=1e-10)
    np.testing.assert_allclose(g, gamma, atol=1e-10)
    path, logp = hmm.viterbi(log_emit, startprob, transmat)
    np.testing.assert_array_equal(path, best_path)
    np.testing.assert_allclose(logp, best_logp, rtol=1e-10)


def test_log_emission_marginalizes_masked_dimension():
    X = np.array([[1.2, np.nan]])
    mask = np.isfinite(X)
    means = np.array([[1.0, 0.5], [2.0, 0.25]])
    var = np.array([[0.04, 0.01], [0.09, 0.02]])
    expected = -0.5 * (
        np.log(2 * np.pi * var[:, 0]) + (1.2 - means[:, 0]) ** 2 / var[:, 0]
    )
    np.testing.assert_allclose(hmm.log_emission(X, mask, means, var)[0], expected)


def test_constrained_kmeans_recovers_grid_points():
    rng = np.random.default_rng(2)
    _, grid = hmm.cn_grid(5)
    true_idx = np.array([1, 4, 7])
    X = np.concatenate([grid[i] + rng.normal(0, 0.02, size=(200, 2)) for i in true_idx])
    idx, labels, _ = hmm.constrained_kmeans(X, grid, 3, restarts=5, rng=0)
    assert set(idx) == set(true_idx)
    assert len(np.unique(labels)) == 3
    with pytest.raises(ValueError):
        hmm.constrained_kmeans(X, grid, len(grid) + 1)


def test_chrom_blocks_rejects_split_chromosome():
    assert hmm.chrom_blocks(["1", "1", "2"]) == [slice(0, 2), slice(2, 3)]
    with pytest.raises(ValueError):
        hmm.chrom_blocks(["1", "2", "1"])


def test_fit_is_monotone_and_recovers_states():
    rng = np.random.default_rng(3)
    states, grid = hmm.cn_grid(4)
    true_idx = [
        int(np.flatnonzero((states == s).all(1))[0])
        for s in [(1, 1), (1, 0), (2, 1), (2, 2)]
    ]
    chroms = np.repeat(["chr1", "chr2", "chr3"], 40)
    z, rdr, baf = _sample_hmm(30, chroms, grid[true_idx], 0.06, 0.95, rng)
    model = hmm.BinCellHMM(4, stay=0.95, maxcn=4, seed=0).fit(rdr, baf, chroms)
    assert np.all(
        np.diff(model.loglik_history_) > -1e-6 * np.abs(model.loglik_history_[1:])
    )
    cn = model.state_cn()[model.predict(rdr, baf, chroms)]
    assert np.mean((cn == states[true_idx][z]).all(-1)) > 0.97


def test_hmmclustering_uses_chisel_layout():
    rng = np.random.default_rng(4)
    chroms = np.repeat(["chr1", "chr2"], 20)
    _, grid = hmm.cn_grid(4)
    _, rdr, baf = _sample_hmm(6, chroms, grid[[1, 2, 4]], 0.05, 0.9, rng)
    data = np.empty((rdr.shape[1], 2 * rdr.shape[0]))
    data[:, 0::2], data[:, 1::2] = rdr.T, baf.T
    members, model = hmm.hmmclustering(
        data, chroms, 3, seed=0, verbose=False, return_model=True
    )
    assert members.shape == (40, 6)
    np.testing.assert_array_equal(members, model.predict(rdr, baf, chroms).T)


@pytest.mark.skipif(
    not PY27.exists(), reason="CHISEL python2.7 env not found (set CHISEL_PY27)"
)
def test_hmm_python27_matches_python3(tmp_path):
    rng = np.random.default_rng(5)
    chroms = ["c1"] * 15 + ["c2"] * 15
    _, grid = hmm.cn_grid(4)
    _, rdr, baf = _sample_hmm(5, chroms, grid[[1, 2, 4]], 0.05, 0.9, rng)
    data = np.empty((30, 10))
    data[:, 0::2], data[:, 1::2] = rdr.T, baf.T
    np.save(tmp_path / "data.npy", data)
    code = (
        "import sys, numpy as np, hmm; d = np.load(sys.argv[1]); "
        "m = hmm.hmmclustering(d, ['c1'] * 15 + ['c2'] * 15, 3, seed=0, verbose=False); "
        "np.save(sys.argv[2], m); np.save(sys.argv[3], hmm.cn_grid(6, mirrored=False)[1])"
    )
    paths = [tmp_path / f for f in ("data.npy", "members27.npy", "grid27.npy")]
    result = subprocess.run(
        [str(PY27), "-c", code] + [str(p) for p in paths],
        cwd=PKG,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    np.testing.assert_allclose(np.load(paths[2]), hmm.cn_grid(6, mirrored=False)[1])
    members = hmm.hmmclustering(data, chroms, 3, seed=0, verbose=False)
    np.testing.assert_array_equal(np.load(paths[1]), members)
