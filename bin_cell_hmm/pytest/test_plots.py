import numpy as np
import pytest

pytest.importorskip("cnplot")

from plots import bins_table, encode_states, plot_state_heatmaps  # noqa: E402


def test_encode_states_is_shared_and_phase_free():
    a1, b1 = np.array([[1, 2, 0]]), np.array([[1, 1, 3]])
    a2, b2 = np.array([[1, 1, 9]]), np.array([[2, 1, 0]])
    codes, cmap, norm, states = encode_states([(a1, b1), (a2, b2)], "cnamaste")
    assert states[: len(states) - 1] == [(1, 1), (2, 1), (3, 0)] and states[-1] == (
        9,
        0,
    )
    assert codes[0][0, 1] == codes[1][0, 0] == states.index((2, 1))
    assert cmap.N == len(states)
    np.testing.assert_array_equal(norm(np.arange(len(states))), np.arange(len(states)))


def test_plot_state_heatmaps_writes_png(tmp_path):
    chroms = np.repeat(["chr1", "chr2"], 10)
    starts = np.tile(np.arange(10) * 1_000_000, 2)
    a = np.ones((6, 20), dtype=int)
    b = np.ones((6, 20), dtype=int)
    a[3:, 5:12] = 2
    path = tmp_path / "ab.png"
    plot_state_heatmaps(
        [("truth", a, b), ("call", a, b * 0)],
        bins_table(chroms, starts, starts + 1_000_000),
        np.array(["normal"] * 3 + ["clone1"] * 3),
        path,
    )
    assert path.stat().st_size > 0


def test_expected_landmarks_holds_only_the_called_states():
    from plots import expected_landmarks

    a = np.array([[1, 1, 2, 2], [1, 1, 1, 1]])
    b = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
    lm = expected_landmarks(a, b).set_index("cn_state")
    assert sorted(lm.index) == ["1|1", "2|1"]
    np.testing.assert_allclose(lm.loc["2|1", "exp_BAF"], 1 / 3)
    # NB: cell 0 has mean total 2.5: (2,1) at 3 / 2.5, (1,1) at 2 / 2.5 there and 1 in cell 1
    np.testing.assert_allclose(lm.loc["2|1", "exp_RDR"], 1.2)
    np.testing.assert_allclose(lm.loc["1|1", "exp_RDR"], 1.0)
    assert bool(lm.loc["1|1", "is_balanced"]) and not bool(lm.loc["2|1", "is_balanced"])


def test_plot_rdr_baf_scatters_writes_one_page_per_coloring(tmp_path):
    from plots import plot_rdr_baf_scatters

    rng = np.random.default_rng(1)
    a = np.ones((8, 30), dtype=int)
    b = np.ones((8, 30), dtype=int)
    a[4:, 10:20] = 2
    rdr = (a + b) / 2 + rng.normal(0, 0.05, a.shape)
    baf = np.clip(b / (a + b) + rng.normal(0, 0.05, a.shape), 0, 1)
    baf[0, 0] = np.nan
    path = tmp_path / "scatter.pdf"
    plot_rdr_baf_scatters(rdr, baf, [("truth", a, b), ("call", b, a)], path)
    data = path.read_bytes()
    assert data.count(b"/Type /Page") - data.count(b"/Type /Pages") == 2
