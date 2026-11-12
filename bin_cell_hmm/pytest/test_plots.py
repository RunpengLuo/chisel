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
