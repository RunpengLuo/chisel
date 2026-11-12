import numpy as np

from metrics import breakpoints, cna_state, phase_free, supervised_metrics


def test_phase_free_and_cna_state():
    assert phase_free(2, 1) == phase_free(1, 2) == 1002
    states = cna_state([1, 2, 0, 2, 2, 3, 1, 0], [1, 2, 0, 0, 1, 1, 0, 1])
    assert list(states) == ["neu", "bamp", "bdel", "cnloh", "amp", "amp", "del", "del"]


def test_supervised_metrics_counts():
    true_a = np.array([[1, 1, 2, 2, 1, 3]])
    true_b = np.array([[1, 1, 1, 0, 0, 1]])
    call_a = np.array([[1, 2, 1, 1, 1, 2]])
    call_b = np.array([[1, 0, 2, 1, 0, 1]])
    m = supervised_metrics(true_a, true_b, call_a, call_b)
    # NB: 4 truly altered (2|1, 2|0, 1|0, 3|1): one called 1|1, two exact (1|2, 1|0), three same state
    assert m["sensitivity"] == 1 - 1 / 4
    assert m["specificity"] == 1 - 1 / 2
    assert m["exact_altered"] == 2 / 4
    assert m["state_altered"] == 3 / 4


def test_supervised_metrics_nan_without_altered():
    ones = np.ones((2, 3), dtype=int)
    m = supervised_metrics(ones, ones, ones, ones)
    assert np.isnan(m["sensitivity"]) and m["specificity"] == 1.0


def test_breakpoints_skip_chromosome_boundaries():
    labels = np.array([[0, 0, 1, 2, 2], [0, 1, 1, 1, 1]])
    chroms = np.array(["c1", "c1", "c1", "c2", "c2"])
    assert breakpoints(labels, chroms) == (2, 2 * 3)
