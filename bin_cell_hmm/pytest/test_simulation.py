import numpy as np

from simulation import SimConfig, simulate, write_combo, write_truth

CFG = SimConfig(n_cells=12, n_chroms=2, bins_per_chrom=15, cell_event_rate=1.0, seed=5)


def test_simulate_is_consistent():
    sim = simulate(CFG)
    assert sim.a.shape == (12, 30) and sim.chroms.shape == (30,)
    assert ((sim.a + sim.b) >= 1).all()
    normal = sim.clone == 0
    diploid = (sim.a[normal] == 1) & (sim.b[normal] == 1)
    assert diploid[~sim.cell_event[normal]].all()
    expected = (
        sim.count
        / sim.normal
        * (sim.normal.sum() / sim.count.sum(axis=1, keepdims=True))
    )
    np.testing.assert_allclose(sim.rdr, expected)
    n = sim.a_count + sim.b_count
    np.testing.assert_allclose(sim.baf[n > 0], sim.b_count[n > 0] / n[n > 0])
    np.testing.assert_array_equal(simulate(CFG).count, sim.count)


def test_write_combo_and_truth_round_trip(tmp_path):
    sim = simulate(CFG)
    write_combo(sim, tmp_path / "combo.tsv")
    write_truth(sim, tmp_path / "truth.tsv")
    rows = [
        line.split("\t")
        for line in open(tmp_path / "combo.tsv")
        if not line.startswith("#")
    ]
    assert len(rows) == 12 * 30 and all(len(r) == 10 for r in rows)
    t, e = 7, 3
    r = rows[t * 12 + e]
    assert (r[0], int(r[1]), r[3]) == (sim.chroms[t], sim.starts[t], sim.cells[e])
    assert int(r[5]) == sim.count[e, t] and int(r[8]) == sim.b_count[e, t]
    np.testing.assert_allclose(float(r[6]), sim.rdr[e, t], rtol=1e-9)
    truth = [
        line.split("\t")
        for line in open(tmp_path / "truth.tsv")
        if not line.startswith("#")
    ]
    assert (int(truth[t * 12 + e][5]), int(truth[t * 12 + e][6])) == (
        sim.a[e, t],
        sim.b[e, t],
    )


def test_wgd_clone_doubles_the_ancestor():
    base = SimConfig(n_cells=40, n_subclonal_events=0, cell_event_rate=0.0, seed=7)
    sim = simulate(base)
    wgd = simulate(SimConfig(**{**base.__dict__, "n_wgd_clones": 1}))
    last = wgd.clone == base.n_clones
    np.testing.assert_array_equal(wgd.a[last], 2 * sim.a[last])
    np.testing.assert_array_equal(wgd.b[last], 2 * sim.b[last])
    other = (wgd.clone > 0) & ~last
    np.testing.assert_array_equal(wgd.a[other], sim.a[other])


def test_sample_id_and_class_config():
    from simulation import SIM_CLASSES, class_config, sample_id

    assert sample_id(1, 2, 10_000_000, 0) == "numcnas1.2_cnasize1e7_ploidy2_random0"
    assert sample_id(6, 3, 50_000_000, 9) == "numcnas6.3_cnasize5e7_ploidy2_random9"
    assert len(SIM_CLASSES) == 9
    cfg = class_config(3, 3, 30_000_000, 2)
    assert (cfg.n_clonal_events, cfg.n_subclonal_events, cfg.event_bins) == (
        3,
        3,
        (6, 6),
    )
    assert (
        cfg == class_config(3, 3, 30_000_000, 2)
        and cfg.seed != class_config(3, 3, 30_000_000, 3).seed
    )


def test_class_events_do_not_overlap_within_a_lineage():
    from simulation import class_config

    cfg = class_config(6, 3, 50_000_000, 0, n_cells=30, cell_event_rate=0.0)
    sim = simulate(cfg)
    for c in range(1, cfg.n_clones + 1):
        cells = sim.clone == c
        altered = (sim.a[cells][0] != 1) | (sim.b[cells][0] != 1)
        assert altered.sum() == (cfg.n_clonal_events + cfg.n_subclonal_events) * 10
