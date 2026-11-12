import numpy as np
import pandas as pd

from plot_simulation import (
    get_class_medians,
    get_complement,
    group_order,
    plot_simulation_metrics,
    sim_groups,
)


def _table():
    rng = np.random.default_rng(0)
    rows = []
    for size in ("1e7", "5e7"):
        for rep in range(3):
            sid = f"numcnas1.2_cnasize{size}_ploidy2_random{rep}"
            rows.append({"sample_id": sid, "method": "truth", "bp_cn_rate": 0.05})
            for method in ("chisel_cloner_bic", "hmm_chisel_cloner_bic"):
                rows.append(
                    {
                        "sample_id": sid,
                        "method": method,
                        "exact_altered": rng.uniform(0.5, 1),
                        "specificity": 1.0 if rep == 0 else rng.uniform(0.95, 1),
                        "bp_cn_rate": rng.uniform(0.03, 0.2),
                        "clone_ari": rng.uniform(0.5, 1),
                    }
                )
    return pd.DataFrame(rows)


def test_sim_groups_and_order():
    groups = sim_groups(
        [
            "numcnas1.2_cnasize1e7_ploidy2_random0",
            "numcnas6.3_cnasize5e7_ploidy2_random1",
            "bad",
        ]
    )
    assert list(groups["label"]) == ["(1,2, 10M)", "(6,3, 50M)"]
    assert group_order(groups) == [
        "numcnas6.3_cnasize5e7_ploidy2",
        "numcnas1.2_cnasize1e7_ploidy2",
    ]


def test_get_complement_puts_specificity_one_at_the_floor():
    complement, floor = get_complement(pd.Series([1.0, 0.99, 0.9, np.nan]))
    assert floor == 1e-3
    np.testing.assert_allclose(complement[:3], [1e-3, 0.01, 0.1])
    assert np.isnan(complement[3])


def test_plot_and_medians(tmp_path):
    table = _table()
    medians = get_class_medians(table)
    assert len(medians) == 4 and set(medians["#replicates"]) == {3}
    figure = plot_simulation_metrics(table)
    assert len(figure.axes) == 4
    figure.savefig(tmp_path / "page.png")
    assert (tmp_path / "page.png").stat().st_size > 0
