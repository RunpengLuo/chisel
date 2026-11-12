"""Scores of per-(bin, cell) copy-number calls against a simulation's truth.

The scored unit is one bin of one cell. AB pairs are compared phase-free: (a, b) and (b, a) are equal,
except in ``exact_altered_hap``, which needs the true pair in its orientation (A first, B second, B
the simulation's haplotype B).

The supervised metrics follow cna_maste's simulation benchmark:
    - ``sensitivity``, ``specificity``, ``exact_altered`` (N-CNA recall): cna_maste
      ``python/cnamaste/qa/metrics.py``, there per spot and segment.
    - ``state_altered`` (CNA recall): cna_maste ``sandbox/luo/benchmark/scripts/metrics/supervised_metrics.py``
      ``get_state_metrics``, with the CNA states of ``cna_states.py``: Numbat's six ``cnv_state`` values
      (https://kharchenkolab.github.io/numbat/articles/descriptions.html), ``loh`` written ``cnloh``.

| state | definition | example |
|---|---|---|
| neu | (1, 1) | 1:1 |
| bamp | balanced, total > 2 | 2:2 |
| bdel | balanced, total < 2 | 0:0 |
| cnloh | unbalanced, total = 2 | 2:0 |
| amp | unbalanced, total > 2 | 2:1, 3:1 |
| del | unbalanced, total < 2 | 1:0 |

``breakpoints`` counts label changes between adjacent bins of the same chromosome in each cell.
"""

import numpy as np


def phase_free(a, b):
    """Return the AB pairs as ``minor * 1000 + major`` codes, equal for (a, b) and (b, a).

    Args:
        a: Copies of haplotype A.
        b: Copies of haplotype B, same shape as ``a``.

    Returns:
        int array of codes, same shape as ``a``.
    """
    a, b = np.asarray(a), np.asarray(b)
    return np.minimum(a, b) * 1000 + np.maximum(a, b)


def cna_state(a, b):
    """Return the CNA state of each AB pair (module table).

    Args:
        a: Copies of haplotype A.
        b: Copies of haplotype B, same shape as ``a``.

    Returns:
        str array of ``neu``, ``del``, ``amp``, ``cnloh``, ``bamp`` or ``bdel``.
    """
    a, b = np.asarray(a), np.asarray(b)
    total = a + b
    balanced = a == b
    return np.select(
        [
            balanced & (total == 2),
            balanced & (total > 2),
            balanced,
            total == 2,
            total > 2,
        ],
        ["neu", "bamp", "bdel", "cnloh", "amp"],
        default="del",
    )


def supervised_metrics(true_a, true_b, call_a, call_b):
    """Score called AB pairs against true ones, each (bin, cell) counted once.

    Truly altered means a true pair other than (1, 1). Each rate is NaN where its denominator is 0.

    Args:
        true_a, true_b: True copies, (n_cells, n_bins).
        call_a, call_b: Called copies, same shape.

    Returns:
        dict with
            sensitivity: 1 - (truly altered called (1, 1)) / (truly altered).
            specificity: 1 - (truly (1, 1) called altered) / (truly (1, 1)).
            exact_altered: (truly altered called their true pair) / (truly altered), N-CNA recall.
            exact_altered_hap: as exact_altered, the pair in its orientation: haplotype-specific
                N-CNA recall.
            state_altered: (truly altered called their true CNA state) / (truly altered), CNA recall.
    """
    true_code = phase_free(true_a, true_b)
    call_code = phase_free(call_a, call_b)
    neutral = true_code == 1001
    called_neutral = call_code == 1001
    altered = ~neutral
    n_alt, n_neu = altered.sum(), neutral.sum()
    same_state = cna_state(true_a, true_b) == cna_state(call_a, call_b)
    same_hap = (np.asarray(true_a) == np.asarray(call_a)) & (
        np.asarray(true_b) == np.asarray(call_b)
    )
    return {
        "sensitivity": 1.0 - (altered & called_neutral).sum() / n_alt
        if n_alt
        else float("nan"),
        "specificity": 1.0 - (neutral & ~called_neutral).sum() / n_neu
        if n_neu
        else float("nan"),
        "exact_altered": (altered & (true_code == call_code)).sum() / n_alt
        if n_alt
        else float("nan"),
        "exact_altered_hap": (altered & same_hap).sum() / n_alt
        if n_alt
        else float("nan"),
        "state_altered": (altered & same_state).sum() / n_alt
        if n_alt
        else float("nan"),
    }


def breakpoints(labels, chroms):
    """Count label changes between adjacent bins of the same chromosome, over all cells.

    Args:
        labels: (n_cells, n_bins) labels, bins sorted by position.
        chroms: (n_bins,) chromosome of each bin.

    Returns:
        n_breakpoints: Adjacent same-chromosome bin pairs, summed over cells, whose labels differ.
        n_boundaries: Adjacent same-chromosome bin pairs, summed over cells:
            n_cells * sum over chromosomes of (n_bins_chrom - 1).
    """
    labels = np.asarray(labels)
    chroms = np.asarray(chroms)
    same_chrom = chroms[1:] == chroms[:-1]
    changed = labels[:, 1:] != labels[:, :-1]
    return int((changed & same_chrom).sum()), int(labels.shape[0] * same_chrom.sum())
