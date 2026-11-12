#!/usr/bin/env python2.7
"""CHISEL's Caller with the bin-cell HMM in place of the global clustering.

Author: Runpeng Luo (2026-11-12)

Dependencies:
    Python 2.7 env with CHISEL's requirements (numpy, scipy, statsmodels, matplotlib);
    CHISEL source at ../src/chisel; hmm.py.

Usage:
    python2.7 hmm_caller.py <INPUT> --hmm-states <K> [--hmm-stay 0.99] [--hmm-maxcn 6]
        [--hmm-base 2] [--hmm-restarts 10] [--hmm-npz PATH] [Caller.py options] > calls.tsv
        INPUT        # CHISEL combo.tsv
        --hmm-*      # hmmclustering hyper-parameters
        --hmm-npz    # optional output of the fitted HMM

Inputs:
    INPUT: CHISEL combo.tsv (CHR START END CELL NORMAL COUNT RDR A_COUNT B_COUNT BAF).
Parameters:
    --hmm-states, --hmm-stay, --hmm-maxcn, --hmm-base, --hmm-restarts: hmmclustering's K,
        stay, maxcn, base, restarts.
    Every other option is Caller.py's, parsed by Caller.parse_args: -P maxploidy,
        -A sensitivity, -s significativity, -d shift, --maxhomodel, --fastscaling, -j jobs,
        --seed. Caller's clustering options (-K, -k, -t, -q, -l) are accepted and unused.
Outputs:
    stdout: calls.tsv in Caller.py's format (CHR START END CELL NORM_COUNT COUNT RDR A_COUNT
        B_COUNT BAF CLUSTER CN_STATE); CLUSTER is the HMM state of the (bin, cell). Cloner.py
        reads it as it reads Caller.py's.
    --hmm-npz: chroms, starts, ends (n_bins,); cells, ploidy (n_cells,); data; members
        (n_bins, n_cells); means, variances, state_cn, init_states (K, 2); loglik.
Notes:
    Steps of Caller.main, with CHISEL's functions called unchanged where marked (*):
        1. Caller.read_combo, Caller.format_combo (*).
        2. hmm.hmmclustering in place of Clusterizer.kclustering.
        3. Cell clusters: cluster c of cell e is the bins where the HMM puts cell e in state c.
           RDR, BAF and size follow Caller.est_clusters, per cell.
        4. Caller.init + Caller.calling per cell (*), in a Pool as Caller.call runs them: base
           cluster (identify_base), per-cell scale for each even ploidy up to -P, and the
           ploidy chosen by BIC.
        5. Caller.infer's orientation rule per (bin, cell), then Caller.phase (*).
    Caller.calling reads clusters[c][e], counts[e][c] and rdrs[e][c] only, so a membership
    that differs between cells needs no change to it.
"""

from __future__ import division, print_function

import argparse
import os
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "chisel"))
sys.path.insert(0, HERE)

import Caller  # noqa: E402
import hmm  # noqa: E402


def parse_args(argv=None):
    """Split the HMM options from Caller.py's, which Caller.parse_args validates.

    Args:
        argv: Command-line arguments without the program name.

    Returns:
        (hmm_args, chisel_args): an argparse Namespace and Caller.parse_args's dict.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hmm-states", type=int, required=True)
    parser.add_argument("--hmm-stay", type=float, default=0.99)
    parser.add_argument("--hmm-maxcn", type=int, default=6)
    parser.add_argument("--hmm-base", type=float, default=2.0)
    parser.add_argument("--hmm-restarts", type=int, default=10)
    parser.add_argument("--hmm-npz", default=None)
    hmm_args, rest = parser.parse_known_args(argv)
    return hmm_args, Caller.parse_args(rest)


def cell_clusters(bins, cells, members, mapb):
    """Group each cell's bins by HMM state and summarize them as Caller.est_clusters does.

    For cluster c and cell e: RDR is the wRDR-weighted mean RDR (0.0 without reads), BAF the
    wBAF-weighted mean mirrored BAF (0.5 without SNP reads), size the bins' total length.
    A state absent from a cell gets RDR 0.0, BAF 0.5 and size 0 there.

    Args:
        bins: Caller.read_combo's table, bins[b][e].
        cells: Sorted cells, the columns of ``members``.
        members: (n_bins, n_cells) HMM states.
        mapb: Row of each bin in ``members``.

    Returns:
        clusters: {c: {e: {'RDR', 'BAF', 'size'}}}, as Caller.est_clusters.
        counts: {e: {c: {b: (A, B)}}}, as in Caller.call.
        rdrs: {e: {c: {b: (RDR, wRDR)}}}, as in Caller.call.
    """
    states = sorted(set(int(s) for s in np.unique(members)))
    pos = sorted(mapb, key=mapb.get)
    clusters = {c: {} for c in states}
    counts, rdrs = {}, {}
    for x, e in enumerate(cells):
        part = {c: [] for c in states}
        for b in pos:
            part[int(members[mapb[b], x])].append(b)
        counts[e], rdrs[e] = {}, {}
        for c in states:
            group = part[c]
            wrdr = sum(bins[b][e]["wRDR"] for b in group)
            wbaf = sum(bins[b][e]["wBAF"] for b in group)
            rdr = (
                sum(bins[b][e]["RDR"] * bins[b][e]["wRDR"] for b in group) / float(wrdr)
                if wrdr > 0
                else 0.0
            )
            baf = (
                sum(bins[b][e]["mirroredBAF"] * bins[b][e]["wBAF"] for b in group)
                / float(wbaf)
                if wbaf > 0
                else 0.5
            )
            clusters[c][e] = {
                "RDR": rdr,
                "BAF": baf,
                "size": sum(b[2] - b[1] for b in group),
            }
            counts[e][c] = {tuple(b): tuple(bins[b][e]["AB"]) for b in group}
            rdrs[e][c] = {
                tuple(b): (bins[b][e]["RDR"], bins[b][e]["wRDR"]) for b in group
            }
    return clusters, counts, rdrs


def call_cells(clusters, counts, rdrs, cells, args):
    """Run CHISEL's Caller.calling on every cell, in a Pool initialized by Caller.init.

    Args:
        clusters, counts, rdrs: ``cell_clusters``'s output.
        cells: Cells to call.
        args: Caller.parse_args's dict.

    Returns:
        calls: {e: {c: (A, B)}} with A >= B.
        ploidy: {e: base ploidy chosen by Caller.calling}.
    """
    totsize = set(sum(clusters[c][e]["size"] for c in clusters) for e in cells)
    assert len(totsize) == 1, "Cells cover different genome lengths"
    initargs = (clusters, totsize.pop(), args, counts, rdrs)
    pool = Pool(
        processes=min(args["j"], len(cells)), initializer=Caller.init, initargs=initargs
    )
    bar = Caller.ProgressBar(total=len(cells), length=40, verbose=False)
    calls, ploidy = {}, {}
    for e, cns, p in pool.imap_unordered(Caller.calling, cells):
        calls[e] = cns
        ploidy[e] = p
        bar.progress(advance=True, msg="Called cell {}".format(e))
    pool.close()
    pool.join()
    return calls, ploidy


def orient(bins, cells, members, mapb, calls):
    """Order each (bin, cell) call by its BAF with Caller.infer's rule.

    Caller.infer: ``(max(p), min(p)) if bins[b][e]['BAF'] <= 0.5 else (min(p), max(p))``.

    Args:
        bins, cells, members, mapb: As ``cell_clusters``.
        calls: ``call_cells``'s calls.

    Returns:
        {b: {e: (a, b)}}, the input of Caller.phase.
    """
    out = {}
    for b in bins:
        row = {}
        for x, e in enumerate(cells):
            p = calls[e][int(members[mapb[b], x])]
            row[e] = (max(p), min(p)) if bins[b][e]["BAF"] <= 0.5 else (min(p), max(p))
        out[b] = row
    return out


def main():
    hmm_args, args = parse_args(sys.argv[1:])
    Caller.log("Arguments: {} {}".format(vars(hmm_args), args), level="INFO")

    Caller.log("Reading combined RDRs and BAFs of barcoded cells")
    bins = Caller.read_combo(args["input"])
    data, mapb, cells = Caller.format_combo(bins)
    pos = sorted(mapb, key=mapb.get)

    Caller.log("Clustering (bin, cell) points with the bin-cell HMM")
    members, model = hmm.hmmclustering(
        data,
        [b[0] for b in pos],
        hmm_args.hmm_states,
        stay=hmm_args.hmm_stay,
        maxcn=hmm_args.hmm_maxcn,
        base=hmm_args.hmm_base,
        restarts=hmm_args.hmm_restarts,
        seed=args["seed"],
        return_model=True,
    )

    Caller.log("Estimating RDR and BAF of every cell's clusters")
    clusters, counts, rdrs = cell_clusters(bins, cells, members, mapb)

    Caller.log("Selecting ploidies (Caller.calling)")
    calls, ploidy = call_cells(clusters, counts, rdrs, cells, args)
    msg = "\n".join(
        "Cells with base ploidy {}: {}".format(p, v)
        for p, v in Counter(ploidy.values()).items()
    )
    Caller.log("Number of cells for every ploidy level:\n{}".format(msg), level="INFO")

    Caller.log("Phasing copy-number states along the genome (Caller.phase)")
    copynumbers = Caller.phase(
        orient(bins, cells, members, mapb, calls), cells, args["j"]
    )

    if hmm_args.hmm_npz:
        np.savez(
            hmm_args.hmm_npz,
            chroms=np.array([b[0] for b in pos]),
            starts=np.array([b[1] for b in pos]),
            ends=np.array([b[2] for b in pos]),
            cells=np.array(cells),
            ploidy=np.array([ploidy[e] for e in cells]),
            data=data,
            members=members,
            means=model.means_,
            variances=model.variances_,
            state_cn=model.state_cn(),
            init_states=model.grid_states_[model.init_idx_],
            loglik=np.array(model.loglik_history_),
        )

    Caller.log("Writing results")
    header = [
        "#CHR",
        "START",
        "END",
        "CELL",
        "NORM_COUNT",
        "COUNT",
        "RDR",
        "A_COUNT",
        "B_COUNT",
        "BAF",
    ]
    print("\t".join(header + ["CLUSTER", "CN_STATE"]))
    for b in pos:
        for x, e in enumerate(cells):
            v = bins[b][e]
            fields = [
                b[0],
                b[1],
                b[2],
                e,
                v["NORM"],
                v["wRDR"],
                v["RDR"],
                v["AB"][0],
                v["AB"][1],
                v["BAF"],
            ]
            fields += [int(members[mapb[b], x]), "{}|{}".format(*copynumbers[b][e])]
            print("\t".join(map(str, fields)))


if __name__ == "__main__":
    main()
