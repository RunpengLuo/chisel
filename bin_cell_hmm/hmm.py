"""Bin-cell HMM: per-cell Markov chains over a shared set of (RDR, BAF) emission states.

Drop-in counterpart of CHISEL's global clustering ``Clusterizer.kclustering``. Both take a
matrix in the layout of ``Caller.format_combo``, where row i is a genomic bin and columns are
``[RDR_e1, BAF_e1, RDR_e2, BAF_e2, ...]`` over the sorted cells. ``kclustering`` returns one
cluster per bin, shared by all cells. ``hmmclustering`` returns one state per (bin, cell).

The BAF is taken as phased: B / (A + B) with B the same haplotype in every bin and cell, on
[0, 1], not mirrored. No emission mean is constrained to BAF <= 0.5, and the copy-number grid
holds both (a, b) and (b, a). Phase switches between bins are not modeled: a switch reads as a
change of state.

Model:
    Each cell is a Markov chain over its ordered bins, restarted at every chromosome. All
    chains share K hidden states, a fixed transition matrix (self-transition ``stay``,
    off-diagonal mass spread uniformly) and a uniform start distribution 1/K. State k emits
    the 2-dim point (RDR, BAF) from a diagonal Gaussian N(mu_k, diag(var_k)). A non-finite
    value is marginalized out of the emission.

Fitting:
    1. Constrained k-means picks K distinct points of the integer copy-number grid
       (a, b) -> ((a + b) / base, b / (a + b)) as the initial means.
    2. Baum-Welch updates only the emission means and variances; start and transition
       probabilities stay fixed. Each chromosome is processed for all cells at once, so the
       forward-backward and Viterbi recursions are vectorized over cells.

The module is valid in Python 2.7 with numpy 1.16 (CHISEL's runtime) and in Python 3.

References:
    Rabiner (1989), A tutorial on hidden Markov models, https://doi.org/10.1109/5.18626
    Zaccaria and Raphael (2021), CHISEL, https://doi.org/10.1038/s41587-020-0661-6
"""

from __future__ import division, print_function

import sys

import numpy as np

LOG_2PI = float(np.log(2.0 * np.pi))


def make_transmat(diag, K):
    """Build a K x K doubly-stochastic transition matrix with constant diagonal.

    Same construction as ``make_transmat`` in hatchet-long-read
    ``src/hatchet/cluster_bins/hmm/hmm_transitions.py``.

    Args:
        diag: Probability of staying in the same state.
        K: Number of states.

    Returns:
        (K, K) array with ``diag`` on the diagonal and ``(1 - diag) / (K - 1)`` elsewhere.
    """
    if K == 1:
        return np.array([[1.0]])
    transmat = np.full((K, K), (1.0 - diag) / (K - 1))
    np.fill_diagonal(transmat, diag)
    return transmat


def cn_grid(maxcn, base=2.0):
    """Enumerate the haplotype-specific integer copy-number states and their expected (RDR, BAF).

    Args:
        maxcn: Maximum total copy number a + b.
        base: Total copy number whose expected RDR is 1.

    Returns:
        states: (G, 2) int array of (a, b) with a + b <= maxcn.
        means: (G, 2) float array of ((a + b) / base, b / (a + b)); the BAF of (0, 0) is 0.5.
    """
    states = []
    for total in range(maxcn + 1):
        for a in range(total, -1, -1):
            states.append((a, total - a))
    states = np.array(states, dtype=int)
    total = states.sum(axis=1)
    baf = np.full(len(states), 0.5)
    # NB: np.divide floors int arrays under Python 2
    np.true_divide(states[:, 1], total, out=baf, where=total > 0)
    return states, np.column_stack([total / float(base), baf])


def _sqdist(X, C):
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1)


def _snap_centers(X, labels, idx, grid):
    sizes = np.bincount(labels, minlength=len(idx))
    new_idx = np.empty_like(idx)
    taken = np.zeros(len(grid), dtype=bool)
    for k in np.argsort(-sizes, kind="stable"):
        target = X[labels == k].mean(axis=0) if sizes[k] > 0 else grid[idx[k]]
        order = np.argsort(((grid - target) ** 2).sum(axis=1), kind="stable")
        pick = order[~taken[order]][0]
        new_idx[k] = pick
        taken[pick] = True
    return new_idx


def constrained_kmeans(X, grid, K, restarts=10, maxiter=100, rng=None):
    """Choose K distinct grid points as cluster centers (k-means with grid-restricted centers).

    Lloyd iterations: assign each point to its nearest selected center, then move each center
    to the unused grid point nearest to the mean of its members (larger clusters choose first).
    Restart 0 starts from the K grid points that are nearest to the most data points; later
    restarts draw K grid points with probability proportional to that count. The restart with
    the lowest within-cluster sum of squared distances is returned.

    Args:
        X: (n, D) points; rows with a non-finite value are ignored.
        grid: (G, D) candidate centers.
        K: Number of centers, 1 <= K <= G.
        restarts: Number of restarts.
        maxiter: Maximum Lloyd iterations per restart.
        rng: ``numpy.random.RandomState`` or seed.

    Returns:
        idx: (K,) indices into ``grid`` of the selected centers.
        labels: (n_finite,) center index of each finite row of X.
        sse: Within-cluster sum of squared distances.

    Raises:
        ValueError: If K is outside [1, G] or X has no finite row.
    """
    rng = rng if isinstance(rng, np.random.RandomState) else np.random.RandomState(rng)
    X = X[np.isfinite(X).all(axis=1)]
    G = len(grid)
    if not 1 <= K <= G:
        raise ValueError(
            "K={} must be in [1, {}] (size of the copy-number grid)".format(K, G)
        )
    if len(X) == 0:
        raise ValueError("No finite (RDR, BAF) point to initialize from")
    dist = _sqdist(X, grid)
    pop = np.bincount(dist.argmin(axis=1), minlength=G).astype(float)
    weight = (pop + 1e-9) / (pop + 1e-9).sum()
    best = None
    for r in range(restarts):
        if r == 0:
            idx = np.argsort(-pop, kind="stable")[:K]
        else:
            idx = rng.choice(G, size=K, replace=False, p=weight)
        for _ in range(maxiter):
            new_idx = _snap_centers(X, dist[:, idx].argmin(axis=1), idx, grid)
            if np.array_equal(new_idx, idx):
                break
            idx = new_idx
        labels = dist[:, idx].argmin(axis=1)
        sse = dist[np.arange(len(X)), idx[labels]].sum()
        if best is None or sse < best[2]:
            best = (idx.copy(), labels, sse)
    return best


def log_emission(X, mask, means, variances):
    """Diagonal-Gaussian log density of every point under every state.

    Args:
        X: (..., D) observations.
        mask: (..., D) bool, False for dimensions to marginalize out.
        means: (K, D) state means.
        variances: (K, D) state variances.

    Returns:
        (..., K) log densities.
    """
    Xf = np.where(mask, X, 0.0)[..., None, :]
    ll = -0.5 * (LOG_2PI + np.log(variances) + (Xf - means) ** 2 / variances)
    return (ll * mask[..., None, :]).sum(axis=-1)


def forward_backward(log_emit, startprob, transmat):
    """Scaled forward-backward on N equal-length sequences sharing start and transitions.

    Args:
        log_emit: (N, T, K) log emission densities.
        startprob: (K,) start probabilities.
        transmat: (K, K) transition probabilities, all entries > 0 when K > 1.

    Returns:
        gamma: (N, T, K) posterior state probabilities.
        loglik: (N,) log-likelihood of each sequence.
    """
    N, T, K = log_emit.shape
    shift = log_emit.max(axis=2, keepdims=True)
    emit = np.exp(log_emit - shift)
    alpha = np.empty_like(emit)
    scale = np.empty((N, T))
    a = startprob * emit[:, 0]
    for t in range(T):
        if t > 0:
            a = np.dot(alpha[:, t - 1], transmat) * emit[:, t]
        scale[:, t] = a.sum(axis=1)
        alpha[:, t] = a / scale[:, t, None]
    beta = np.empty_like(emit)
    beta[:, T - 1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[:, t] = (
            np.dot(emit[:, t + 1] * beta[:, t + 1], transmat.T) / scale[:, t + 1, None]
        )
    gamma = alpha * beta
    gamma /= gamma.sum(axis=2, keepdims=True)
    loglik = np.log(scale).sum(axis=1) + shift[..., 0].sum(axis=1)
    return gamma, loglik


def viterbi(log_emit, startprob, transmat):
    """Most likely state path of N equal-length sequences sharing start and transitions.

    Args:
        log_emit: (N, T, K) log emission densities.
        startprob: (K,) start probabilities.
        transmat: (K, K) transition probabilities.

    Returns:
        path: (N, T) int state indices.
        logprob: (N,) log joint probability of each path.
    """
    N, T, K = log_emit.shape
    with np.errstate(divide="ignore"):
        log_trans = np.log(transmat)
        delta = np.log(startprob) + log_emit[:, 0]
    back = np.empty((N, T, K), dtype=np.intp)
    for t in range(1, T):
        scores = delta[:, :, None] + log_trans
        back[:, t] = scores.argmax(axis=1)
        delta = scores.max(axis=1) + log_emit[:, t]
    path = np.empty((N, T), dtype=np.intp)
    path[:, T - 1] = delta.argmax(axis=1)
    rows = np.arange(N)
    for t in range(T - 1, 0, -1):
        path[:, t - 1] = back[rows, t, path[:, t]]
    return path, delta.max(axis=1)


def split_data(data):
    """Split a CHISEL-layout data matrix into per-cell RDR and BAF matrices.

    Args:
        data: (n_bins, 2 * n_cells) array with columns [RDR_e1, BAF_e1, RDR_e2, BAF_e2, ...].

    Returns:
        rdr: (n_cells, n_bins) array.
        baf: (n_cells, n_bins) array.

    Raises:
        ValueError: If ``data`` is not 2-D with an even number of columns.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] % 2 != 0:
        raise ValueError(
            "data must be (n_bins, 2 * n_cells), got {}".format(data.shape)
        )
    return data[:, 0::2].T, data[:, 1::2].T


def chrom_blocks(chroms):
    """Return the row slices of contiguous runs of equal chromosome labels.

    Args:
        chroms: (n_bins,) chromosome label of each row, rows sorted by position.

    Returns:
        List of ``slice`` objects, one per chromosome.

    Raises:
        ValueError: If a chromosome appears in more than one run.
    """
    chroms = list(chroms)
    starts = [0] + [i for i in range(1, len(chroms)) if chroms[i] != chroms[i - 1]]
    ends = starts[1:] + [len(chroms)]
    labels = [chroms[s] for s in starts]
    if len(set(labels)) != len(labels):
        raise ValueError("Rows of the same chromosome must be contiguous")
    return [slice(s, e) for s, e in zip(starts, ends)]


class BinCellHMM(object):
    """Bin-cell HMM with shared diagonal-Gaussian emission states and fixed transitions.

    Args:
        K: Number of hidden states shared by all cells.
        stay: Fixed self-transition probability between adjacent bins of a cell.
        maxcn: Maximum total copy number of the initialization grid.
        base: Total copy number whose expected RDR is 1 in the grid.
        restarts: Restarts of the constrained k-means initialization.
        maxiter: Maximum Baum-Welch iterations.
        tol: Stop when the total log-likelihood gains less than this.
        varfloor: Lower bound of every emission variance.
        maxinitpoints: Subsample size of (RDR, BAF) points for the initialization.
        seed: Random seed.
        verbose: Log progress to stderr.

    Attributes (set by ``fit``):
        grid_states_: (G, 2) integer (a, b) grid.
        grid_means_: (G, 2) expected (RDR, BAF) of the grid states.
        init_idx_: (K,) grid indices chosen by constrained k-means.
        means_: (K, 2) emission means (RDR, BAF).
        variances_: (K, 2) emission variances.
        startprob_: (K,) uniform start probabilities.
        transmat_: (K, K) fixed transition matrix.
        loglik_history_: Total log-likelihood at each E-step.
        converged_: Whether the log-likelihood gain fell below ``tol``.
    """

    def __init__(
        self,
        K,
        stay=0.99,
        maxcn=6,
        base=2.0,
        restarts=10,
        maxiter=100,
        tol=1e-2,
        varfloor=1e-4,
        maxinitpoints=200000,
        seed=None,
        verbose=False,
    ):
        if K > 1 and not 0.0 < stay < 1.0:
            raise ValueError("stay must be in (0, 1) when K > 1")
        self.K = K
        self.stay = stay
        self.maxcn = maxcn
        self.base = base
        self.restarts = restarts
        self.maxiter = maxiter
        self.tol = tol
        self.varfloor = varfloor
        self.maxinitpoints = maxinitpoints
        self.seed = seed
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            sys.stderr.write("[bin_cell_hmm] {}\n".format(msg))

    @staticmethod
    def _prepare(rdr, baf, chroms):
        rdr = np.asarray(rdr, dtype=float)
        baf = np.asarray(baf, dtype=float)
        if rdr.ndim != 2 or rdr.shape != baf.shape:
            raise ValueError(
                "rdr and baf must be (n_cells, n_bins) arrays of equal shape"
            )
        if len(chroms) != rdr.shape[1]:
            raise ValueError("chroms must have one label per bin")
        X = np.stack([rdr, baf], axis=-1)
        return X, np.isfinite(X), chrom_blocks(chroms)

    def _init_params(self, X, rng):
        self.grid_states_, self.grid_means_ = cn_grid(self.maxcn, self.base)
        pts = X.reshape(-1, 2)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) > self.maxinitpoints:
            pts = pts[rng.choice(len(pts), self.maxinitpoints, replace=False)]
        idx, labels, sse = constrained_kmeans(
            pts, self.grid_means_, self.K, self.restarts, rng=rng
        )
        self.init_idx_ = idx
        self.means_ = self.grid_means_[idx].copy()
        pooled = ((pts - self.means_[labels]) ** 2).mean(axis=0)
        self.variances_ = np.tile(np.maximum(pooled, self.varfloor), (self.K, 1))
        self.startprob_ = np.full(self.K, 1.0 / self.K)
        self.transmat_ = make_transmat(self.stay, self.K)
        self._log(
            "init: {} grid states, K={}, sse={:.4f}, states {}".format(
                len(self.grid_states_),
                self.K,
                sse,
                [tuple(int(v) for v in s) for s in self.grid_states_[idx]],
            )
        )

    def _e_step(self, X, mask, blocks):
        s0 = np.zeros((self.K, 2))
        s1 = np.zeros((self.K, 2))
        s2 = np.zeros((self.K, 2))
        total = 0.0
        for blk in blocks:
            Xb, Mb = X[:, blk], mask[:, blk]
            log_emit = log_emission(Xb, Mb, self.means_, self.variances_)
            gamma, loglik = forward_backward(log_emit, self.startprob_, self.transmat_)
            m = Mb.astype(float)
            xm = np.where(Mb, Xb, 0.0)
            s0 += np.einsum("ntk,ntd->kd", gamma, m)
            s1 += np.einsum("ntk,ntd->kd", gamma, xm)
            s2 += np.einsum("ntk,ntd->kd", gamma, xm**2)
            total += loglik.sum()
        return (s0, s1, s2), total

    def _m_step(self, s0, s1, s2):
        ok = s0 > 1e-8
        safe = np.where(ok, s0, 1.0)
        means = np.where(ok, s1 / safe, self.means_)
        variances = np.where(ok, s2 / safe - means**2, self.variances_)
        self.means_ = means
        self.variances_ = np.maximum(variances, self.varfloor)

    def fit(self, rdr, baf, chroms):
        """Initialize on the copy-number grid, then run Baum-Welch on the emission parameters.

        Args:
            rdr: (n_cells, n_bins) RDR; non-finite entries are marginalized.
            baf: (n_cells, n_bins) BAF; non-finite entries are marginalized.
            chroms: (n_bins,) chromosome label of each bin, bins sorted by position.

        Returns:
            self.
        """
        X, mask, blocks = self._prepare(rdr, baf, chroms)
        self._log(
            "fit: {} cells x {} bins, {} chromosomes".format(
                X.shape[0], X.shape[1], len(blocks)
            )
        )
        self._init_params(X, np.random.RandomState(self.seed))
        self.loglik_history_ = []
        self.converged_ = False
        for it in range(self.maxiter):
            stats, loglik = self._e_step(X, mask, blocks)
            self.loglik_history_.append(loglik)
            self._log("iter {}: loglik {:.4f}".format(it, loglik))
            if it > 0 and loglik - self.loglik_history_[-2] < self.tol:
                self.converged_ = True
                break
            self._m_step(*stats)
        self._log(
            "fit done: converged={}, iterations={}".format(
                self.converged_, len(self.loglik_history_)
            )
        )
        return self

    def predict(self, rdr, baf, chroms):
        """Viterbi state path of every cell.

        Args:
            rdr: (n_cells, n_bins) RDR.
            baf: (n_cells, n_bins) BAF.
            chroms: (n_bins,) chromosome label of each bin.

        Returns:
            (n_cells, n_bins) int array of state indices.
        """
        X, mask, blocks = self._prepare(rdr, baf, chroms)
        path = np.empty(X.shape[:2], dtype=np.intp)
        for blk in blocks:
            log_emit = log_emission(
                X[:, blk], mask[:, blk], self.means_, self.variances_
            )
            path[:, blk] = viterbi(log_emit, self.startprob_, self.transmat_)[0]
        return path

    def predict_proba(self, rdr, baf, chroms):
        """Posterior state probabilities of every (cell, bin).

        Args:
            rdr: (n_cells, n_bins) RDR.
            baf: (n_cells, n_bins) BAF.
            chroms: (n_bins,) chromosome label of each bin.

        Returns:
            (n_cells, n_bins, K) array.
        """
        X, mask, blocks = self._prepare(rdr, baf, chroms)
        post = np.empty(X.shape[:2] + (self.K,))
        for blk in blocks:
            log_emit = log_emission(
                X[:, blk], mask[:, blk], self.means_, self.variances_
            )
            post[:, blk] = forward_backward(log_emit, self.startprob_, self.transmat_)[
                0
            ]
        return post

    def state_cn(self):
        """Integer (a, b) of the grid point nearest to each fitted state mean.

        Returns:
            (K, 2) int array.
        """
        return self.grid_states_[_sqdist(self.means_, self.grid_means_).argmin(axis=1)]


def hmmclustering(
    data,
    chroms,
    K,
    stay=0.99,
    maxcn=6,
    base=2.0,
    restarts=10,
    maxiter=100,
    tol=1e-2,
    varfloor=1e-4,
    seed=None,
    verbose=True,
    return_model=False,
):
    """Cluster (bin, cell) points of a CHISEL-layout data matrix with the bin-cell HMM.

    Counterpart of ``Clusterizer.kclustering(data, restarts, threshold, seed, lord, j, LB, UB)``.
    It takes a ``data`` matrix in the same layout, with the phased BAF in place of CHISEL's
    mirrored one, plus the chromosome of each row and the HMM hyper-parameters.
    ``hmm_caller.format_phased`` builds it; in ``Caller.main`` the call reads::

        data, mapb, cells = format_phased(bins)
        chroms = [b[0] for b in sorted(mapb, key=mapb.get)]
        members = hmmclustering(data, chroms, K=..., seed=args['seed'])

    Args:
        data: (n_bins, 2 * n_cells) matrix with columns [RDR_e1, BAF_e1, RDR_e2, BAF_e2, ...],
            the BAF phased on [0, 1], rows sorted by genomic position.
        chroms: (n_bins,) chromosome of each row.
        K: Number of hidden states shared by all cells.
        stay: Fixed self-transition probability between adjacent bins.
        maxcn: Maximum total copy number of the initialization grid.
        base: Total copy number whose expected RDR is 1 in the grid.
        restarts: Restarts of the constrained k-means initialization.
        maxiter: Maximum Baum-Welch iterations.
        tol: Log-likelihood gain below which Baum-Welch stops.
        varfloor: Lower bound of every emission variance.
        seed: Random seed.
        verbose: Log progress to stderr.
        return_model: Also return the fitted ``BinCellHMM``.

    Returns:
        members: (n_bins, n_cells) int array; ``members[mapb[b], x]`` is the state of bin b in
            the x-th sorted cell. With ``return_model=True``, the tuple (members, model).
    """
    rdr, baf = split_data(data)
    model = BinCellHMM(
        K,
        stay=stay,
        maxcn=maxcn,
        base=base,
        restarts=restarts,
        maxiter=maxiter,
        tol=tol,
        varfloor=varfloor,
        seed=seed,
        verbose=verbose,
    ).fit(rdr, baf, chroms)
    members = model.predict(rdr, baf, chroms).T
    return (members, model) if return_model else members
