"""
robust_agg.py — Byzantine-robust gradient aggregators, corresponding to
Section 5.4 (robust strategic-layer aggregation) and the deterministic
containment guarantee of Proposition 1 (Section 5.5) for coordinate-wise
median specifically.

All aggregators take an (N, D) array of per-agent contributions (already
trust-reweighted upstream, i.e. x_i = u_i * g_i per Section 5.4) and return
a single (D,) aggregate.
"""
from __future__ import annotations
import numpy as np


def naive_mean(contributions: np.ndarray) -> np.ndarray:
    """Unweighted average. Not Byzantine-robust — provided only as the
    non-robust baseline for comparison (Section 6.5, method 1)."""
    return contributions.mean(axis=0)


def coordinate_wise_median(contributions: np.ndarray) -> np.ndarray:
    """Coordinate-wise median (CWM). This is the aggregator covered by
    Proposition 1's deterministic containment bound: for f < N/2 Byzantine
    rows, the output on every coordinate is confined to the interval
    containing the honest majority, independent of any distributional
    assumption. See Section 5.5.
    """
    return np.median(contributions, axis=0)


def trimmed_mean(contributions: np.ndarray, beta: float) -> np.ndarray:
    """Coordinate-wise trimmed mean: discard the top and bottom beta*N
    contributions per coordinate, average the remainder. Requires
    beta >= eta (Byzantine fraction) for the standard robustness guarantee
    (Yin et al. 2018) — see the Corollary 1 caveat in Section 5.5 about not
    asserting a specific rate without checking the exact regularity
    conditions against the actual honest-gradient distribution.
    """
    n = contributions.shape[0]
    k = int(np.floor(beta * n))
    if k == 0:
        return contributions.mean(axis=0)
    sorted_vals = np.sort(contributions, axis=0)
    trimmed = sorted_vals[k: n - k] if n - k > k else sorted_vals[k:k + 1]
    return trimmed.mean(axis=0)


def krum(contributions: np.ndarray, f: int) -> np.ndarray:
    """Krum (Blanchard et al. 2017): select the single contribution
    minimizing the sum of squared distances to its (N-f-2) nearest
    neighbors. Provable tolerance f < (N-2)/2. O(N^2 D) cost — flagged as a
    scalability caveat in Section 5.8 / 6.9 for large N.
    """
    n = contributions.shape[0]
    m = max(n - f - 2, 1)
    # pairwise squared distances
    diffs = contributions[:, None, :] - contributions[None, :, :]
    sq_dists = np.sum(diffs ** 2, axis=-1)
    scores = np.zeros(n)
    for i in range(n):
        d = np.sort(sq_dists[i])
        d = d[d > 0] if len(d[d > 0]) >= m else d  # exclude self-distance (0)
        scores[i] = d[:m].sum()
    best = int(np.argmin(scores))
    return contributions[best].copy()


def multi_krum(contributions: np.ndarray, f: int, m_select: int | None = None) -> np.ndarray:
    """Multi-Krum: average the m_select lowest-scoring Krum candidates
    rather than taking a single one, trading a modest robustness reduction
    for improved convergence stability (Section 2.2)."""
    n = contributions.shape[0]
    m = max(n - f - 2, 1)
    if m_select is None:
        m_select = max(n - 2 * f - 2, 1)
    diffs = contributions[:, None, :] - contributions[None, :, :]
    sq_dists = np.sum(diffs ** 2, axis=-1)
    scores = np.zeros(n)
    for i in range(n):
        d = np.sort(sq_dists[i])
        d = d[d > 0] if len(d[d > 0]) >= m else d
        scores[i] = d[:m].sum()
    selected = np.argsort(scores)[:m_select]
    return contributions[selected].mean(axis=0)


ROBUST_AGGREGATORS = {
    "naive": lambda x, **kw: naive_mean(x),
    "median": lambda x, **kw: coordinate_wise_median(x),
    "trimmed_mean": lambda x, beta=0.2, **kw: trimmed_mean(x, beta),
    "krum": lambda x, f=0, **kw: krum(x, f),
    "multi_krum": lambda x, f=0, **kw: multi_krum(x, f),
}
