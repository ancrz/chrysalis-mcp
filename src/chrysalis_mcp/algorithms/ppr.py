"""Personalized PageRank over a small weighted undirected graph (numpy, pure, no I/O)."""

from collections.abc import Sequence

import numpy as np

DEFAULT_ALPHA = 0.5  # HippoRAG damping: probability mass returned to the personalization vector
DEFAULT_TOL = 1e-6
DEFAULT_MAX_ITER = 50


def personalized_pagerank(
    n: int,
    edges: Sequence[tuple[int, int, float]],
    personalization: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    tol: float = DEFAULT_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
) -> np.ndarray:
    """Scores ``p`` with ``p <- (1-alpha) A^T p + alpha s`` until L1 delta < ``tol``.

    ``edges`` are ``(u, v, w)`` over node indices ``[0, n)``, treated as undirected (each edge
    contributes in both directions); non-positive weights are ignored and self-loops skipped.
    ``A`` is row-normalized; the mass of dangling nodes (no edges) teleports to ``s``, so ``p``
    sums to 1. ``personalization`` is normalized to sum 1 (all-zero input returns zeros).
    """
    if n == 0:
        return np.zeros(0)
    s = np.asarray(personalization, dtype=np.float64)
    if s.shape != (n,) or not 0.0 < alpha <= 1.0:
        raise ValueError("personalization must have shape (n,) and alpha must be in (0, 1]")
    total = float(s.sum())
    if total <= 0.0:
        return np.zeros(n)
    s = s / total

    kept = [(u, v, w) for u, v, w in edges if u != v and w > 0.0]
    if kept:
        arr = np.asarray(kept, dtype=np.float64)
        peak = float(arr[:, 2].max())
        w = arr[:, 2] / peak  # rescaling leaves the row-normalized chain unchanged
        src = np.concatenate([arr[:, 0], arr[:, 1]]).astype(np.int64)
        dst = np.concatenate([arr[:, 1], arr[:, 0]]).astype(np.int64)
        wt = np.concatenate([w, w])
    else:
        src = dst = np.zeros(0, dtype=np.int64)
        wt = np.zeros(0)
    deg = np.bincount(src, weights=wt, minlength=n)
    dangling = deg == 0.0
    safe = np.where(dangling, 1.0, deg)
    step = wt / safe[src]

    p = s.copy()
    for _ in range(max_iter):
        flow = np.bincount(dst, weights=p[src] * step, minlength=n)
        nxt = (1.0 - alpha) * (flow + float(p[dangling].sum()) * s) + alpha * s
        done = float(np.abs(nxt - p).sum()) < tol
        p = nxt
        if done:
            break
    return p
