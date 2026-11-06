"""FlyHash: sparse random projection + winner-take-all (fly olfactory circuit).

``h(x) = WTA_k(M x)`` where ``M`` is an (m x d) sparse binary matrix with ``c`` ones per row and
WTA_k keeps the ``k`` largest responses. Similar inputs share many active cells, so overlap of
codes approximates similarity (catalog B1). Pure and deterministic: same seed -> same codes.

Reverse-engineered from the bio-digital catalog entry B1; no code imported.
"""

import hashlib
import unicodedata

import numpy as np

NGRAM = 3
_HASH_BYTES = 8


class FlyHash:
    def __init__(self, d: int, m: int, k: int, c: int, seed: int) -> None:
        if not (d > 0 and m > 0 and 0 < k <= m and 0 < c <= d):
            raise ValueError("require d, m > 0, 0 < k <= m and 0 < c <= d")
        if m % 8:
            raise ValueError("m must be a multiple of 8")
        self.d, self.m, self.k, self.c, self.seed = d, m, k, c, seed
        rng = np.random.default_rng(seed)
        # Row i = the c distinct input indices sampled by Kenyon cell i.
        self._projection = np.stack([rng.choice(d, size=c, replace=False) for _ in range(m)])
        self._order = np.arange(m)

    def hash(self, x: np.ndarray) -> np.ndarray:
        """Exactly ``k`` active cell ids, ascending. Ties are broken by the lower cell id."""
        v = np.asarray(x, dtype=np.float64)
        if v.shape != (self.d,):
            raise ValueError(f"expected shape ({self.d},), got {v.shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError("input contains non-finite values")
        scores = v[self._projection].sum(axis=1)
        winners = np.lexsort((self._order, -scores))[: self.k]
        return np.sort(winners)

    def to_bits(self, active: np.ndarray) -> bytes:
        """``m/8`` bytes; bit ``i`` of byte ``j`` is cell ``8*j + i`` (matches the substrate)."""
        bits = np.zeros(self.m, dtype=np.uint8)
        bits[np.asarray(active, dtype=np.int64)] = 1
        return np.packbits(bits, bitorder="little").tobytes()


def features(text: str, d: int) -> np.ndarray:
    """Model-free embedding: hashed character 3-grams, accent/case-folded, L2-normalized."""
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    padded = f" {' '.join(folded.split())} "
    vec = np.zeros(d, dtype=np.float64)
    for i in range(len(padded) - NGRAM + 1):
        digest = hashlib.blake2b(padded[i : i + NGRAM].encode("utf-8"), digest_size=_HASH_BYTES)
        vec[int.from_bytes(digest.digest(), "big") % d] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec
