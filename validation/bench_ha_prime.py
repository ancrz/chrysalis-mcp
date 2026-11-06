"""H-a': FlyHash retrieves as well as exact cosine on clustered data, at a fraction of the memory.

Corpus: clusters of ``CLUSTER_SIZE`` texts, each a perturbation (``WORDS_PERTURBED`` of
``WORDS_PER_TEXT`` words replaced) of a cluster template, so ground truth is the cluster label, not
the cosine ranking itself. Metric = cluster precision@(CLUSTER_SIZE-1), query excluded, for exact
cosine over the model-free n-gram features and for FlyHash code overlap; ``fly_over_cosine`` is
their ratio. Ties in the (heavily tied) FlyHash overlap are broken by a fixed per-item random key,
so the result does not depend on corpus order. A sweep over (k, c) shows how tunable the code is.
"""

import json
import random
import time
from typing import Any

import numpy as np

from chrysalis_mcp.algorithms.flyhash import FlyHash, features
from chrysalis_mcp.substrate import FlyHashParams
from validation import RESULTS_DIR
from validation.datasets import synthetic_vocab

SIZES = (1000, 10000)
SWEEP_K = (64, 256, 410)
SWEEP_C = (12, 24)
N_QUERIES = 100
CLUSTER_SIZE = 10
WORDS_PER_TEXT = 8
WORDS_PERTURBED = 2
RESULT_FILE = "ha_prime.json"


def setting_key(k: int, c: int) -> str:
    return f"k{k}_c{c}"


def make_corpus(n: int, rng: random.Random) -> tuple[list[str], np.ndarray]:
    """``n // CLUSTER_SIZE`` clusters of perturbed templates: (texts, cluster label per text)."""
    vocab = synthetic_vocab(rng)
    texts: list[str] = []
    labels: list[int] = []
    for cluster in range(n // CLUSTER_SIZE):
        template = [rng.choice(vocab) for _ in range(WORDS_PER_TEXT)]
        for _ in range(CLUSTER_SIZE):
            words = list(template)
            for i in rng.sample(range(WORDS_PER_TEXT), WORDS_PERTURBED):
                words[i] = rng.choice(vocab)
            texts.append(" ".join(words))
            labels.append(cluster)
    return texts, np.array(labels)


def top_neighbours(
    scores: np.ndarray, tiebreak: np.ndarray, ids: np.ndarray, self_id: int, k: int
) -> list[int]:
    """Item ids of the k best scores, self excluded. ``scores``/``tiebreak``/``ids`` share a
    (possibly permuted) presentation order; ties are broken by the per-item key, so the answer
    is independent of that order."""
    order = np.lexsort((tiebreak, -scores))
    return [int(ids[j]) for j in order if ids[j] != self_id][:k]


def _precision(labels: np.ndarray, picked: list[int], query_id: int) -> float:
    return float(np.mean(labels[picked] == labels[query_id]))


def run(
    sizes: tuple[int, ...] = SIZES,
    n_queries: int = N_QUERIES,
    seed: int = 0,
    sweep_k: tuple[int, ...] = SWEEP_K,
    sweep_c: tuple[int, ...] = SWEEP_C,
    reverse_corpus: bool = False,
) -> dict[str, Any]:
    p = FlyHashParams()
    top = CLUSTER_SIZE - 1
    results: dict[str, Any] = {}
    for n in sizes:
        rng = random.Random(seed)
        texts, labels = make_corpus(n, rng)
        tiebreak = np.array([rng.random() for _ in texts])  # keyed by item id, not position
        query_ids = [rng.randrange(len(texts)) for _ in range(n_queries)]
        corpus = np.stack([features(t, p.d) for t in texts]).astype(np.float32)
        # Presentation order must not matter: rank on a permuted view, map back to item ids.
        perm = np.arange(len(texts))[::-1] if reverse_corpus else np.arange(len(texts))
        t_cos = 0.0
        cos_picks = []
        for qid in query_ids:
            t0 = time.perf_counter()
            scores = corpus[perm] @ corpus[qid]
            cos_picks.append(top_neighbours(scores, tiebreak[perm], perm, qid, top))
            t_cos += time.perf_counter() - t0
        cos_prec = float(
            np.mean([_precision(labels, pk, q) for pk, q in zip(cos_picks, query_ids, strict=True)])
        )
        row: dict[str, Any] = {
            "cosine": {
                "precision": round(cos_prec, 4),
                "bytes_per_node": 4 * p.d,
                "query_ms": round(1000 * t_cos / n_queries, 3),
            },
            "fly": {},
        }
        for k in sweep_k:
            for c in sweep_c:
                fly = FlyHash(p.d, p.m, k, c, p.seed)
                codes = np.zeros((len(texts), p.m), dtype=np.uint8)
                for i in range(len(texts)):
                    codes[i, fly.hash(corpus[i].astype(np.float64))] = 1
                view = codes[perm]  # presentation order; built once, outside the timed region
                prec = agree = t_fly = 0.0
                for qid, cpick in zip(query_ids, cos_picks, strict=True):
                    t0 = time.perf_counter()
                    overlap = view[:, fly.hash(corpus[qid].astype(np.float64))].sum(
                        axis=1, dtype=np.int32
                    )
                    picked = top_neighbours(overlap, tiebreak[perm], perm, qid, top)
                    t_fly += time.perf_counter() - t0
                    prec += _precision(labels, picked, qid)
                    agree += len(set(picked) & set(cpick)) / top
                row["fly"][setting_key(k, c)] = {
                    "k": k,
                    "c": c,
                    "precision": round(prec / n_queries, 4),
                    "fly_over_cosine": round(prec / n_queries / cos_prec, 4) if cos_prec else 0.0,
                    "agreement_with_cosine": round(agree / n_queries, 4),
                    "bytes_per_node": p.m // 8,
                    "query_ms": round(1000 * t_fly / n_queries, 3),
                }
        results[str(len(texts))] = row
    return {
        "hypothesis": "H-a'",
        "config": {
            "flyhash_default": vars(p),
            "n_queries": n_queries,
            "seed": seed,
            "cluster_size": CLUSTER_SIZE,
            "precision_at": top,
            "sweep_k": list(sweep_k),
            "sweep_c": list(sweep_c),
        },
        "data": results,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / RESULT_FILE).write_text(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
