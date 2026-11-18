"""FlyVec-lite: unsupervised, bio-plausible word representations learned from the text itself.

Reverse-engineered from Liang et al. 2021 (ICLR, "Can a Fruit Fly Learn Word Embeddings?", FlyVec)
and the local learning rule of Krotov & Hopfield 2019 (PNAS); no code imported. Each training
sample is one word (target) in a short sentence (context): the input is
``x = [bag_of_context (V), one_hot_target (V)]`` and ``K`` hidden "Kenyon cell" units with weights
``W (K x 2V)`` compete: the winner moves towards ``x`` (Hebbian, Oja-normalised) and the unit
ranked ``DELTA`` is pushed away (anti-Hebbian, strength ``ANTI``). A word's code is the ``k``
hidden units with the largest weight on its target one-hot; a text is the normalised sum of its
tokens' codes.

Honest limits of this "lite" variant (documented, not hidden): the update is the ``p = 2`` form of
the energy-based rule (the published FlyVec uses a higher ``p`` and ~400 units on a web-scale
corpus); the vocabulary, the sample count and the epochs are hard-capped, so on a small knowledge
base it learns co-occurrence inside a triple (``subject relation object``), nothing more. Words
never seen in training (typos, new words) have no code: unlike character n-grams the encoder
does not
generalise to them. Deterministic for a given seed.
"""

import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

MAX_VOCAB = 4096
MAX_SAMPLES = 40000
MAX_EPOCHS = 20
MAX_HIDDEN = 1024
DELTA = 2  # the unit with this rank is pushed away (rank 1 = winner)
ANTI = 0.4  # strength of that anti-Hebbian push
LR0 = 0.05  # initial learning rate, decays linearly to 0 over the run
MAX_SENTENCE_TOKENS = 12
_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Casefolded, accent-folded word tokens (``_`` splits words, digits are tokens)."""
    folded = unicodedata.normalize("NFKD", text.casefold().replace("_", " "))
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return _WORD.findall(folded)


def train(
    sentences: Sequence[Sequence[str]],
    *,
    hidden: int = 256,
    active: int = 16,
    epochs: int = 5,
    seed: int = 1729,
    max_vocab: int = 2048,
    max_samples: int = 20000,
) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
    """Train on tokenised sentences; returns ``({token: sorted hidden unit ids}, info)``."""
    if not (1 <= hidden <= MAX_HIDDEN and 1 <= active <= hidden):
        raise ValueError(f"need 1 <= active <= hidden <= {MAX_HIDDEN}")
    if not (1 <= epochs <= MAX_EPOCHS and 1 <= max_vocab <= MAX_VOCAB):
        raise ValueError("epochs/max_vocab out of range")
    if not 1 <= max_samples <= MAX_SAMPLES:
        raise ValueError(f"max_samples must be in [1, {MAX_SAMPLES}]")
    counts: dict[str, int] = {}
    for sent in sentences:
        for tok in sent:
            counts[tok] = counts.get(tok, 0) + 1
    # most frequent first, ties alphabetical: deterministic vocabulary under the cap
    vocab = [t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][:max_vocab]
    index = {t: i for i, t in enumerate(vocab)}
    v = len(vocab)
    samples: list[tuple[list[int], int]] = []  # (context ids, target id)
    for sent in sentences:
        ids = [index[t] for t in sent[:MAX_SENTENCE_TOKENS] if t in index]
        for pos, target in enumerate(ids):
            ctx = ids[:pos] + ids[pos + 1 :]
            if ctx:
                samples.append((ctx, target))
    info: dict[str, Any] = {
        "vocab": v,
        "samples": len(samples),
        "hidden": hidden,
        "active": active,
        "epochs": epochs,
        "seed": seed,
        "rule": "krotov-hopfield p=2 (lite)",
    }
    if v == 0 or not samples:
        return {}, {**info, "samples_used": 0}
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(hidden, 2 * v))
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    per_epoch = min(len(samples), max_samples)
    total, step = epochs * per_epoch, 0
    second = min(DELTA, hidden) - 1  # 0 when there is a single unit: no anti-Hebbian partner
    for _ in range(epochs):
        order = rng.permutation(len(samples))[:per_epoch]
        for si in order:
            ctx, target = samples[int(si)]
            cols = np.array([*ctx, v + target])
            cur = w[:, cols].sum(axis=1)  # <W_mu, x>: x has ones at ctx and at the target
            lr = LR0 * (1.0 - step / total)
            step += 1
            ranked = np.lexsort((np.arange(hidden), -cur))[: second + 1]
            pairs = [(int(ranked[0]), 1.0)]
            if second:
                pairs.append((int(ranked[second]), -ANTI))
            for unit, gain in pairs:
                # dW = lr * g * (x - <W,x> W): the Oja term touches every column, x only `cols`
                w[unit] *= 1.0 - lr * gain * cur[unit]
                w[unit, cols] += lr * gain
                w[unit] /= max(float(np.linalg.norm(w[unit])), 1e-12)
    target_block = w[:, v:]  # (K x V): weight of each unit on each target word
    codes: dict[str, tuple[int, ...]] = {}
    for tok, i in index.items():
        winners = np.lexsort((np.arange(hidden), -target_block[:, i]))[:active]
        codes[tok] = tuple(int(u) for u in np.sort(winners))
    return codes, {**info, "samples_used": per_epoch * epochs}


def encode(tokens: Iterable[str], codes: Mapping[str, Sequence[int]], hidden: int) -> np.ndarray:
    """L2-normalised sum of the tokens' hidden-unit indicators; zeros when no token is known."""
    vec = np.zeros(hidden, dtype=np.float64)
    for tok in tokens:
        for unit in codes.get(tok, ()):
            vec[unit] += 1.0
    norm = np.linalg.norm(vec)
    return (vec / norm if norm else vec).astype(np.float32)
