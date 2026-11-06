import ast
import inspect

import numpy as np
import pytest

import chrysalis_mcp.algorithms as algorithms_pkg
from chrysalis_mcp.algorithms import FlyHash, features
from chrysalis_mcp.substrate.store import active_to_bits

D, M, K, C = 64, 256, 16, 6


@pytest.fixture
def fh() -> FlyHash:
    return FlyHash(D, M, K, C, seed=3)


def _rand(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(D)


def test_exactly_k_active_sorted_in_range(fh):
    for s in range(20):
        a = fh.hash(_rand(s))
        assert len(a) == K == len(set(a.tolist()))
        assert list(a) == sorted(a) and a.min() >= 0 and a.max() < M


def test_deterministic_and_seed_sensitive():
    x = _rand(1)
    assert np.array_equal(FlyHash(D, M, K, C, 3).hash(x), FlyHash(D, M, K, C, 3).hash(x))
    assert not np.array_equal(FlyHash(D, M, K, C, 3).hash(x), FlyHash(D, M, K, C, 4).hash(x))


def test_ties_broken_by_lower_index(fh):
    assert fh.hash(np.zeros(D)).tolist() == list(range(K))


def test_to_bits_popcount_and_substrate_agreement(fh):
    a = fh.hash(_rand(5))
    bits = fh.to_bits(a)
    assert len(bits) == M // 8
    assert sum(b.bit_count() for b in bits) == K
    assert bits == active_to_bits(a.tolist(), M)


def test_input_validation(fh):
    for bad in (np.zeros(D + 1), np.full(D, np.nan), np.full(D, np.inf)):
        with pytest.raises(ValueError):
            fh.hash(bad)
    with pytest.raises(ValueError):
        FlyHash(D, 10, 2, 2, 0)  # m not multiple of 8
    with pytest.raises(ValueError):
        FlyHash(D, M, M + 1, C, 0)


def test_similar_inputs_overlap_more_than_unrelated(fh):
    x = _rand(1)
    near = x + np.random.default_rng(9).normal(0, 0.02, D)
    far = _rand(2)
    ov = lambda a, b: len(set(fh.hash(a)) & set(fh.hash(b)))  # noqa: E731
    assert ov(x, near) > ov(x, far)


def test_features_properties():
    f = features("París", 1024)
    assert f.shape == (1024,) and abs(np.linalg.norm(f) - 1) < 1e-9
    assert np.array_equal(f, features("paris", 1024))  # case/accent folded
    assert np.array_equal(features("a  b", 1024), features("a b", 1024))  # whitespace folded
    assert not np.array_equal(f, features("berlin", 1024))


def test_algorithms_import_nothing_from_package():
    root = inspect.getfile(algorithms_pkg).rsplit("/", 1)[0]
    import pathlib

    files = list(pathlib.Path(root).glob("*.py"))
    assert len(files) >= 2
    for f in files:
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert node.level <= 1 and not (
                    node.level == 0 and "chrysalis" in (node.module or "")
                )
                assert node.level == 0 or node.module in ("flyhash", "ppr", None), f.name
            elif isinstance(node, ast.Import):
                assert not any("chrysalis" in a.name for a in node.names), f.name
