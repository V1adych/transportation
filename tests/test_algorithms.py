import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from north_west_corner import north_west_corner
from russel import russel
from vogel import vogel

test_cases = [
    dict(
        S=np.array([20, 30, 50], dtype=np.float32),
        D=np.array([30, 10, 30, 30], dtype=np.float32),
        C=np.array(
            [
                [5, 4, 3, 2],
                [7, 6, 5, 3],
                [6, 5, 4, 3],
            ],
            dtype=np.float32,
        ),
        valid=True,
    ),
    dict(
        S=np.array([100, 200, 300], dtype=np.float32),
        D=np.array([50, 500, 50], dtype=np.float32),
        C=np.array(
            [
                [10, 7, 8],
                [3, 2, 4],
                [2, 4, 6],
            ],
            dtype=np.float32,
        ),
        valid=True,
    ),
    dict(
        S=np.array([10, 10, 10, 10, 10], dtype=np.float32),
        D=np.array([10, 10, 10, 10, 10], dtype=np.float32),
        C=np.array(
            [
                [2, 3, 1, 4, 5],
                [4, 1, 2, 3, 2],
                [3, 2, 5, 1, 2],
                [1, 4, 3, 2, 1],
                [2, 3, 2, 1, 4],
            ],
            dtype=np.float32,
        ),
        valid=True,
    ),
    dict(
        S=np.array([50, 70, 120, 80, 60], dtype=np.float32),
        D=np.array([100, 120, 70, 90], dtype=np.float32),
        C=np.array(
            [
                [2, 3, 1, 4],
                [4, 1, 2, 3],
                [3, 2, 5, 1],
                [1, 4, 3, 2],
                [2, 3, 2, 1],
            ],
            dtype=np.float32,
        ),
        valid=True,
    ),
    dict(
        S=np.array([20, 40, 10], dtype=np.float32),
        D=np.array([30, 10, 10], dtype=np.float32),
        C=np.array(
            [
                [2, 3, 1],
                [4, 1, 2],
                [3, 2, 5],
            ],
            dtype=np.float32,
        ),
        valid=False,
    ),
]


def test_no_negative_supply():
    for test_case in test_cases:
        S = test_case["S"]
        D = test_case["D"]
        C = test_case["C"]
        valid = test_case["valid"]
        solved, X = north_west_corner(S, C, D)
        assert valid == solved
        if valid:
            assert np.allclose(S, np.sum(X, axis=1))
            assert np.allclose(D, np.sum(X, axis=0))


def test_vogel():
    for test_case in test_cases:
        S = test_case["S"]
        D = test_case["D"]
        C = test_case["C"]
        valid = test_case["valid"]
        solved, X = vogel(S, C, D)
        assert valid == solved
        if valid:
            assert np.allclose(S, np.sum(X, axis=1))
            assert np.allclose(D, np.sum(X, axis=0))


def test_russel():
    for test_case in test_cases:
        S = test_case["S"]
        D = test_case["D"]
        C = test_case["C"]
        valid = test_case["valid"]
        solved, X = russel(S, C, D)
        assert valid == solved
        if valid:
            assert np.allclose(S, np.sum(X, axis=1))
            assert np.allclose(D, np.sum(X, axis=0))
