"""Constants and helpers shared by the general kernel (:mod:`orcut._core`) and the witness network
(:mod:`orcut._witness`)."""
from __future__ import annotations

import numpy as np
from numba import njit

M_LAZY = 0
M_NONLAZY = 1
M_IDLE = 2
M_LLY = 3
M_WALK = 4

_LMAX = np.int64(1) << np.int64(50)   # common denominators up to 2^50 keep W1 / den exact in float64


@njit(cache=True, inline="always")
def _gcd(a, b):
    a = abs(a)
    b = abs(b)
    while b != 0:
        a, b = b, a % b
    return a
