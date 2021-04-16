import math
import numpy as np

from typing import Any


def alpha(idx: int) -> float:
    """
    Calculate alpha.

    Args:
        idx: index of position

    Returns:
        value
    """

    return 1 / (2 ** 0.5) if idx == 0 else 1


def g_function(u: int, v: int, g: np.ndarray) -> float:
    """
    Apply Discrete Cosine.

    Args:
        u: index of row in image
        v: index of column in image
        g: image size 8x8

    Returns:
        value
    """

    sigma = 0
    for x in range(8):
        for y in range(8):
            sigma += g[x, y] * math.cos((2*x + 1) * u * math.pi/16)\
                             * math.cos((2*y + 1) * v * math.pi/16)
    return alpha(u) * alpha(v) * sigma / 4


def f_function(x: int, y: int, F: np.ndarray) -> float:
    """
    Apply Inverse Discrete Cosine.

    Args:
        x: index of row in image
        y: index of column in image
        F: image size 8x8 after apply DCT

    Returns:
        value
    """

    sigma = 0
    for u in range(8):
        for v in range(8):
            sigma += alpha(u) * alpha(v) \
                    * F[u, v] * math.cos((2*x + 1) * u * math.pi/16)\
                              * math.cos((2*y + 1) * v * math.pi/16)
    return sigma / 4


def apply_func(inp: np.ndarray, func: Any) -> np.ndarray:
    """
    Apply function

    Args:
        inp: input 8x8
        func: function to apply

    Returns:
        result shape 8x8
    """

    res = []

    for i in range(8):
        for j in range(8):
            res.append(func(i, j, inp))

    res = np.array(res).reshape((8, 8))
    return res


def dct_encode(inp: np.ndarray) -> np.ndarray:
    """
    Apply dct encoder

    Args:
        inp: input 8x8

    Returns:
        result
    """

    # TODO: can speed up with parallel or vectorize
    return apply_func(inp, g_function)


def dct_decode(inp: np.ndarray) -> np.ndarray:
    """
    Apply inverse dct

    Args:
        inp: input 8x8

    Returns:
        result
    """

    # TODO: can speed up with parallel or vectorize
    return apply_func(inp, f_function)
