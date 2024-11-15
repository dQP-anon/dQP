from math import isqrt
from ._interface import Pairing


def _pair(m, n):
    return (m + n) * (m + n + 1) // 2 + n


def _unpair(z):
    a = 8 * z + 1
    m = isqrt(100 * a // 4)
    w = m // 10
    if m % 10 < 5:
        w -= 1
    t = (w * w + w) // 2
    y = z - t
    x = w - y
    return x, y


cantor = Pairing(_pair, _unpair)
