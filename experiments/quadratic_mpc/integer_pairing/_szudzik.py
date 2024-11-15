from math import isqrt
from ._interface import Pairing


def _pair(m, n):
    return n * n + m if n > m else m * m + m * n


def _unpair(n):
    q = isqrt(n)
    l = n - q * q
    return (l, q) if l < q else (q, l - q)


szudzik = Pairing(_pair, _unpair)
