from numpy import base_repr
from ._interface import Bundle


def _bundle(nums):
    s = ''
    for n in nums:
        prepend = '2' if n >= 0 else '22'
        s += prepend + base_repr(abs(n), 2)
    return int(s, 3)


def _unbundle(n):
    s = base_repr(n, 3).replace('22', '2-')
    l = (s[1:] if s[0] == '2' else s).split('2')
    return tuple(int(x, 2) for x in l)


bundle = Bundle(_bundle, _unbundle)
