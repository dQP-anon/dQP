from abc import ABC
from functools import reduce
from collections import deque


class Pairing(ABC):
    def __init__(self, pair_fun, unpair_fun):
        self._pair = pair_fun
        self._unpair = unpair_fun

    @staticmethod
    def _encodeNegatives(l):
        return [2 * x if x >= 0 else -2 * x + 1 for x in l]

    @staticmethod
    def _decodeNegatives(l):
        return [x // 2 if x % 2 == 0 else -(x - 1) // 2 for x in l]

    @staticmethod
    def _containsNegatives(l):
        for x in l:
            if x < 0:
                return True
        return False

    def pair(self, *args):
        """Encodes a list of integers into one non-negative integer.

        Returns:
            int: args encoded into an integer
        """
        assert args
        assert all(isinstance(x, int) for x in args)

        if self._containsNegatives(args):
            args = self._encodeNegatives(args)

        return reduce(self._pair, args) if len(args) > 1 else args[0]

    def unpair(self, n, dim=2, neg=False):
        """Unpairs any number n into an appropriate tuple.

        Args:
            n (int): encoding of the paired tuple
            dim (int): size of the paired tuple encoded by n. Defaults to 2.
            neg (bool): implies the presence of negative values. Defaults to False.

        Returns:
            Tuple[int]: encoding n decoded into a dim-length integer tuple
        """
        assert isinstance(n, int)
        assert isinstance(dim, int)
        assert isinstance(neg, bool)
        assert dim >= 1

        p = deque([])
        for _ in range(dim - 1):
            n, m = self._unpair(n)
            p.appendleft(m)
        p.appendleft(n)

        if neg:
            p = self._decodeNegatives(p)
        return tuple(p)


class Bundle(ABC):
    def __init__(self, bundle_fun, unbundle_fun):
        self._bundle = bundle_fun
        self._unbundle = unbundle_fun

    def pair(self, *args):
        """Encodes any number of integers into one integer, 
        along with the information about negative integers and 
        the number of encoded integers.

        Returns:
            _type_: _description_
        """
        assert args
        assert all(isinstance(x, int) for x in args)
        return self._bundle(args)

    def unpair(self, n):
        """Decodes n if n was a result of bundling an integer tuple.

        Args:
            n (int): non-negative integer obtained via bundle.pair

        Returns:
            Tuple[int]: encoding n decoded into an integer tuple
        """
        assert isinstance(n, int)
        assert n >= 0
        return self._unbundle(n)