import functools
from typing import Mapping, TypeVar, Tuple

X = TypeVar('X')
Y = TypeVar('Y')
Z = TypeVar('Z')

epsilon = 1e-8


def memoize(func):
    cache = func.cache = {}

    @functools.wraps(func)
    def memoized_func(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return memoized_func


def zip_dict_of_tuple(d: Mapping[X, Tuple[Y, Z]])\
        -> Tuple[Mapping[X, Y], Mapping[X, Z]]:
    d1 = {k: v1 for k, (v1, _) in d.items()}
    d2 = {k: v2 for k, (_, v2) in d.items()}
    return d1, d2


def is_approx_eq(a: float, b: float) -> bool:
    return abs(a - b) <= epsilon
