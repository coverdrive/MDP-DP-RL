import functools
from typing import Mapping, TypeVar, Tuple, Sequence, List

FlattenedDict = List[Tuple[Tuple, float]]

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


def sum_dicts(dicts: Sequence[Mapping[X, float]]) -> Mapping[X, float]:
    return {k: sum(d.get(k, 0) for d in dicts)
            for k in set.union(*[set(d1) for d1 in dicts])}


def is_approx_eq(a: float, b: float) -> bool:
    return abs(a - b) <= epsilon


def transpose_dict_of_dicts(d: Mapping[X, Mapping[Y, Z]])\
        -> Mapping[Y, Mapping[X, Z]]:
    """
    Returns the transposed dictionary of dictionaries.
    Works on irregularly shaped (non-rectangular) dicts of dicts
    """
    all_y = set(y for _, di in d.items() for y, _ in di.items())
    return {y: {x: val for x, di in d.items()
                for y1, val in di.items() if y1 == y} for y in all_y}


def transpose_dict_of_lists(d: Mapping[X, Sequence[Y]])\
        -> Sequence[Mapping[X, Y]]:
    """
    Returns the transposed list of dictionaries.
    Works on irregularly shaped (non-rectangular) dicts of lists
    """
    max_len = max(len(l) for _, l in d.items())
    return [{k: l[i] for k, l in d.items() if i < len(l)}
            for i in range(max_len)]


def transpose_list_of_dicts(l: Sequence[Mapping[X, Y]])\
        -> Mapping[X, Sequence[Y]]:
    """
    Returns the transposed dictionary of lists.
    Works on irregularly shaped (non-rectangular) lists of dicts
    Will 'compress' the result on irregularly shaped input
    """
    all_k = set(k for d in l for k, _ in d.items())
    return {k: [val for d in l for k1, val in d.items()
                if k1 == k] for k in all_k}


def transpose_list_of_lists(l: Sequence[Sequence[X]]) -> Sequence[Sequence[X]]:
    """
    Returns the transposed list of lists.
    Works on irregularly shaped (non-rectangular) lists of lists
    Will 'compress' the result on irregularly shaped input
    """
    max_len = max(len(lin) for lin in l)
    return [[lin[i] for lin in l if i < len(lin)] for i in range(max_len)]


def merge_dicts(d1: FlattenedDict, d2: FlattenedDict, operation):
    merged = d1 + d2
    from itertools import groupby
    from operator import itemgetter
    from functools import reduce
    sortd = sorted(merged, key=itemgetter(0))
    grouped = groupby(sortd, key=itemgetter(0))
    return [(key, reduce(operation, [x for _, x in group])) for key, group in grouped]

