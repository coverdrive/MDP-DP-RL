from typing import Mapping, TypeVar, Set
from utils.gen_utils import memoize, is_approx_eq
S = TypeVar('S')


@memoize
def get_all_states(transitions: Mapping[S, Mapping[S, float]]) -> Set[S]:
    return set(transitions.keys())


@memoize
def verify_transitions(transitions: Mapping[S, Mapping[S, float]]) -> bool:
    all_states = get_all_states(transitions)
    vals = transitions.values()
    b1 = set().union(*vals).issubset(all_states)
    b2 = all(is_approx_eq(sum(d.values()), 1.0) for d in iter(vals))
    return b1 and b2


if __name__ == '__main__':
    trans = {
        1: {1: 0.3, 2: 0.6, 3: 0.1},
        2: {1: 0.4, 2: 0.2, 3: 0.4},
        3: {1: 0.6, 2: 0.4}
    }
    all_st = get_all_states(trans)
    ver = verify_transitions(trans)
    print(all_st)
    print(ver)
