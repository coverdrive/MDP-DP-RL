from typing import Mapping, TypeVar, Set, Union, Tuple, Sequence
from utils.gen_utils import memoize, is_approx_eq, sum_dicts

S = TypeVar('S')
A = TypeVar('A')
Type1 = Mapping[S, Mapping[S, float]]
Type2 = Mapping[S, Mapping[A, Tuple[Mapping[S, float], float]]]


@memoize
def get_all_states(d: Union[Type1, Type2]) -> Set[S]:
    return set(d.keys())


@memoize
def get_actions_for_states(mdp_data: Type2) -> Mapping[S, Set[A]]:
    return {k: set(v.keys()) for k, v in mdp_data.items()}


@memoize
def get_all_actions(mdp_data: Type2) -> Set[A]:
    return set().union(*get_actions_for_states(mdp_data).values())


def get_lean_transitions(d: Mapping[S, float]) -> Mapping[S, float]:
    return {s: v for s, v in d.items() if not is_approx_eq(v, 0.0)}


def verify_transitions(
    states: Set[S],
    tr_seq: Sequence[Mapping[S, float]]
) -> bool:
    b1 = set().union(*tr_seq).issubset(states)
    b2 = all(is_approx_eq(sum(d.values()), 1.0) for d in tr_seq)
    return b1 and b2


@memoize
def verify_mp(mp_data: Type1) -> bool:
    all_states = get_all_states(mp_data)
    val_seq = list(mp_data.values())
    return verify_transitions(all_states, val_seq)


@memoize
def verify_mdp(mdp_data: Type2) -> bool:
    all_states = get_all_states(mdp_data)
    check_actions = all(len(v) > 0 for _, v in mdp_data.items())
    val_seq = [v2 for _, v1 in mdp_data.items() for _, (v2, _) in v1.items()]
    return verify_transitions(all_states, val_seq) and check_actions


def mdp_rep_to_mrp_rep1(
    mdp_rep: Mapping[S, Mapping[A, Mapping[S, float]]],
    policy_rep: Mapping[S, Mapping[A, float]]
) -> Mapping[S, Mapping[S, float]]:
    return {s: sum_dicts([{s1: policy_rep[s].get(a, 0) * v2 for s1, v2 in v1.items()}
                         for a, v1 in v.items()]) for s, v in mdp_rep.items()}


def mdp_rep_to_mrp_rep2(
    mdp_refined_rep: Mapping[S, Mapping[A, float]],
    policy_rep: Mapping[S, Mapping[A, float]]
) -> Mapping[S, float]:
    return {s: sum([policy_rep[s].get(a, 0) * v1 for a, v1 in v.items()])
            for s, v in mdp_refined_rep.items()}


if __name__ == '__main__':
    trans = {
        1: {1: 0.3, 2: 0.6, 3: 0.1},
        2: {1: 0.4, 2: 0.2, 3: 0.4},
        3: {1: 0.6, 2: 0.4}
    }
    all_st = get_all_states(trans)
    ver = verify_mp(trans)
    print(all_st)
    print(ver)
    data = {
        1: {
            'a': ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
            'b': ({2: 0.3, 3: 0.7}, 2.8),
            'c': ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2)
        },
        2: {
            'a': ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
            'c': ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2)
        },
        3: {
            'b': ({3: 1.0}, 0.0)
        }
    }
    all_st = get_all_states(data)
    sa = get_actions_for_states(data)
    all_act = get_all_actions(data)
    print(all_st)
    print(sa)
    print(all_act)
    ver = verify_mdp(data)
    print(ver)
    mdp_data1 = {
        1: {
            'a': {1: 0.3, 2: 0.6, 3: 0.1},
            'b': {2: 0.3, 3: 0.7},
            'c': {1: 0.2, 2: 0.4, 3: 0.4}
        },
        2: {
            'a': {1: 0.3, 2: 0.6, 3: 0.1},
            'c': {1: 0.2, 2: 0.4, 3: 0.4}
        },
        3: {
            'b': {3: 1.0}
        }
    }
    policy = {
        1: {'a': 0.4, 'b': 0.6},
        2: {'a': 0.7, 'c': 0.3},
        3: {'b': 1.0}
    }
    mrp_data1 = mdp_rep_to_mrp_rep1(mdp_data1, policy)
    print(mrp_data1)
    mdp_data2 = {
        1: {'a': 7.8, 'b': 2.3, 'c': -13.0},
        2: {'a': -0.4, 'c': 0.7},
        3: {'b': 4.2}
    }
    mrp_data2 = mdp_rep_to_mrp_rep2(mdp_data2, policy)
    print(mrp_data2)

