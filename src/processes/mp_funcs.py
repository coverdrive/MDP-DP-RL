from typing import Mapping, TypeVar, Set, Tuple, Sequence, Any, Callable
from utils.gen_utils import memoize, is_approx_eq, sum_dicts
import numpy as np
from operator import itemgetter
from scipy.stats import rv_discrete

S = TypeVar('S')
A = TypeVar('A')
SSf = Mapping[S, Mapping[S, float]]
SAf = Mapping[S, Mapping[A, float]]
SASf = Mapping[S, Mapping[A, Mapping[S, float]]]
SATSff = Mapping[S, Mapping[A, Tuple[Mapping[S, float], float]]]


@memoize
def get_all_states(d: Mapping[S, Any]) -> Set[S]:
    return set(d.keys())


@memoize
def get_actions_for_states(mdp_data: Mapping[S, Mapping[A, Any]])\
        -> Mapping[S, Set[A]]:
    return {k: set(v.keys()) for k, v in mdp_data.items()}


@memoize
def get_all_actions(mdp_data: Mapping[S, Mapping[A, Any]]) -> Set[A]:
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
def verify_mp(mp_data: SSf) -> bool:
    all_st = get_all_states(mp_data)
    val_seq = list(mp_data.values())
    return verify_transitions(all_st, val_seq)


@memoize
def verify_mdp(mdp_data: SATSff) -> bool:
    all_st = get_all_states(mdp_data)
    check_actions = all(len(v) > 0 for _, v in mdp_data.items())
    val_seq = [v2 for _, v1 in mdp_data.items() for _, (v2, _) in v1.items()]
    return verify_transitions(all_st, val_seq) and check_actions


@memoize
def verify_policy(policy_data: SAf) -> bool:
    return all(is_approx_eq(sum(v.values()), 1.0) for s, v in policy_data.items())


def mdp_rep_to_mrp_rep1(
    mdp_rep: SASf,
    policy_rep: SAf
) -> SSf:
    return {s: sum_dicts([{s1: policy_rep[s].get(a, 0) * v2 for s1, v2
                           in v1.items()}
                          for a, v1 in v.items()]) for s, v in mdp_rep.items()}


def mdp_rep_to_mrp_rep2(
    mdp_rep: SAf,
    policy_rep: SAf
) -> Mapping[S, float]:
    return {s: sum([policy_rep[s].get(a, 0) * v1 for a, v1 in v.items()])
            for s, v in mdp_rep.items()}


# noinspection PyShadowingNames
def get_rv_gen_func_single(prob_dict: Mapping[S, float])\
        -> Callable[[], S]:
    outcomes, probabilities = zip(*prob_dict.items())
    rvd = rv_discrete(values=(range(len(outcomes)), probabilities))
    return lambda rvd=rvd, outcomes=outcomes: outcomes[rvd.rvs(size=1)[0]]


# noinspection PyShadowingNames
def get_rv_gen_func(prob_dict: Mapping[S, float])\
        -> Callable[[int], Sequence[S]]:
    outcomes, probabilities = zip(*prob_dict.items())
    rvd = rv_discrete(values=(range(len(outcomes)), probabilities))
    return lambda n, rvd=rvd, outcomes=outcomes: [outcomes[k]
                                                  for k in rvd.rvs(size=n)]


def get_state_reward_gen_func(
        prob_dict: Mapping[S, float],
        rew_dict: Mapping[S, float]
) -> Callable[[], Tuple[S, float]]:
    gf = get_rv_gen_func_single(prob_dict)

    # noinspection PyShadowingNames
    def ret_func(gf=gf, rew_dict=rew_dict) -> Tuple[S, float]:
        state_outcome = gf()
        reward_outcome = rew_dict[state_outcome]
        return state_outcome, reward_outcome

    return ret_func


def get_state_reward_gen_dict(rr: SASf, tr: SASf) \
        -> Mapping[S, Mapping[A, Callable[[], Tuple[S, float]]]]:
    return {s: {a: get_state_reward_gen_func(tr[s][a], rr[s][a])
                for a, _ in v.items()}
            for s, v in rr.items()}


def get_epsilon_action_probs(
    action_value_dict: Mapping[A, float],
    epsilon: float
) -> Mapping[A, float]:
    return {a: epsilon / len(action_value_dict) +
            (1. - epsilon if a == max(
                action_value_dict.items(),
                key=itemgetter(1)
            )[0] else 0.)
            for a in action_value_dict.keys()}


def get_softmax_action_probs(
    action_value_dict: Mapping[A, float]
) -> Mapping[A, float]:
    aq = {a: q - max(action_value_dict.values())
          for a, q in action_value_dict.items()}
    exp_sum = sum(np.exp(q) for q in aq.values())
    return {a: np.exp(q) / exp_sum for a, q in aq.items()}


if __name__ == '__main__':
    trans = {
        1: {1: 0.3, 2: 0.6, 3: 0.1},
        2: {1: 0.4, 2: 0.2, 3: 0.4},
        3: {1: 0.6, 2: 0.4}
    }
    all_the_st = get_all_states(trans)
    ver = verify_mp(trans)
    print(all_the_st)
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
    all_the_st = get_all_states(data)
    sa = get_actions_for_states(data)
    all_act = get_all_actions(data)
    print(all_the_st)
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

