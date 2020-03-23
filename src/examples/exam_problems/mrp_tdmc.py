from typing import Sequence, Tuple, Mapping
from operator import itemgetter
import numpy as np
from itertools import groupby
from numpy.random import randint

S = str
DataType = Sequence[Sequence[Tuple[S, float]]]
ProbFunc = Mapping[S, Mapping[S, float]]
RewardFunc = Mapping[S, float]
ValueFunc = Mapping[S, float]


def get_state_return_samples(
    data: DataType
) -> Sequence[Tuple[S, float]]:
    return [(s, sum(r for (_, r) in l[i:]))
            for l in data for i, (s, _) in enumerate(l)]


def get_mc_value_function(
    state_return_samples: Sequence[Tuple[S, float]]
) -> ValueFunc:
    sorted_samples = sorted(state_return_samples, key=itemgetter(0))
    return {s: np.mean([r for _, r in l])
            for s, l in groupby(sorted_samples, itemgetter(0))}


def get_state_reward_next_state_samples(
    data: DataType
) -> Sequence[Tuple[S, float, S]]:
    return [(s, r, l[i+1][0] if i < len(l) - 1 else 'T')
            for l in data for i, (s, r) in enumerate(l)]


def get_probability_and_reward_functions(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> Tuple[ProbFunc, RewardFunc]:
    d = {s: [(r, s1) for _, r, s1 in l] for s, l in
         groupby(sorted(srs_samples, key=itemgetter(0)), itemgetter(0))}

    prob_func = {s: {s1: len(list(l1)) / len(l) for s1, l1 in
                     groupby(sorted(l, key=itemgetter(1)), itemgetter(1))
                     if s1 != 'T'} for s, l in d.items()}
    reward_func = {s: np.mean([r for r, _ in l]) for s, l in d.items()}

    return prob_func, reward_func


def get_mrp_value_function(
    prob_func: ProbFunc,
    reward_func: RewardFunc
) -> ValueFunc:
    states_list = list(reward_func.keys())
    reward_vec = np.array([reward_func[s] for s in states_list])
    prob_matrix = np.array([[prob_func[s][s1] if s1 in prob_func[s] else 0.
                            for s1 in states_list] for s in states_list])
    vec = np.linalg.inv(np.eye(len(states_list)) - prob_matrix).dot(reward_vec)
    return {states_list[i]: vec[i] for i in range(len(states_list))}


def get_td_value_function(
    srs_samples: Sequence[Tuple[S, float, S]],
    num_updates: int = 300000,
    learning_rate: float = 0.3,
    learning_rate_decay: int = 30
) -> ValueFunc:
    ret = {s: [0.] for s in set(x for x, _, _ in srs_samples)}
    samples = len(srs_samples)
    for updates in range(num_updates):
        s, r, s1 = srs_samples[randint(samples, size=1)[0]]
        ret[s].append(ret[s][-1] + learning_rate *
                      (updates / learning_rate_decay + 1) ** -0.5
                      * (r + (ret[s1][-1] if s1 != 'T' else 0.) - ret[s][-1]))
    return {s: np.mean(v[-int(len(v) * 0.9):]) for s, v in ret.items()}


def get_lstd_value_function(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> ValueFunc:
    nt_states = list(set(x for x, _, _ in srs_samples))
    num_nt_states = len(nt_states)
    phi = np.eye(num_nt_states)
    a_mat = np.zeros((num_nt_states, num_nt_states))
    b_vec = np.zeros(num_nt_states)
    for s, r, s1 in srs_samples:
        p1 = phi[nt_states.index(s)]
        p2 = phi[nt_states.index(s1)] if s1 != 'T' else np.zeros(num_nt_states)
        a_mat += np.outer(p1, p1 - p2)
        b_vec += p1 * r
    return {nt_states[i]: v for i, v in
            enumerate(np.linalg.inv(a_mat).dot(b_vec))}


if __name__ == '__main__':
    given_data: DataType = [
        [('A', 2.), ('A', 6.), ('B', 1.), ('B', 2.)],
        [('A', 3.), ('B', 2.), ('A', 4.), ('B', 2.), ('B', 0.)],
        [('B', 3.), ('B', 6.), ('A', 1.), ('B', 1.)],
        [('A', 0.), ('B', 2.), ('A', 4.), ('B', 4.), ('B', 2.), ('B', 3.)],
        [('B', 8.), ('B', 2.)]
    ]

    print("------------- STATE-RETURN SAMPLES --------------")
    sr_samps = get_state_return_samples(given_data)
    print(sr_samps)
    print("------------- MONTE CARLO VALUE FUNCTION --------------")
    print(get_mc_value_function(sr_samps))

    print("------------- SRS SAMPLES ----------------")
    srs_samps = get_state_reward_next_state_samples(given_data)
    print(srs_samps)

    print("------------- MRP --------------")
    pfunc, rfunc = get_probability_and_reward_functions(srs_samps)
    print(pfunc)
    print(rfunc)
    print("-------------- MRP VALUE FUNCTION ----------")
    print(get_mrp_value_function(pfunc, rfunc))

    print("------------- TD VALUE FUNCTION --------------")
    print(get_td_value_function(srs_samps))

    print("------------- LSTD VALUE FUNCTION --------------")
    print(get_lstd_value_function(srs_samps))
