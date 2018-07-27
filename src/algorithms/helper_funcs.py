from typing import TypeVar, Mapping, Set, Callable, Sequence, Tuple
from processes.policy import Policy
from processes.det_policy import DetPolicy
from scipy.stats import rv_discrete
import numpy as np
from scipy.linalg import toeplitz
from operator import itemgetter
from processes.mp_funcs import get_epsilon_action_probs
from processes.mp_funcs import get_softmax_action_probs

S = TypeVar('S')
A = TypeVar('A')
Type1 = Mapping[S, Mapping[A, Mapping[S, float]]]


def get_uniform_policy(state_action_dict: Mapping[S, Set[A]]) -> Policy:
    return Policy({s: {a: 1. / len(v) for a in v} for s, v in
                   state_action_dict.items()})


def get_rv_gen_func(prob_dict: Mapping[S, float]) -> Callable[[int], Sequence[S]]:
    outcomes, probabilities = zip(*prob_dict.items())
    rvd = rv_discrete(values=(range(len(outcomes)), probabilities))
    return lambda n, rvd=rvd, outcomes=outcomes: [outcomes[k]
                                                  for k in rvd.rvs(size=n)]


def get_state_reward_gen_func(
        prob_dict: Mapping[S, float],
        rew_dict: Mapping[S, float]
) -> Callable[[], Tuple[S, float]]:
    gf = get_rv_gen_func(prob_dict)

    def ret_func(gf=gf, rew_dict=rew_dict) -> Tuple[S, float]:
        state_outcome = gf(1)[0]
        reward_outcome = rew_dict[state_outcome]
        return state_outcome, reward_outcome

    return ret_func


def get_state_reward_gen_dict(rr: Type1, tr: Type1) \
        -> Mapping[S, Mapping[A, Callable[[], Tuple[S, float]]]]:
    return {s: {a: get_state_reward_gen_func(tr[s][a], rr[s][a])
                for a, _ in v.items()}
            for s, v in rr.items()}


def get_returns_from_rewards(
    rewards: Sequence[float],
    gamma: float,
    points: int
) -> np.ndarray:
    return toeplitz(
        np.insert(np.zeros(points - 1), 0, 1.),
        np.concatenate((
            np.power(gamma, np.arange(len(rewards) - points + 1)),
            np.zeros(points - 1)
        ))
    ).dot(rewards)


def get_det_policy_from_qf(qf_dict: Mapping[S, Mapping[A, float]]) -> DetPolicy:
    return DetPolicy({s: max(v.items(), key=itemgetter(1))[0]
                      for s, v in qf_dict.items()})


def get_epsilon_policy_from_qf(
    qf_dict: Mapping[S, Mapping[A, float]],
    epsilon: float
) -> Policy:
    return Policy({s: get_epsilon_action_probs(v, epsilon)
                   for s, v in qf_dict.items()})


def get_softmax_policy_from_qf(
    qf_dict: Mapping[S, Mapping[A, float]]
) -> Policy:
    return Policy({s: get_softmax_action_probs(v) for s, v in qf_dict.items()})


def get_soft_policy_from_qf(
    qf_dict: Mapping[S, Mapping[A, float]],
    softmax: bool,
    epsilon: float
):
    return get_softmax_policy_from_qf(qf_dict) if softmax\
        else get_epsilon_policy_from_qf(qf_dict, epsilon)


def get_vf_from_qf_and_policy(
    qf_dict: Mapping[S, Mapping[A, float]],
    pol: Policy
) -> Mapping[A, float]:
    return {s: sum(pol.get_state_action_probability(s, a) * q
            for a, q in v.items()) for s, v in qf_dict.items()}


def get_return_eval_steps(max_steps, gamma, eps):
    low_limit = 0.2 * max_steps
    high_limit = max_steps - 1
    if gamma == 0.:
        val = high_limit
    elif gamma == 1.:
        val = low_limit
    else:
        val = min(
            high_limit,
            max(
                low_limit,
                max_steps - np.log(eps) / np.log(gamma)
            )
        )
    return int(np.floor(val))


if __name__ == '__main__':
    probabilities_dict = {'a': 0.4, 'b': 0.5, 'c': 0.1}
    f = get_rv_gen_func(probabilities_dict)
    out_values = f(20)
    print(out_values)
    mdp_refined_data = {
        1: {
            'a': {1: (0.3, 9.2), 2: (0.6, 4.5), 3: (0.1, 5.0)},
            'b': {2: (0.3, -0.5), 3: (0.7, 2.6)},
            'c': {1: (0.2, 4.8), 2: (0.4, -4.9), 3: (0.4, 0.0)}
        },
        2: {
            'a': {1: (0.3, 9.8), 2: (0.6, 6.7), 3: (0.1, 1.8)},
            'c': {1: (0.2, 4.8), 2: (0.4, 9.2), 3: (0.4, -8.2)}
        },
        3: {
            'a': {3: (1.0, 0.0)},
            'b': {3: (1.0, 0.0)}
        }
    }
    from processes.mdp_refined import MDPRefined

    this_gamma = 0.95
    mdp_ref_obj = MDPRefined(mdp_refined_data, this_gamma)
    rr_data = mdp_ref_obj.rewards_refined
    tr_data = mdp_ref_obj.transitions
    sr_gen_dict = get_state_reward_gen_dict(rr_data, tr_data)
    f1 = sr_gen_dict[1]['b']
    f2 = sr_gen_dict[2]['a']
    l1 = [f1() for _ in range(10)]
    l2 = [f2() for _ in range(10)]
    print(l1)
    print(l2)
    rewards_list = [1., 2., 3., 4., 5., 6.]
    gamma_val = 0.9
    returns_list = get_returns_from_rewards(rewards_list, gamma_val)
    print(returns_list)
