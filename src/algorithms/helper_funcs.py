from typing import TypeVar, Mapping, Set, Sequence, Optional
from processes.policy import Policy
from processes.det_policy import DetPolicy
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


def get_returns_from_rewards(
    rewards: Sequence[float],
    gamma: float,
    points: Optional[int] = None
) -> np.ndarray:
    cnt = points if points is not None else len(rewards)
    return toeplitz(
        np.insert(np.zeros(cnt - 1), 0, 1.),
        np.concatenate((
            np.power(gamma, np.arange(len(rewards) - cnt + 1)),
            np.zeros(cnt - 1)
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
    rewards_list = [1., 2., 3., 4., 5., 6.]
    gamma_val = 0.9
    returns_list = get_returns_from_rewards(rewards_list, gamma_val)
    print(returns_list)
