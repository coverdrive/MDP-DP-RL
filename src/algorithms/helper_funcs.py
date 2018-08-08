from typing import TypeVar, Mapping, Set, Sequence, Optional, Callable, Tuple
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


def get_uniform_policy_func(state_action_func: Callable[[S], Set[A]]) \
        -> Callable[[S], Mapping[A, float]]:

    # noinspection PyShadowingNames
    def upf(s: S, state_action_func=state_action_func) -> Mapping[A, float]:
        actions = state_action_func(s)
        return {a: 1. / len(actions) for a in actions}

    return upf


def get_returns_from_rewards_terminating(
    rewards: Sequence[float],
    gamma: float
) -> np.ndarray:
    sz = len(rewards)
    return toeplitz(
        np.insert(np.zeros(sz - 1), 0, 1.),
        np.power(gamma, np.arange(sz))
    ).dot(rewards)


def get_returns_from_rewards_non_terminating(
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


def get_det_policy_from_qf_dict(qf_dict: Mapping[S, Mapping[A, float]])\
        -> DetPolicy:
    return DetPolicy({s: max(v.items(), key=itemgetter(1))[0]
                      for s, v in qf_dict.items()})


def get_soft_policy_from_qf_dict(
    qf_dict: Mapping[S, Mapping[A, float]],
    softmax: bool,
    epsilon: float
) -> Policy:
    if softmax:
        ret = Policy({s: get_softmax_action_probs(v) for s, v in
                      qf_dict.items()})
    else:
        ret = Policy({s: get_epsilon_action_probs(v, epsilon) for s, v in
                      qf_dict.items()})
    return ret


def get_soft_policy_func_from_qf(
    qf: Callable[[Tuple[S, A]], float],
    state_action_func: Callable[[S], Set[A]],
    softmax: bool,
    epsilon: float
) -> Callable[[S], Mapping[A, float]]:

    # noinspection PyShadowingNames
    def get_act_value_dict_from_state(
        s: S,
        qf=qf,
        state_action_func=state_action_func
    ) -> Mapping[A, float]:
        return {a: qf((s, a)) for a in state_action_func(s)}

    # noinspection PyShadowingNames
    def sp_func(s: S, softmax=softmax, epsilon=epsilon) -> Mapping[A, float]:
        av_dict = get_act_value_dict_from_state(s)
        return get_softmax_action_probs(av_dict) if softmax else\
            get_epsilon_action_probs(av_dict, epsilon)

    return sp_func


def get_vf_dict_from_qf_dict_and_policy(
    qf_dict: Mapping[S, Mapping[A, float]],
    pol: Policy
) -> Mapping[A, float]:
    return {s: sum(pol.get_state_action_probability(s, a) * q
            for a, q in v.items()) for s, v in qf_dict.items()}


def get_policy_func_for_fa(
    pol_func: Callable[[S], Callable[[A], float]],
    state_action_func: Callable[[S], Set[A]]
) -> Callable[[S], Mapping[A, float]]:

    # noinspection PyShadowingNames
    def pf(s: S, pol_func=pol_func, state_action_func=state_action_func)\
            -> Mapping[A, float]:
        return {a: pol_func(s)(a) for a in state_action_func(s)}

    return pf


def get_nt_return_eval_steps(max_steps, gamma, eps):
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
    count = 4
    nt_returns_list = get_returns_from_rewards_non_terminating(
        rewards_list,
        gamma_val,
        count
    )
    print(nt_returns_list)
    term_returns_list = get_returns_from_rewards_terminating(
        rewards_list,
        gamma_val
    )
    print(term_returns_list)
