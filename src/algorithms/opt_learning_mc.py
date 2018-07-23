from typing import TypeVar, Mapping, Optional, Tuple, Sequence
from algorithms.opt_learning_base import OptLearningBase
from processes.mdp_refined import MDPRefined
from processes.policy import Policy
from processes.det_policy import DetPolicy
from algorithms.helper_funcs import get_rv_gen_func, get_returns_from_rewards
from algorithms.helper_funcs import get_soft_policy_from_qf
from algorithms.helper_funcs import get_det_policy_from_qf
import numpy as np

S = TypeVar('S')
A = TypeVar('A')
VFType = Mapping[S, float]
QVFType = Mapping[S, Mapping[A, float]]


class OptLearningMC(OptLearningBase):

    def __init__(
        self,
        mdp_ref_obj: MDPRefined,
        first_visit: bool,
        softmax: bool,
        epsilon: float,
        num_episodes: int
    ) -> None:

        super().__init__(mdp_ref_obj, softmax, epsilon, num_episodes)
        self.first_visit: bool = first_visit

    def get_mc_path(
        self,
        pol: Policy,
        start_state: S,
        start_action: Optional[A] = None,
        max_steps: Optional[int] = None
    ) -> Sequence[Tuple[S, A, float, bool]]:

        res = []
        next_state = start_state
        steps = 0
        terminate = False
        if max_steps is None:
            max_steps = 1e8
        state_occ = {s: False for s in self.state_action_dict.keys()}
        act_gen_dict = {s: get_rv_gen_func(pol.get_state_probabilities(s))
                        for s in self.state_action_dict.keys()}

        while not terminate:
            state = next_state
            state_occ[state] = True if not state_occ[state] else False
            action = act_gen_dict[state](1)[0]\
                if (steps > 0 or start_action is None) else start_action
            next_state, reward = self.state_reward_gen_dict[state][action]()
            res.append((state, action, reward, state_occ[state]))
            steps += 1
            terminate = steps >= max_steps or state in self.terminal_states
        return res

    def get_value_func_dict(self, pol: Policy) -> VFType:
        sa_dict = self.state_action_dict
        s_uniform_dict = {s: 1. / len(sa_dict) for s in sa_dict.keys()}
        start_gen_f = get_rv_gen_func(s_uniform_dict)
        counts_dict = {s: 0 for s in sa_dict.keys()}
        vf_dict = {s: 0.0 for s in sa_dict.keys()}
        episodes = 0

        while episodes < self.num_episodes:
            start_state = start_gen_f(1)[0]
            mc_path = self.get_mc_path(
                pol,
                start_state,
                start_action=None,
                max_steps=10000
            )
            returns = get_returns_from_rewards(
                np.array([x for _, _, x, _ in mc_path]),
                self.gamma
            )
            for i, (s, _, _, f) in enumerate(mc_path):
                if not self.first_visit or f:
                    counts_dict[s] += 1
                    c = counts_dict[s]
                    vf_dict[s] = (vf_dict[s] * (c - 1) + returns[i]) / c
            episodes += 1

        return vf_dict

    def get_act_value_func_dict(self, pol: Policy) -> QVFType:
        sa_dict = self.state_action_dict
        sa_uniform_dict = {(s, a): 1. / sum(len(v) for v in sa_dict.values())
                           for s, v1 in sa_dict.items() for a in v1}
        start_gen_f = get_rv_gen_func(sa_uniform_dict)
        counts_dict = {s: {a: 0 for a in v} for s, v in sa_dict.items()}
        qf_dict = {s: {a: 0.0 for a in v} for s, v in sa_dict.items()}
        episodes = 0

        while episodes < self.num_episodes:
            start_state, start_action = start_gen_f(1)[0]
            mc_path = self.get_mc_path(
                pol,
                start_state,
                start_action,
                max_steps=10000
            )
            returns = get_returns_from_rewards(
                np.array([x for _, _, x, _ in mc_path]),
                self.gamma
            )
            for i, (s, a, _, f) in enumerate(mc_path):
                if not self.first_visit or f:
                    counts_dict[s][a] += 1
                    c = counts_dict[s][a]
                    qf_dict[s][a] = (qf_dict[s][a] * (c - 1) + returns[i]) / c
            episodes += 1

        return qf_dict

    def get_optimal(self) -> Tuple[DetPolicy, VFType]:
        pol = self.get_init_policy()
        sa_dict = self.state_action_dict
        s_uniform_dict = {s: 1. / len(sa_dict) for s in sa_dict.keys()}
        start_gen_f = get_rv_gen_func(s_uniform_dict)
        counts_dict = {s: {a: 0 for a in v} for s, v in sa_dict.items()}
        qf_dict = {s: {a: 0.0 for a in v} for s, v in sa_dict.items()}
        episodes = 0

        while episodes < self.num_episodes:
            start_state = start_gen_f(1)[0]
            mc_path = self.get_mc_path(
                pol,
                start_state,
                start_action=None,
                max_steps=10000
            )
            returns = get_returns_from_rewards(
                np.array([x for _, _, x, _ in mc_path]),
                self.gamma
            )
            for i, (s, a, _, f) in enumerate(mc_path):
                if not self.first_visit or f:
                    counts_dict[s][a] += 1
                    c = counts_dict[s][a]
                    qf_dict[s][a] = (qf_dict[s][a] * (c - 1) + returns[i]) / c
            pol = get_soft_policy_from_qf(qf_dict, self.softmax, self.epsilon)
            episodes += 1

        pol = get_det_policy_from_qf(qf_dict)
        vf_dict = self.get_value_func_dict(pol)
        return pol, vf_dict


if __name__ == '__main__':
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
    mdp_ref_obj1 = MDPRefined(mdp_refined_data)

    first_visit_flag = True
    softmax_flag = False
    episodes_limit = 10000
    epsilon_val = 0.1
    mc_obj = OptLearningMC(
        mdp_ref_obj1,
        first_visit_flag,
        softmax_flag,
        epsilon_val,
        episodes_limit
    )

    policy_data = {
        1: {'a': 0.4, 'b': 0.6},
        2: {'a': 0.7, 'c': 0.3},
        3: {'b': 1.0}
    }
    pol_obj = Policy(policy_data)

    this_mc_path = mc_obj.get_mc_path(pol_obj, 1)
    print(this_mc_path)

    this_qf_dict = mc_obj.get_act_value_func_dict(pol_obj)
    print(this_qf_dict)
    this_vf_dict = mc_obj.get_value_func_dict(pol_obj)
    print(this_vf_dict)

    opt_pol, opt_vf_dict = mc_obj.get_optimal()
    print(opt_pol.policy_data)
    print(opt_vf_dict)
