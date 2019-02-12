from typing import Tuple

import numpy as np

from processes.mdp import MDP
from processes.mdp_rep_for_adp_pg import MDPRepForADPPG
from processes.mdp_rep_for_rl_pg import MDPRepForRLPG
from processes.mdp_rep_for_rl_tabular import MDPRepForRLTabular
from processes.mp_funcs import flatten_sasf_dict
from processes.mp_funcs import flatten_ssf_dict
from processes.mp_funcs import get_rv_gen_func
from processes.mp_funcs import get_rv_gen_func_single
from processes.mp_funcs import get_sampling_func_from_prob_dict
from processes.mp_funcs import get_state_reward_gen_dict
from processes.mp_funcs import get_state_reward_gen_func
from processes.mp_funcs import mdp_rep_to_mrp_rep1
from processes.mp_funcs import unflatten_sasf_dict
from processes.mp_funcs import unflatten_ssf_dict
from processes.mrp_refined import MRPRefined
from processes.policy import Policy
from utils.gen_utils import zip_dict_of_tuple, merge_dicts
from utils.standard_typevars import SASf, SAf, SASTff


class MDPRefined(MDP):

    def __init__(
        self,
        info: SASTff,
        gamma: float
    ) -> None:
        d1, d2, d3 = MDPRefined.split_info(info)
        super().__init__(
            {s: {a: (v1, d3[s][a]) for a, v1 in v.items()}
             for s, v in d1.items()},
            gamma
        )
        self.rewards_refined: SASf = d2

    @staticmethod
    def split_info(info: SASTff) -> Tuple[SASf, SASf, SAf]:
        c = {s: {a: zip_dict_of_tuple(v1) for a, v1 in v.items()}
             for s, v in info.items()}
        d = {k: zip_dict_of_tuple(v) for k, v in c.items()}
        d1, d2 = zip_dict_of_tuple(d)
        d3 = {s: {a: sum(np.prod(x) for x in v1.values())
                  for a, v1 in v.items()} for s, v in info.items()}
        return d1, d2, d3

    def get_mrp_refined(self, pol: Policy) -> MRPRefined:
        flat_transitions = flatten_sasf_dict(self.transitions)
        flat_rewards_refined = flatten_sasf_dict(self.rewards_refined)

        flat_exp_rewards = merge_dicts(flat_rewards_refined, flat_transitions, lambda x, y: x * y)
        exp_rewards = unflatten_sasf_dict(flat_exp_rewards)

        tr = mdp_rep_to_mrp_rep1(self.transitions, pol.policy_data)
        rew_ref = mdp_rep_to_mrp_rep1(
            exp_rewards,
            pol.policy_data
        )
        flat_tr = flatten_ssf_dict(tr)
        flat_rew_ref = flatten_ssf_dict(rew_ref)
        flat_norm_rewards = merge_dicts(flat_rew_ref, flat_tr, lambda x, y: x / y)
        norm_rewards = unflatten_ssf_dict(flat_norm_rewards)

        return MRPRefined(
            {s: {s1: (v1, norm_rewards[s][s1]) for s1, v1 in v.items()}
             for s, v in tr.items()},
            self.gamma
        )

    def get_mdp_rep_for_rl_tabular(self) -> MDPRepForRLTabular:
        return MDPRepForRLTabular(
            state_action_dict=self.state_action_dict,
            terminal_states=self.terminal_states,
            state_reward_gen_dict=get_state_reward_gen_dict(
                self.transitions,
                self.rewards_refined
            ),
            gamma=self.gamma
        )

    def get_mdp_rep_for_adp_pg(self) -> MDPRepForADPPG:
        return MDPRepForADPPG(
            gamma=self.gamma,
            init_states_gen_func=get_rv_gen_func(
                {s: 1. / len(self.state_action_dict) for s in
                 self.state_action_dict.keys()}
            ),
            state_reward_gen_func=lambda s, a, n:
                [(s1, self.rewards_refined[s][a][s1]) for s1 in
                 get_sampling_func_from_prob_dict(self.transitions[s][a])(n)],
            # reward_func=lambda s, a: self.rewards[s][a],
            # transitions_func=lambda s, a: self.transitions[s][a],
            terminal_state_func=lambda s: s in self.terminal_states
        )

    def get_mdp_rep_for_rl_pg(self) -> MDPRepForRLPG:
        return MDPRepForRLPG(
            gamma=self.gamma,
            init_state_gen_func=get_rv_gen_func_single(
                {s: 1. / len(self.state_action_dict) for s in
                 self.state_action_dict.keys()}
            ),
            state_reward_gen_func=lambda s, a: get_state_reward_gen_func(
                self.transitions[s][a],
                self.rewards_refined[s][a],
            )(),
            terminal_state_func=lambda s: s in self.terminal_states,
        )


if __name__ == '__main__':
    # data = {
    #     1: {
    #         'a': {1: (0.3, 9.2), 2: (0.6, 4.5), 3: (0.1, 5.0)},
    #         'b': {2: (0.3, -0.5), 3: (0.7, 2.6)},
    #         'c': {1: (0.2, 4.8), 2: (0.4, -4.9), 3: (0.4, 0.0)}
    #     },
    #     2: {
    #         'a': {1: (0.3, 9.8), 2: (0.6, 6.7), 3: (0.1, 1.8)},
    #         'c': {1: (0.2, 4.8), 2: (0.4, 9.2), 3: (0.4, -8.2)}
    #     },
    #     3: {
    #         'a': {3: (1.0, 0.0)},
    #         'b': {3: (1.0, 0.0)}
    #     }
    # }
    # mdp_refined_obj = MDPRefined(data, 0.95)
    # print(mdp_refined_obj.all_states)
    # print(mdp_refined_obj.transitions)
    # print(mdp_refined_obj.rewards)
    # print(mdp_refined_obj.rewards_refined)
    # terminal = mdp_refined_obj.get_terminal_states()
    # print(terminal)

    print("This is MDPRefined")
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
    mdp_refined_obj = MDPRefined(mdp_refined_data, 0.97)
    print("Transitions")
    print(mdp_refined_obj.transitions)
    print("Rewards Refined")
    print(mdp_refined_obj.rewards_refined)

    print("----------------")
    print("This is the Policy")
    policy_data = {
        1: {'a': 0.4, 'b': 0.6},
        2: {'a': 0.7, 'c': 0.3},
        3: {'b': 1.0}
    }
    pol_obj = Policy(policy_data)
    print(pol_obj.policy_data)

    print("----------------")
    print("This is MRPRefined")
    mrp_refined_obj = mdp_refined_obj.get_mrp_refined(pol_obj)
    print("Transitions")
    print(mrp_refined_obj.transitions)
    print("Rewards Refined")
    print(mrp_refined_obj.rewards_refined)

    print("-----------------")
    print("This is MDP")
    print("Rewards")
    print(mdp_refined_obj.rewards)

    print("-----------------")
    print("This is MRP from MDP")
    mrp_obj1 = mdp_refined_obj.get_mrp(pol_obj)
    print("Rewards")
    print(mrp_obj1.rewards)

    print("---------------")
    print("This is MRP from MRPRefined")
    print("Rewards")
    print(mrp_refined_obj.rewards)


