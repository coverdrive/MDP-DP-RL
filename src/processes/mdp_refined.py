from typing import Tuple
from utils.gen_utils import zip_dict_of_tuple
from processes.mdp import MDP
import numpy as np
from processes.policy import Policy
from processes.mp_funcs import mdp_rep_to_mrp_rep1
from processes.mrp_refined import MRPRefined
from processes.mdp_rep_for_rl_tabular import MDPRepForRLTabular
from processes.mp_funcs import get_state_reward_gen_dict
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
        tr = mdp_rep_to_mrp_rep1(self.transitions, pol.policy_data)
        rew_ref = mdp_rep_to_mrp_rep1(
            self.rewards_refined,
            pol.policy_data
        )
        return MRPRefined(
            {s: {s1: (v1, rew_ref[s][s1]) for s1, v1 in v.items()}
             for s, v in tr.items()},
            self.gamma
        )

    def get_mdp_rep_for_rl_tabular(self) -> MDPRepForRLTabular:
        return MDPRepForRLTabular(
            state_action_dict=self.state_action_dict,
            terminal_states=self.terminal_states,
            state_reward_gen_dict=get_state_reward_gen_dict(
                self.rewards_refined,
                self.transitions
            ),
            gamma=self.gamma
        )


if __name__ == '__main__':
    data = {
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
    mdp_refined_obj = MDPRefined(data, 0.95)
    print(mdp_refined_obj.all_states)
    print(mdp_refined_obj.transitions)
    print(mdp_refined_obj.rewards)
    print(mdp_refined_obj.rewards_refined)
    terminal = mdp_refined_obj.get_terminal_states()
    print(terminal)
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
    mdp2_obj = MDPRefined(mdp_refined_data, 0.97)
    policy_data = {
        1: {'a': 0.4, 'b': 0.6},
        2: {'a': 0.7, 'c': 0.3},
        3: {'b': 1.0}
    }
    pol_obj = Policy(policy_data)
    mrp_refined_obj = mdp2_obj.get_mrp_refined(pol_obj)
    print(mrp_refined_obj.transitions)
    print(mrp_refined_obj.rewards_refined)
