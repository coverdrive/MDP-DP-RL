from typing import TypeVar, Mapping, Set, Callable, Tuple
from processes.mdp_refined import MDPRefined
from processes.mp_funcs import get_state_reward_gen_dict
from processes.mp_funcs import get_rv_gen_func_single
from processes.mdp_rep_for_rl_fa import MDPRepForRLFA

S = TypeVar('S')
A = TypeVar('A')
Type1 = Mapping[S, Mapping[A, Callable[[], Tuple[S, float]]]]


class MDPRepForRLTabular(MDPRepForRLFA):

    def __init__(self, mdp_ref_obj: MDPRefined) -> None:
        self.state_action_dict: Mapping[S, Set[A]] = mdp_ref_obj.state_action_dict
        self.terminal_states: Set[S] = mdp_ref_obj.terminal_states
        self.state_reward_gen_dict: Type1 = get_state_reward_gen_dict(
            mdp_ref_obj.rewards_refined,
            mdp_ref_obj.transitions
        )
        super().__init__(
            state_action_func=lambda x: self.state_action_dict[x],
            gamma=mdp_ref_obj.gamma,
            terminal_state_func=lambda x: x in self.terminal_states,
            state_reward_gen_func=lambda x, y: self.state_reward_gen_dict[x][y](),
            init_state_gen=get_rv_gen_func_single(
                {s: 1. / len(self.state_action_dict) for s
                 in self.state_action_dict.keys()}
            ),
            init_state_action_gen=get_rv_gen_func_single(
                {(s, a): 1. / sum(len(v) for v
                                  in self.state_action_dict.values())
                 for s, v1 in self.state_action_dict.items() for a in v1}
            )
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
    this_gamma = 0.95
    mdp_refined_obj = MDPRefined(data, this_gamma)
    this_mdp_rep_for_rl = MDPRepForRLTabular(mdp_refined_obj)
    print(this_mdp_rep_for_rl.state_action_dict)
    print(this_mdp_rep_for_rl.terminal_states)
    print(this_mdp_rep_for_rl.state_reward_gen_dict)



