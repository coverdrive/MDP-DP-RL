from typing import Mapping, Set, Callable, Tuple
from processes.mp_funcs import get_rv_gen_func_single
from processes.mdp_rep_for_rl_fa import MDPRepForRLFA
from utils.generic_typevars import S, A

Type1 = Mapping[S, Mapping[A, Callable[[], Tuple[S, float]]]]


class MDPRepForRLTabular(MDPRepForRLFA):

    def __init__(
        self,
        state_action_dict: Mapping[S, Set[A]],
        terminal_states: Set[S],
        state_reward_gen_dict: Type1,
        gamma: float
    ) -> None:
        self.state_action_dict: Mapping[S, Set[A]] = state_action_dict
        self.terminal_states: Set[S] = terminal_states
        self.state_reward_gen_dict: Type1 = state_reward_gen_dict
        super().__init__(
            state_action_func=lambda x: self.state_action_dict[x],
            gamma=gamma,
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
    print(0)
