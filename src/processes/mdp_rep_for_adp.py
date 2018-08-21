from typing import Set, Callable, Sequence, Mapping
from processes.mdp_rep_for_adp_pg import MDPRepForADPPG
from utils.generic_typevars import S, A


class MDPRepForADP(MDPRepForADPPG):

    def __init__(
        self,
        state_action_func: Callable[[S], Set[A]],
        gamma: float,
        sample_states_gen_func: Callable[[int], Sequence[S]],
        reward_func: Callable[[S, A], float],
        transitions_func: Callable[[S, A], Mapping[S, float]]
    ) -> None:
        super().__init__(
            gamma=gamma,
            sample_states_gen_func=sample_states_gen_func,
            reward_func=reward_func,
            transitions_func=transitions_func
        )
        self.state_action_func: Callable[[S], Set[A]] = state_action_func


if __name__ == '__main__':
    print(0)

