from typing import Callable, Sequence, Generic, Tuple
from utils.generic_typevars import S, A


class MDPRepForADPPG(Generic[S, A]):

    def __init__(
        self,
        gamma: float,
        init_states_gen_func: Callable[[int], Sequence[S]],
        state_reward_gen_func: Callable[[S, A, int], Sequence[Tuple[S, float]]],
        # reward_func: Callable[[S, A], float],
        # transitions_func: Callable[[S, A], Mapping[S, float]],
        terminal_state_func: Callable[[S], bool],
    ) -> None:
        self.gamma: float = gamma
        self.init_states_gen_func: Callable[[int], Sequence[S]] = \
            init_states_gen_func
        self.state_reward_gen_func: Callable[[S, A, int], Sequence[Tuple[S, float]]] =\
            state_reward_gen_func
        # self.reward_func: Callable[[S, A], float] = reward_func
        # self.transitions_func: Callable[[S, A], Mapping[S, float]] = \
        #     transitions_func
        self.terminal_state_func = terminal_state_func


if __name__ == '__main__':
    print(0)
