from typing import Callable, Generic, Tuple
from utils.generic_typevars import S, A


class MDPRepForRLPG(Generic[S, A]):

    def __init__(
        self,
        gamma: float,
        init_state_gen_func: Callable[[], S],
        state_reward_gen_func: Callable[[S, A], Tuple[S, float]],
        terminal_state_func: Callable[[S], bool],
    ) -> None:
        self.gamma: float = gamma
        self.init_state_gen_func: Callable[[], S] = init_state_gen_func
        self.state_reward_gen_func: Callable[[S, A], Tuple[S, float]] = \
            state_reward_gen_func
        self.terminal_state_func = terminal_state_func


if __name__ == '__main__':
    print(0)
