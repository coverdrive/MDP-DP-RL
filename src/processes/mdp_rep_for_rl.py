from typing import TypeVar, Callable, Tuple, Generic

S = TypeVar('S')
A = TypeVar('A')


class MDPRepForRL(Generic[S, A]):

    def __init__(
        self,
        gamma: float,
        terminal_state_func: Callable[[S], bool],
        state_reward_gen_func: Callable[[S, A], Tuple[S, float]],
        init_state_gen: Callable[[], S],
        init_state_action_gen: Callable[[], Tuple[S, A]],
        max_a_func: Callable[[S, Callable[[A], float]], A]
    ) -> None:
        self.gamma: float = gamma
        self.terminal_state_func: Callable[[S], bool] = terminal_state_func
        self.state_reward_gen_func: Callable[[S, A], Tuple[S, float]] =\
            state_reward_gen_func
        self.init_state_gen: Callable[[], S] = init_state_gen
        self.init_state_action_gen: Callable[[], Tuple[S, A]] = init_state_action_gen
        self.max_a_func: Callable[[S, Callable[[A], float]], A] = max_a_func


if __name__ == '__main__':
    print(0)
