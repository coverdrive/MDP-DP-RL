from typing import TypeVar, Set, Callable, Sequence, Generic, Mapping

S = TypeVar('S')
A = TypeVar('A')


class MDPRepForADPPG(Generic[S, A]):

    def __init__(
        self,
        state_action_func: Callable[[S], Set[A]],
        gamma: float,
        sample_states_gen_func: Callable[[int], Sequence[S]],
        reward_func: Callable[[S, A], float],
        transitions_func: Callable[[S, A], Mapping[S, float]],
        score_func: Callable[[A, Sequence[float]], Sequence[float]],
        sample_actions_gen_func: Callable[[Sequence[float], int], Sequence[A]]

    ) -> None:
        self.state_action_func: Callable[[S], Set[A]] = state_action_func
        self.gamma: float = gamma
        self.sample_states_gen_func: Callable[[int], Sequence[S]] =\
            sample_states_gen_func
        self.reward_func: Callable[[S, A], float] = reward_func
        self.transitions_func: Callable[[S, A], Mapping[S, float]] =\
            transitions_func
        self.score_func: Callable[[A, Sequence[float]], Sequence[float]] =\
            score_func
        self.sample_actions_gen_func: Callable[[Sequence[float], int], Sequence[A]] =\
            sample_actions_gen_func


if __name__ == '__main__':
    print(0)
