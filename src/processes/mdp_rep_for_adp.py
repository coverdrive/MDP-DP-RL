from typing import Set, Callable, Sequence, Mapping, Generic
from utils.generic_typevars import S, A


class MDPRepForADP(Generic[S, A]):

    def __init__(
        self,
        state_action_func: Callable[[S], Set[A]],
        gamma: float,
        sample_states_gen_func: Callable[[int], Sequence[S]],
        reward_func: Callable[[S, A], float],
        transitions_func: Callable[[S, A], Mapping[S, float]]
    ) -> None:
        self.state_action_func: Callable[[S], Set[A]] = state_action_func
        self.gamma: float = gamma
        self.sample_states_gen_func: Callable[[int], Sequence[S]] = \
            sample_states_gen_func
        self.reward_func: Callable[[S, A], float] = reward_func
        self.transitions_func: Callable[[S, A], Mapping[S, float]] = \
            transitions_func


if __name__ == '__main__':
    print(0)

