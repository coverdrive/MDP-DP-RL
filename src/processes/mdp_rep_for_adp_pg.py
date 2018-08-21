from typing import Callable, Sequence, Generic, Mapping
from utils.generic_typevars import S, A


class MDPRepForADPPG(Generic[S, A]):

    def __init__(
        self,
        gamma: float,
        sample_states_gen_func: Callable[[int], Sequence[S]],
        reward_func: Callable[[S, A], float],
        transitions_func: Callable[[S, A], Mapping[S, float]],

    ) -> None:
        self.gamma: float = gamma
        self.sample_states_gen_func: Callable[[int], Sequence[S]] =\
            sample_states_gen_func
        self.reward_func: Callable[[S, A], float] = reward_func
        self.transitions_func: Callable[[S, A], Mapping[S, float]] =\
            transitions_func


if __name__ == '__main__':
    print(0)
