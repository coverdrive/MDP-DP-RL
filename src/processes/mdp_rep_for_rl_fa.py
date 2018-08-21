from typing import Set, Callable, Tuple, Optional, Generic
from processes.mp_funcs import get_rv_gen_func_single
from utils.generic_typevars import S, A


class MDPRepForRLFA(Generic[S, A]):

    def __init__(
        self,
        state_action_func: Callable[[S], Set[A]],
        gamma: float,
        terminal_state_func: Callable[[S], bool],
        state_reward_gen_func: Callable[[S, A], Tuple[S, float]],
        init_state_gen: Callable[[], S],
        init_state_action_gen: Optional[Callable[[], Tuple[S, A]]]
    ) -> None:
        # noinspection PyShadowingNames
        def init_sa(
            init_state_gen=init_state_gen,
            state_action_func=state_action_func
        ) -> Tuple[S, A]:
            s = init_state_gen()
            actions = state_action_func(s)
            a = get_rv_gen_func_single({a: 1. / len(actions) for a in actions})()
            return s, a

        self.state_action_func: Callable[[S], Set[A]] = state_action_func
        self.gamma: float = gamma
        self.terminal_state_func: Callable[[S], bool] = terminal_state_func
        self.state_reward_gen_func: Callable[[S, A], Tuple[S, float]] = \
            state_reward_gen_func
        self.init_state_gen: Callable[[], S] = init_state_gen
        self.init_state_action_gen: Callable[[], Tuple[S, A]] =\
            (init_state_action_gen if init_state_action_gen is not None
             else init_sa)


if __name__ == '__main__':
    print(0)
