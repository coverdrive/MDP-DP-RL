from typing import TypeVar, Set, Callable, Tuple, Optional
from processes.mp_funcs import get_rv_gen_func_single
from processes.mdp_rep_for_rl import MDPRepForRL
from operator import itemgetter

S = TypeVar('S')
A = TypeVar('A')


class MDPRepForRLFiniteA(MDPRepForRL):

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

        def max_action(
            state: S,
            action_to_reward: Callable[[A], float]
        ) -> A:
            return max(
                [(y, action_to_reward(y)) for y in state_action_func(state)],
                key=itemgetter(1)
            )[0]

        super().__init__(
            gamma=gamma,
            terminal_state_func=terminal_state_func,
            state_reward_gen_func=state_reward_gen_func,
            init_state_gen=init_state_gen,
            init_state_action_gen=(init_state_action_gen if
                                   init_state_action_gen is not None else
                                   init_sa),
            max_a_func=max_action
        )
        self.state_action_func: Callable[[S], Set[A]] = state_action_func


if __name__ == '__main__':
    print(0)
