from typing import Mapping, TypeVar, Set, Tuple, Generic
from utils.gen_utils import zip_dict_of_tuple, is_approx_eq
from processes.mp_funcs import get_all_states, verify_mdp

S = TypeVar('S')
A = TypeVar('A')


class MDP(Generic[S, A]):

    def __init__(
        self,
        info: Mapping[S, Mapping[A, Tuple[Mapping[S, float], float]]],
    ) -> None:
        if verify_mdp(info):
            d = {k: zip_dict_of_tuple(v) for k, v in info.items()}
            d1, d2 = zip_dict_of_tuple(d)
            self.all_states: Set[S] = get_all_states(info)
            self.transitions: Mapping[S, Mapping[A, Mapping[S, float]]] = d1
            self.rewards: Mapping[S, Mapping[A, float]] = d2
        else:
            raise ValueError

    def get_sink_states(self) -> Set[S]:
        return {k for k, v in self.transitions.items() if
                all(len(v1) == 1 and k in v1.keys() for _, v1 in v.items())
                }

    def get_terminal_states(self) -> Set[S]:
        sink = self.get_sink_states()
        return {s for s in sink if
                all(is_approx_eq(r, 0.0) for _, r in self.rewards[s].items())}


if __name__ == '__main__':
    data = {
        1: {
            'a': ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
            'b': ({2: 0.3, 3: 0.7}, 2.8),
            'c': ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2)
        },
        2: {
            'a': ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
            'c': ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2)
        },
        3: {
            'a': ({3: 1.0}, 0.0),
            'b': ({3: 1.0}, 0.0)
        }
    }
    mdp = MDP(data)
    print(mdp.all_states)
    print(mdp.transitions)
    print(mdp.rewards)
    terminal = mdp.get_terminal_states()
    print(terminal)
