from typing import Mapping, TypeVar, Tuple
from utils.gen_utils import zip_dict_of_tuple
from processes.mdp import MDP
import numpy as np

S = TypeVar('S')
A = TypeVar('A')
Type1 = Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]]
Type2 = Mapping[S, Mapping[A, Mapping[S, float]]]


class MDPRefined(MDP):

    def __init__(
        self,
        info: Type1,
        gamma: float = 1.
    ) -> None:
        d1, d2, d3 = MDPRefined.split_info(info)
        super().__init__(
            {s: {a: (v1, d3[s][a]) for a, v1 in v.items()}
             for s, v in d1.items()},
            gamma
        )
        self.rewards_refined: Type2 = d2

    @staticmethod
    def split_info(info: Type1) -> Tuple[Type2, Type2,
                                         Mapping[S, Mapping[A, float]]]:
        c = {s: {a: zip_dict_of_tuple(v1) for a, v1 in v.items()}
             for s, v in info.items()}
        d = {k: zip_dict_of_tuple(v) for k, v in c.items()}
        d1, d2 = zip_dict_of_tuple(d)
        d3 = {s: {a: sum(np.prod(x) for x in v1.values())
                  for a, v1 in v.items()} for s, v in info.items()}
        return d1, d2, d3


if __name__ == '__main__':
    data = {
        1: {
            'a': {1: (0.3, 9.2), 2: (0.6, 4.5), 3: (0.1, 5.0)},
            'b': {2: (0.3, -0.5), 3: (0.7, 2.6)},
            'c': {1: (0.2, 4.8), 2: (0.4, -4.9), 3: (0.4, 0.0)}
        },
        2: {
            'a': {1: (0.3, 9.8), 2: (0.6, 6.7), 3: (0.1, 1.8)},
            'c': {1: (0.2, 4.8), 2: (0.4, 9.2), 3: (0.4, -8.2)}
        },
        3: {
            'a': {3: (1.0, 0.0)},
            'b': {3: (1.0, 0.0)}
        }
    }
    mdp_refined = MDPRefined(data, 0.95)
    print(mdp_refined.all_states)
    print(mdp_refined.transitions)
    print(mdp_refined.rewards)
    print(mdp_refined.rewards_refined)
    terminal = mdp_refined.get_terminal_states()
    print(terminal)
