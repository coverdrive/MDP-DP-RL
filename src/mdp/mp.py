from typing import Mapping, TypeVar
from mdp.mp_funcs import get_all_states, verify_transitions


class MP(object):

    S = TypeVar('S')

    def __init__(
        self,
        tr: Mapping[S, Mapping[S, float]],
    ) -> None:
        if verify_transitions(tr):
            self.transitions = tr
            self.all_states = get_all_states(tr)
        else:
            raise ValueError

    def get_terminal_states(self):
        return {k for k, v in self.transitions.items()
                if len(v) == 1 and k in v.keys()}


if __name__ == '__main__':
    transitions = {
        1: {1: 0.3, 2: 0.6, 3: 0.1},
        2: {1: 0.4, 2: 0.2, 3: 0.4},
        3: {3: 1.0}
    }
    mp = MP(transitions)
    print(mp.transitions)
    print(mp.get_terminal_states())
