from typing import Mapping, TypeVar, Set, Generic, Sequence
from processes.mp_funcs import get_all_states, verify_mp
import numpy as np

S = TypeVar('S')
Transitions = Mapping[S, Mapping[S, float]]


class MP(Generic[S]):

    def __init__(
        self,
        tr: Transitions,
    ) -> None:
        if verify_mp(tr):
            self.all_states: Sequence[S] = list(get_all_states(tr))
            self.transitions: Transitions = tr
            self.trans_matrix: np.ndarray = self.get_trans_matrix()
        else:
            raise ValueError

    def get_sink_states(self) -> Set[S]:
        return {k for k, v in self.transitions.items()
                if len(v) == 1 and k in v.keys()}

    def get_trans_matrix(self) -> np.ndarray:
        n = len(self.all_states)
        m = np.zeros((n, n))
        for i in range(n):
            for s, d in self.transitions[self.all_states[i]].items():
                m[i, self.all_states.index(s)] = d
        return m


if __name__ == '__main__':
    transitions = {
        1: {1: 0.3, 2: 0.6, 3: 0.1},
        2: {1: 0.4, 2: 0.2, 3: 0.4},
        3: {3: 1.0}
    }
    mp = MP(transitions)
    print(mp.transitions)
    print(mp.all_states)
    print(mp.trans_matrix)
    print(mp.get_sink_states())
