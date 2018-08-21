from typing import Mapping, Set, Generic, Sequence
from processes.mp_funcs import get_all_states, verify_mp, get_lean_transitions
import numpy as np
from scipy.linalg import eig
from utils.generic_typevars import S
from utils.standard_typevars import SSf


class MP(Generic[S]):

    def __init__(
        self,
        tr: SSf
    ) -> None:
        if verify_mp(tr):
            self.all_states_list: Sequence[S] = list(get_all_states(tr))
            self.transitions: SSf = {s: get_lean_transitions(v)
                                     for s, v in tr.items()}
        else:
            raise ValueError

    def get_sink_states(self) -> Set[S]:
        return {k for k, v in self.transitions.items()
                if len(v) == 1 and k in v.keys()}

    def get_stationary_distribution(self) -> Mapping[S, float]:
        sz = len(self.all_states_list)
        mat = np.zeros((sz, sz))
        for i, s1 in enumerate(self.all_states_list):
            for j, s2 in enumerate(self.all_states_list):
                mat[i, j] = self.transitions[s1].get(s2, 0.)

        eig_vals, eig_vecs = eig(mat.T)
        stat = np.array(
            eig_vecs[:, np.where(np.abs(eig_vals - 1.) < 1e-8)[0][0]].flat
        ).astype(float)
        norm_stat = stat / sum(stat)
        return {s: norm_stat[i] for i, s in enumerate(self.all_states_list)}


if __name__ == '__main__':
    transitions = {
        1: {1: 0.1, 2: 0.6, 3: 0.1, 4: 0.2},
        2: {1: 0.25, 2: 0.22, 3: 0.24, 4: 0.29},
        3: {1: 0.7, 2: 0.3},
        4: {1: 0.3, 2: 0.5, 3: 0.2}
    }
    mp_obj = MP(transitions)
    print(mp_obj.transitions)
    print(mp_obj.all_states_list)
    print(mp_obj.get_sink_states())
    stationary = mp_obj.get_stationary_distribution()
    print(stationary)
