from typing import Mapping, TypeVar, Set, Generic, Sequence
from processes.mp_funcs import get_all_states, verify_mp, get_lean_transitions

S = TypeVar('S')
Transitions = Mapping[S, Mapping[S, float]]


class MP(Generic[S]):

    def __init__(
        self,
        tr: Transitions,
    ) -> None:
        if verify_mp(tr):
            self.all_states_list: Sequence[S] = list(get_all_states(tr))
            self.transitions: Transitions = {s: get_lean_transitions(v)
                                             for s, v in tr.items()}
        else:
            raise ValueError

    def get_sink_states(self) -> Set[S]:
        return {k for k, v in self.transitions.items()
                if len(v) == 1 and k in v.keys()}


if __name__ == '__main__':
    transitions = {
        1: {1: 0.3, 2: 0.6, 3: 0.1},
        2: {1: 0.4, 2: 0.2, 3: 0.4},
        3: {3: 1.0}
    }
    mp_obj = MP(transitions)
    print(mp_obj.transitions)
    print(mp_obj.all_states_list)
    print(mp_obj.get_sink_states())
