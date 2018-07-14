from typing import Mapping, TypeVar, Set, Tuple
from processes.mp import MP
from utils.gen_utils import zip_dict_of_tuple, is_approx_eq
import numpy as np


class MRP(MP):
    S = TypeVar('S')

    def __init__(
        self,
        info: Mapping[S, Tuple[Mapping[S, float], float]],
    ):
        d1, d2 = zip_dict_of_tuple(info)
        super().__init__(d1),
        self.rewards = d2
        self.rewards_vec: np.ndarray = self.get_rewards_vec()

    def get_rewards_vec(self) -> np.ndarray:
        return np.array([self.rewards[s] for s in self.all_states])

    def get_terminal_states(self) -> Set[S]:
        sink = self.get_sink_states()
        return {s for s in sink if is_approx_eq(self.rewards[s], 0.0)}


if __name__ == '__main__':
    data = {
        1: ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
        2: ({1: 0.4, 2: 0.2, 3: 0.4}, 4.0),
        3: ({3: 1.0}, 0.0)
    }
    mrp = MRP(data)
    print(mrp.trans_matrix)
    print(mrp.rewards_vec)
    terminal = mrp.get_terminal_states()
    print(terminal)
