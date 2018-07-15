from typing import Mapping, TypeVar, Set, Tuple
from processes.mp import MP
from utils.gen_utils import zip_dict_of_tuple, is_approx_eq
import numpy as np

S = TypeVar('S')


class MRP(MP):

    def __init__(
        self,
        info: Mapping[S, Tuple[Mapping[S, float], float]],
        gamma: float = 1.
    ):
        d1, d2 = zip_dict_of_tuple(info)
        super().__init__(d1)
        self.rewards = d2
        self.rewards_vec: np.ndarray = self.get_rewards_vec()
        self.gamma = gamma

    def get_rewards_vec(self) -> np.ndarray:
        return np.array([self.rewards[s] for s in self.all_states])

    def get_terminal_states(self) -> Set[S]:
        sink = self.get_sink_states()
        return {s for s in sink if is_approx_eq(self.rewards[s], 0.0)}

    def get_value_func_vec(self) -> np.ndarray:
        return np.linalg.inv(
            np.eye(len(self.all_states)) - self.gamma * self.trans_matrix
        ).dot(self.rewards_vec)


if __name__ == '__main__':
    data = {
        1: ({1: 0.6, 2: 0.3, 3: 0.1}, 7.0),
        2: ({1: 0.1, 2: 0.2, 3: 0.7}, 10.0),
        3: ({3: 1.0}, 0.0)
    }
    mrp = MRP(data, 0.99)
    print(mrp.trans_matrix)
    print(mrp.rewards_vec)
    terminal = mrp.get_terminal_states()
    print(terminal)
    value_func_vec = mrp.get_value_func_vec()
    print(value_func_vec)
