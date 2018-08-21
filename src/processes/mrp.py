from typing import Mapping, Set, Sequence
from processes.mp import MP
from utils.gen_utils import zip_dict_of_tuple, is_approx_eq
import numpy as np
from utils.generic_typevars import S
from utils.standard_typevars import STSff


class MRP(MP):

    def __init__(
        self,
        info: STSff,
        gamma: float
    ):
        d1, d2 = zip_dict_of_tuple(info)
        super().__init__(d1)
        self.gamma: float = gamma
        self.rewards: Mapping[S, float] = d2
        self.terminal_states = self.get_terminal_states()
        self.nt_states_list: Sequence[S] = self.get_nt_states_list()
        self.trans_matrix: np.ndarray = self.get_trans_matrix()
        self.rewards_vec: np.ndarray = self.get_rewards_vec()

    def get_terminal_states(self) -> Set[S]:
        sink = self.get_sink_states()
        return {s for s in sink if is_approx_eq(self.rewards[s], 0.0)}

    def get_nt_states_list(self) -> Sequence[S]:
        return [s for s in self.all_states_list
                if s not in self.terminal_states]

    def get_trans_matrix(self) -> np.ndarray:
        """
        This transition matrix is only for the non-terminal states
        """
        n = len(self.nt_states_list)
        m = np.zeros((n, n))
        for i in range(n):
            for s, d in self.transitions[self.nt_states_list[i]].items():
                if s in self.nt_states_list:
                    m[i, self.nt_states_list.index(s)] = d
        return m

    def get_rewards_vec(self) -> np.ndarray:
        """
        This rewards vec is only for the non-terminal states
        """
        return np.array([self.rewards[s] for s in self.nt_states_list])

    def get_value_func_vec(self) -> np.ndarray:
        """
        This value func vec is only for the non-terminal states
        """
        return np.linalg.inv(
            np.eye(len(self.nt_states_list)) - self.gamma * self.trans_matrix
        ).dot(self.rewards_vec)


if __name__ == '__main__':
    data = {
        1: ({1: 0.6, 2: 0.3, 3: 0.1}, 7.0),
        2: ({1: 0.1, 2: 0.2, 3: 0.7}, 10.0),
        3: ({3: 1.0}, 0.0)
    }
    mrp_obj = MRP(data, 1.0)
    print(mrp_obj.trans_matrix)
    print(mrp_obj.rewards_vec)
    terminal = mrp_obj.get_terminal_states()
    print(terminal)
    value_func_vec = mrp_obj.get_value_func_vec()
    print(value_func_vec)
