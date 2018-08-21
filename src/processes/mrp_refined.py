from typing import Mapping, Tuple
from processes.mrp import MRP
from utils.gen_utils import zip_dict_of_tuple
import numpy as np
from utils.generic_typevars import S
from utils.standard_typevars import SSf, SSTff


class MRPRefined(MRP):

    def __init__(
        self,
        info: SSTff,
        gamma: float
    ) -> None:
        d1, d2, d3 = MRPRefined.split_info(info)
        super().__init__({k: (v, d3[k]) for k, v in d1.items()}, gamma)
        self.rewards_refined: SSf = d2

    @staticmethod
    def split_info(info: SSTff) -> Tuple[SSf, SSf, Mapping[S, float]]:
        d = {k: zip_dict_of_tuple(v) for k, v in info.items()}
        d1, d2 = zip_dict_of_tuple(d)
        d3 = {k: sum(np.prod(x) for x in v.values()) for k, v in info.items()}
        return d1, d2, d3


if __name__ == '__main__':
    data = {
        1: {1: (0.3, 9.2), 2: (0.6, 3.4), 3: (0.1, -0.3)},
        2: {1: (0.4, 0.0), 2: (0.2, 8.9), 3: (0.4, 3.5)},
        3: {3: (1.0, 0.0)}
    }
    mrp_refined_obj = MRPRefined(data, 0.95)
    print(mrp_refined_obj.trans_matrix)
    print(mrp_refined_obj.rewards_vec)


