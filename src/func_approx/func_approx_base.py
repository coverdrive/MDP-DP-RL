from typing import Sequence, Callable, Tuple, TypeVar, List, Set
from abc import ABC, abstractmethod
import numpy as np

X = TypeVar('X')
very_small_pos = 1e-6


class FuncApproxBase(ABC):

    def __init__(
        self,
        feature_funcs: Sequence[Callable[[X], float]],
        reglr_coeff: float,
        learning_rate: float,
        adam: bool,
        adam_decay1: float,
        adam_decay2: float,
        add_unit_feature: bool = True
    ):
        self.feature_funcs: Sequence[Callable[[X], float]] =\
            ([FuncApproxBase.get_unit_func] if add_unit_feature else []) + feature_funcs
        self.num_features = len(self.feature_funcs)
        self.reglr_coeff = reglr_coeff
        self.learning_rate = learning_rate
        self.adam = adam
        self.adam_decay1 = adam_decay1
        self.adam_decay2 = adam_decay2
        self.params: List[np.ndarray] = self.init_params()
        self.adam_caches: Tuple[List[np.ndarray], List[np.ndarray]]\
            = self.init_adam_caches()

    @staticmethod
    def get_unit_func(_: X) -> float:
        return 1.

    @staticmethod
    def get_identity_feature_funcs(n: int) -> List[Callable[[X], float]]:
        return [(lambda x, i=i: x[i]) for i in range(n)]

    @staticmethod
    def get_indicator_feature_funcs(values: Set[X])\
            -> List[Callable[[X], float]]:
        return [(lambda x, v=v: 1. if x == v else 0.) for v in values]

    def get_feature_vals(self, x_vals: X) -> np.ndarray:
        return np.array([f(x_vals) for f in self.feature_funcs])

    def get_feature_vals_pts(self, x_vals_seq: Sequence[X]) -> np.ndarray:
        return np.vstack(self.get_feature_vals(x) for x in x_vals_seq)

    @abstractmethod
    def init_params(self) -> Sequence[np.ndarray]:
        pass

    @abstractmethod
    def init_adam_caches(self)\
            -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        pass

    @abstractmethod
    def get_func_eval(self, x_vals: X) -> float:
        pass

    @abstractmethod
    def get_func_eval_pts(self, x_vals_seq: Sequence[X]) -> np.ndarray:
        pass

    @abstractmethod
    def get_sum_loss_gradient(
        self,
        x_vals_seq: Sequence[X],
        supervisory_seq: Sequence[float]
    ) -> Sequence[np.ndarray]:
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def get_sum_objective_gradient(
        self,
        x_vals_seq: Sequence[X],
        dObj_dOL: np.ndarray
    ) -> Sequence[np.ndarray]:
        pass

    @abstractmethod
    def get_el_tr_sum_loss_gradient(
        self,
        x_vals_seq: Sequence[X],
        supervisory_seq: Sequence[float],
        gamma_lambda: float
    ) -> Sequence[np.ndarray]:
        pass

    # noinspection PyPep8Naming
    @abstractmethod
    def get_el_tr_sum_objective_gradient(
        self,
        x_vals_seq: Sequence[X],
        dObj_dOL: np.ndarray,
        factors: np.ndarray,
        gamma_lambda: float
    ) -> Sequence[np.ndarray]:
        pass

    def update_params(
        self,
        x_vals_seq: Sequence[X],
        supervisory_seq: Sequence[float]
    ) -> None:
        avg_loss_gradient = [g / len(x_vals_seq) for g in
                             self.get_sum_loss_gradient(x_vals_seq, supervisory_seq)]
        self.update_params_from_gradient(avg_loss_gradient)

    def update_params_from_gradient(
        self,
        gradient: Sequence[np.ndarray]
    ) -> None:
        for l in range(len(self.params)):
            g = gradient[l] + self.reglr_coeff * self.params[l]
            if self.adam:
                self.adam_caches[0][l] = self.adam_decay1 * self.adam_caches[0][l] +\
                    (1 - self.adam_decay1) * g
                self.adam_caches[1][l] = self.adam_decay2 * self.adam_caches[1][l] +\
                    (1 - self.adam_decay2) * g ** 2
                self.params[l] -= self.learning_rate * self.adam_caches[0][l] /\
                    (np.sqrt(self.adam_caches[1][l]) + very_small_pos) *\
                    np.sqrt(1 - self.adam_decay2) / (1 - self.adam_decay1)
            else:
                self.params[l] -= self.learning_rate * g
