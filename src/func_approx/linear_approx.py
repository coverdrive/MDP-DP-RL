from typing import Sequence, Callable, Tuple, TypeVar
from func_approx.func_approx_base import FuncApproxBase
from func_approx.eligibility_traces import get_decay_toeplitz_matrix
from scipy.stats import norm
import numpy as np

X = TypeVar('X')


class LinearApprox(FuncApproxBase):

    def __init__(
        self,
        feature_funcs: Sequence[Callable[[X], float]],
        reglr_coeff: float = 0.,
        learning_rate: float = 0.1,
        adam: bool = True,
        adam_decay1: float = 0.9,
        adam_decay2: float = 0.99,
        add_unit_feature: bool = True
    ):
        super().__init__(
            feature_funcs,
            reglr_coeff,
            learning_rate,
            adam,
            adam_decay1,
            adam_decay2,
            add_unit_feature
        )

    def init_params(self) -> Sequence[np.ndarray]:
        return [np.zeros(self.num_features)]

    def init_adam_caches(self)\
            -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        return [np.zeros(self.num_features)],\
               [np.zeros(self.num_features)]

    def get_func_eval(self, x_vals: X):
        """
        This must return a float but lint is not happy, so removed the
        return type annotation
        """
        return np.dot(self.get_feature_vals(x_vals), self.params[0])

    def get_func_eval_pts(self, x_vals_seq: Sequence[X]) -> np.ndarray:
        return np.dot(
            self.get_feature_vals_pts(x_vals_seq),
            self.params[0]
        )

    def get_sum_loss_gradient(
        self,
        x_vals_seq: Sequence[X],
        supervisory_seq: Sequence[float]
    ) -> Sequence[np.ndarray]:
        # return [np.dot(self.get_func_eval_pts(x_vals_seq) - supervisory_seq,
        #               self.get_feature_vals_pts(x_vals_seq))]
        return [np.sum((self.get_func_eval(x) - supervisory_seq[i]) * self.get_feature_vals(x)
                       for i, x in enumerate(x_vals_seq))]

    # noinspection PyPep8Naming
    def get_sum_objective_gradient(
        self,
        x_vals_seq: Sequence[X],
        dObj_dOL: np.ndarray
    ) -> Sequence[np.ndarray]:
        return [dObj_dOL.dot(self.get_feature_vals_pts(x_vals_seq))]

    def get_el_tr_sum_loss_gradient(
        self,
        x_vals_seq: Sequence[X],
        supervisory_seq: Sequence[float],
        gamma_lambda: float
    ) -> Sequence[np.ndarray]:
        toeplitz_mat = get_decay_toeplitz_matrix(len(x_vals_seq), gamma_lambda)
        errors = self.get_func_eval_pts(x_vals_seq) - supervisory_seq
        func_grad = self.get_feature_vals_pts(x_vals_seq)
        return [errors.dot(toeplitz_mat.dot(func_grad))]

    # noinspection PyPep8Naming
    def get_el_tr_sum_objective_gradient(
        self,
        x_vals_seq: Sequence[X],
        dObj_dOL: np.ndarray,
        factors: np.ndarray,
        gamma_lambda: float
    ) -> Sequence[np.ndarray]:
        toep = get_decay_toeplitz_matrix(len(x_vals_seq), gamma_lambda)
        features = self.get_feature_vals_pts(x_vals_seq)
        return [factors.dot(toep.dot(np.diag(dObj_dOL).dot(features)))]


if __name__ == '__main__':
    la = LinearApprox(
        feature_funcs=FuncApproxBase.get_identity_feature_funcs(3),
        reglr_coeff=0.,
        learning_rate=0.1,
        adam=True,
        adam_decay1=0.9,
        adam_decay2=0.999,
        add_unit_feature=True
    )
    alpha = 2.0
    beta_1 = 10.0
    beta_2 = 4.0
    beta_3 = -6.0
    beta = (beta_1, beta_2, beta_3)
    x_pts = np.arange(-10.0, 10.0, 0.5)
    y_pts = np.arange(-10.0, 10.0, 0.5)
    z_pts = np.arange(-10.0, 10.0, 0.5)
    pts = [(x, y, z) for x in x_pts for y in y_pts for z in z_pts]

    # noinspection PyShadowingNames
    def superv_func(pt, alpha=alpha, beta=beta):
        return alpha + np.dot(beta, pt)

    n = norm(loc=0., scale=1.)
    superv_pts = [superv_func(r) + n.rvs(size=1)[0] for r in pts]
    # import matplotlib.pyplot as plt
    for _ in range(1000):
        print(la.params[0])
        la.update_params(pts, superv_pts)
        pred_pts = [la.get_func_eval(x) for x in pts]
        print(np.linalg.norm(np.array(pred_pts) - np.array(superv_pts)) /
              np.sqrt(len(superv_pts)))

