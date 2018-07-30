from typing import Sequence, Callable, Tuple, TypeVar, Union
from func_approx.func_approx_base import FuncApproxBase
from func_approx.dnn_spec import DNNSpec
from scipy.stats import norm
import numpy as np

X = TypeVar('X')
sci_type = Union[float, np.ndarray]
very_small_pos = 1e-6


class DNN(FuncApproxBase):

    def __init__(
        self,
        feature_funcs: Sequence[Callable[[X], float]],
        dnn_obj: DNNSpec,
        reglr_coeff: float,
        learning_rate: float,
        adam: bool,
        adam_decay1: float,
        adam_decay2: float,
    ):
        super().__init__(
            feature_funcs,
            reglr_coeff,
            learning_rate,
            adam,
            adam_decay1,
            adam_decay2
        )
        self.neurons: Sequence[int] = dnn_obj.neurons
        self.hidden_activation: Callable[[sci_type], sci_type]\
            = dnn_obj.hidden_activation
        self.hidden_activation_deriv: Callable[[sci_type], sci_type]\
            = dnn_obj.hidden_activation_deriv
        self.output_activation: Callable[[sci_type], sci_type]\
            = dnn_obj.output_activation
        self.output_activation_deriv: Callable[[sci_type], sci_type]\
            = dnn_obj.output_activation_deriv

    def init_params(self) -> Sequence[np.ndarray]:
        """
        These are Xavier input parameters
        """
        inp_size = self.num_features + 1
        params = []
        for layer_neurons in self.neurons:
            mat = np.random.rand(layer_neurons, inp_size) / np.sqrt(inp_size)
            params.append(mat)
            inp_size = layer_neurons + 1
        params.append(np.random.randn(1, inp_size) / np.sqrt(inp_size))
        return params

    def init_adam_caches(self)\
            -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        return [np.zeros_like(p) for p in self.params],\
               [np.zeros_like(p) for p in self.params]

    def get_forward_prop(self, x_vals: X) -> Sequence[sci_type]:
        inp = self.get_feature_vals(x_vals)
        outputs = [inp]
        for this_params in self.params[:-1]:
            out = self.hidden_activation(np.dot(this_params, inp))
            inp = np.insert(out, 0, 1.)
            outputs.append(inp)
        outputs.append(self.output_activation(np.dot(self.params[-1], inp)[0])
                       + very_small_pos)
        return outputs

    def get_func_eval(self, x_vals: X) -> float:
        """
        This must return a float but lint is not happy, so removed the
        return type annotation
        """
        return self.get_forward_prop(x_vals)[-1]

    def get_back_prop(self, fwd_prop: Sequence[np.ndarray]) -> :

    def get_gradient(
            self,
            x_vals_seq: Sequence[X],
            supervisory_seq: Sequence[float]
    ) -> Sequence[np.ndarray]:


if __name__ == '__main__':
    this_dnn_obj = DNNSpec(
        neurons=[2, 3],
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda x: x,
        output_activation=lambda x: x,
        output_activation_deriv=lambda x: x,
    )
    nn = DNN(
        feature_funcs=FuncApproxBase.get_identity_feature_funcs(3),
        dnn_obj=this_dnn_obj,
        reglr_coeff=0.,
        learning_rate=0.1,
        adam=True,
        adam_decay1=0.9,
        adam_decay2=0.999
    )
    init_eval = nn.get_func_eval((2.0, 3.0, -4.0))
    print(init_eval)

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
    def y_func(pt, alpha=alpha, beta=beta):
        return alpha + np.dot(beta, pt)

    n = norm(loc=0., scale=10.)
    y_pts = [y_func(r) for r in pts] + n.rvs(size=len(pts))
    for _ in range(1000):
        print(nn.params)
        nn.update_params(pts, y_pts)
        y_pred_pts = [nn.get_func_eval(x) for x in pts]
        print(np.linalg.norm(np.array(y_pred_pts) - np.array(y_pts)) / len(y_pts))
