from typing import Sequence, Callable
from scipy.linalg import toeplitz
import numpy as np


def get_decay_toeplitz_matrix(
    size: int,
    decay_param: float
) -> np.ndarray:
    return toeplitz(
        np.power(decay_param, np.arange(size)),
        np.insert(np.zeros(size - 1), 0, 1.)
    )


# noinspection PyPep8Naming
def get_generalized_back_prop(
    dnn_params: Sequence[np.ndarray],
    fwd_prop: Sequence[np.ndarray],
    dObj_dOL: np.ndarray,
    factors: np.ndarray,
    decay_param: float,
    hidden_activation_deriv: Callable[[np.ndarray], np.ndarray],
    output_activation_deriv: Callable[[np.ndarray], np.ndarray]
) -> Sequence[np.ndarray]:
    """
    :param dnn_params: list (of length L+1) of (|O_L| x |I_L| + 1) 2-D array
    :param fwd_prop: list (of length L+2), the first (L+1)elements are
     n x (|I_l| + 1) 2-D arrays representing the inputs to the (L+1) layers,
     and the last element is a n x 1 2-D array
    :param dObj_dOL: 1-D array of length n
    :param factors: 1-D array of length n
    :param decay_param: [0,1] float representing decay in time
    :param hidden_activation_deriv: function representing the derivative
    of the hidden layer activation function (expressed as a function of the
    output of the hidden layer activation function).
    :param output_activation_deriv: function representing the derivative
    of the output layer activation function (expressed as a function of the
    output of the output layer activation function).
    L is the number of hidden layers, n is the number of points
    :return: list (of length L+1) of |O_l| x (|I_l| + 1) 2-D arrays,
             i.e., same as the type of self.params
    """
    output = fwd_prop[-1][:, 0]
    layer_inputs = fwd_prop[:-1]
    # deriv initialized to 1 x n  = |O_L| x n 2-D array
    deriv = (dObj_dOL * output_activation_deriv(output)).reshape(1, -1)
    decay_matrix = get_decay_toeplitz_matrix(len(factors), decay_param)
    back_prop = []
    for l in reversed(range(len(dnn_params))):
        # layer l gradient is factors tensordot (decay_matrix tensordot
        # (deriv_l einsum layer_inputs_l) which is of dimension
        # n tensordot ((n x n) tensordot ((|O_l| x n) einsum (n x (|I_l| + 1)))
        # = n tensordot ((n x n) tensordot (n x |O_l| x (|I_l| + 1)))
        # = n tensordot (n x |O_l| x (|I_l| + 1)) = |O_l| x (|I_l| + 1)
        t1 = np.einsum('ij,jk->jik', deriv, layer_inputs[l])
        if decay_param != 0:
            t2 = np.tensordot(decay_matrix, t1, axes=1)
        else:
            t2 = t1
        t3 = np.tensordot(factors, t2, axes=1)
        back_prop.append(t3)
        # deriv_l is dnn_params_{l+1}^T dot deriv_{l+1} haddamard g'(S_l), which is
        # ((|I_{l+1}| + 1) x |O_{l+1}|) dot (|O_{l+1}| x n) haddamard
        # ((|I_{l+1}| + 1) x n) --- g'(S_L) is expressed as hidden layer
        # activation derivative as a function of O_l (=I_{l+1}).
        # (notice first row  of the result is removed after this calculation).
        # So, deriv_l has dimension |I_{l+1}| x n = |O_l| x n
        deriv = (np.dot(dnn_params[l].T, deriv) *
                 hidden_activation_deriv(layer_inputs[l].T))[1:]
    return back_prop[::-1]
