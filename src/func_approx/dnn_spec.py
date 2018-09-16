from typing import Callable, Sequence, NamedTuple
import numpy as np


class DNNSpec(NamedTuple):

    neurons: Sequence[int]
    hidden_activation: Callable[[np.ndarray], np.ndarray]
    hidden_activation_deriv: Callable[[np.ndarray], np.ndarray]
    output_activation: Callable[[np.ndarray], np.ndarray]
    output_activation_deriv: Callable[[np.ndarray], np.ndarray]

    SMALL_POS = 1e-8

    @staticmethod
    def fexp(arg: np.ndarray) -> np.ndarray:
        return np.vectorize(
            lambda x: max(DNNSpec.SMALL_POS, x)
        )(np.exp(arg))

    @staticmethod
    def relu(arg: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda x: x if x > 0. else 0.)(arg)

    @staticmethod
    def relu_deriv(res: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda x: 1. if x > 0. else 0.)(res)

    @staticmethod
    def identity(arg: np.ndarray) -> np.ndarray:
        return arg

    @staticmethod
    def identity_deriv(res: np.ndarray) -> np.ndarray:
        return np.ones_like(res)

    @staticmethod
    def sigmoid(arg: np.ndarray) -> np.ndarray:
        return np.vectorize(
            lambda x: max(
                DNNSpec.SMALL_POS,
                1. / (1. + max(DNNSpec.SMALL_POS, np.exp(-x)))
            )
        )(arg)

    @staticmethod
    def sigmoid_deriv(res: np.ndarray) -> np.ndarray:
        return res * (1. - res)

    @staticmethod
    def softplus(arg: np.ndarray) -> np.ndarray:
        return np.log(1. + DNNSpec.fexp(arg))

    @staticmethod
    def softplus_deriv(res: np.ndarray) -> np.ndarray:
        return 1. + DNNSpec.fexp(-res)

    @staticmethod
    def log_squish(arg: np.ndarray) -> np.ndarray:
        return np.sign(arg) * np.log(1 + np.abs(arg))

    @staticmethod
    def log_squish_deriv(res: np.ndarray) -> np.ndarray:
        return DNNSpec.fexp(-np.abs(res))

    @staticmethod
    def pos_log_squish(arg: np.ndarray) -> np.ndarray:
        return np.vectorize(
            lambda x: 1. + np.log(1. + x) if x > 0. else DNNSpec.fexp(x)
        )(arg)

    @staticmethod
    def pos_log_squish_deriv(res: np.ndarray) -> np.ndarray:
        return np.vectorize(
            lambda x: DNNSpec.fexp(1. - x) if x > 1. else x
        )(res)
