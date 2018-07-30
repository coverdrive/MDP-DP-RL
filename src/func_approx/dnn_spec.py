from typing import Callable, Sequence, NamedTuple, Union
import numpy as np

sci_type = Union[float, np.ndarray]

class DNNSpec(NamedTuple):
    neurons: Sequence[int]
    hidden_activation: Callable[[sci_type], sci_type]
    hidden_activation_deriv: Callable[[sci_type], sci_type]
    output_activation: Callable[[sci_type], sci_type]
    output_activation_deriv: Callable[[sci_type], sci_type]
