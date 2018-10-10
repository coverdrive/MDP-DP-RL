from typing import Callable, Sequence, NamedTuple, Optional, Tuple
from func_approx.dnn_spec import DNNSpec
from func_approx.func_approx_base import FuncApproxBase
from func_approx.linear_approx import LinearApprox
from func_approx.dnn import DNN
from utils.generic_typevars import S, A


class FuncApproxSpec(NamedTuple):
    state_feature_funcs: Sequence[Callable[[S], float]]
    sa_feature_funcs: Sequence[Callable[[Tuple[S, A]], float]]
    dnn_spec: Optional[DNNSpec]
    reglr_coeff: float = 0.
    learning_rate: float = 0.1
    adam_params: Tuple[bool, float, float] = (True, 0.9, 0.99)
    add_unit_feature: bool = True

    def get_vf_func_approx_obj(self) -> FuncApproxBase:
        if self.dnn_spec is None:
            ret = LinearApprox(
                feature_funcs=self.state_feature_funcs,
                reglr_coeff=self.reglr_coeff,
                learning_rate=self.learning_rate,
                adam=self.adam_params[0],
                adam_decay1=self.adam_params[1],
                adam_decay2=self.adam_params[2],
                add_unit_feature=self.add_unit_feature
            )
        else:
            ret = DNN(
                feature_funcs=self.state_feature_funcs,
                dnn_obj=self.dnn_spec,
                reglr_coeff=self.reglr_coeff,
                learning_rate=self.learning_rate,
                adam=self.adam_params[0],
                adam_decay1=self.adam_params[1],
                adam_decay2=self.adam_params[2],
                add_unit_feature=self.add_unit_feature
            )
        return ret

    def get_qvf_func_approx_obj(self) -> FuncApproxBase:
        if self.dnn_spec is None:
            ret = LinearApprox(
                feature_funcs=self.sa_feature_funcs,
                reglr_coeff=self.reglr_coeff,
                learning_rate=self.learning_rate,
                adam=self.adam_params[0],
                adam_decay1=self.adam_params[1],
                adam_decay2=self.adam_params[2],
                add_unit_feature=self.add_unit_feature
            )
        else:
            ret = DNN(
                feature_funcs=self.sa_feature_funcs,
                dnn_obj=self.dnn_spec,
                reglr_coeff=self.reglr_coeff,
                learning_rate=self.learning_rate,
                adam=self.adam_params[0],
                adam_decay1=self.adam_params[1],
                adam_decay2=self.adam_params[2],
                add_unit_feature=self.add_unit_feature
            )
        return ret
