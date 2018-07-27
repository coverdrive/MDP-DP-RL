from typing import TypeVar, Mapping
from abc import ABC, abstractmethod
from processes.policy import Policy
from processes.det_policy import DetPolicy

S = TypeVar('S')
A = TypeVar('A')
VFType = Mapping[S, float]
QVFType = Mapping[S, Mapping[A, float]]


class OptBase(ABC):

    @abstractmethod
    def get_init_policy(self) -> Policy:
        pass

    @abstractmethod
    def get_value_func_dict(self, pol: Policy) -> VFType:
        pass

    @abstractmethod
    def get_act_value_func_dict(self, pol: Policy) -> QVFType:
        pass

    @abstractmethod
    def get_optimal_det_policy(self) -> DetPolicy:
        pass

