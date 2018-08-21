from typing import Callable
from abc import ABC, abstractmethod
from utils.generic_typevars import S, A
from utils.standard_typevars import VFType, QFType, PolicyType


class OptBase(ABC):

    @abstractmethod
    def get_value_func(self, polf: PolicyType) -> VFType:
        pass

    @abstractmethod
    def get_act_value_func(self, polf: PolicyType) -> QFType:
        pass

    @abstractmethod
    def get_optimal_det_policy_func(self) -> Callable[[S], A]:
        pass

    # noinspection PyShadowingNames
    def get_optimal_value_func(self) -> VFType:
        pf = self.get_optimal_det_policy_func()
        return self.get_value_func(
            lambda s, pf=pf: lambda n, s=s, pf=pf: [pf(s)] * n
        )

    # noinspection PyShadowingNames
    def get_optimal_act_value_func(self) -> QFType:
        pf = self.get_optimal_det_policy_func()
        return self.get_act_value_func(
            lambda s, pf=pf: lambda n, s=s, pf=pf: [pf(s)] * n
        )

