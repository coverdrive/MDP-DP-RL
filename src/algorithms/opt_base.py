from typing import TypeVar, Callable, Set
from abc import ABC, abstractmethod

S = TypeVar('S')
A = TypeVar('A')
Type1 = Callable[[S], float]
Type2 = Callable[[S], Callable[[A], float]]


class OptBase(ABC):

    @abstractmethod
    def get_value_func(self, pol_func: Type2) -> Type1:
        pass

    @abstractmethod
    def get_act_value_func(self, pol_func: Type2) -> Type2:
        pass

    @abstractmethod
    def get_optimal_det_policy_func(self) -> Callable[[S], A]:
        pass

    def get_optimal_value_func(self) -> Type1:
        pf = self.get_optimal_det_policy_func()
        return self.get_value_func(
            lambda s, pf=pf: lambda a, s=s, pf=pf: 1. if a == pf(s) else 0.
        )

    def get_optimal_act_value_func(self) -> Type1:
        pf = self.get_optimal_det_policy_func()
        return self.get_act_value_func(
            lambda s, pf=pf: lambda a, s=s, pf=pf: 1. if a == pf(s) else 0.
        )

