from typing import TypeVar, Mapping, Set, Callable
from abc import abstractmethod
from algorithms.opt_base import OptBase
from processes.policy import Policy
from processes.det_policy import DetPolicy

S = TypeVar('S')
A = TypeVar('A')
VFType = Mapping[S, float]
QVFType = Mapping[S, Mapping[A, float]]


class TabularBase(OptBase):

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

    @abstractmethod
    def get_state_action_dict(self) -> Mapping[S, Set[A]]:
        pass

    def get_value_func(self, pol_func: Callable[[S], Callable[[A], float]])\
            -> Callable[[S], float]:
        pol = Policy({s: {a: pol_func(s)(a) for a in v}
                      for s, v in self.get_state_action_dict()})

        # noinspection PyShadowingNames
        def vf(state: S, pol=pol) -> float:
            return self.get_value_func_dict(pol)[state]

        return vf

    def get_act_value_func(self, pol_func: Callable[[S], Callable[[A], float]])\
            -> Callable[[S], Callable[[A], float]]:
        pol = Policy({s: {a: pol_func(s)(a) for a in v}
                      for s, v in self.get_state_action_dict()})

        # noinspection PyShadowingNames
        def qvf(state: S, pol=pol) -> Callable[[A], float]:

            # noinspection PyShadowingNames
            def inner_f(action: A, pol=pol, state=state) -> float:
                return self.get_act_value_func_dict(pol)[state][action]

            return inner_f

        return qvf

    def get_optimal_det_policy_func(self) -> Callable[[S], A]:
        return lambda s: self.get_optimal_det_policy().get_action_for_state(s)

