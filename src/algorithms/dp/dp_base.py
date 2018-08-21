from typing import Mapping, Set
from abc import abstractmethod
from algorithms.tabular_base import TabularBase
from processes.policy import Policy
from processes.det_policy import DetPolicy
from processes.mdp import MDP
from operator import itemgetter
from algorithms.helper_funcs import get_uniform_policy
from algorithms.helper_funcs import get_det_policy_from_qf_dict
from utils.generic_typevars import S, A
from utils.standard_typevars import VFDictType, QFDictType


class DPBase(TabularBase):

    def __init__(self, mdp_obj: MDP, tol: float) -> None:
        self.mdp_obj: MDP = mdp_obj
        self.tol = tol

    def get_state_action_dict(self) -> Mapping[S, Set[A]]:
        return self.mdp_obj.state_action_dict

    def get_init_policy(self) -> Policy:
        return get_uniform_policy(self.mdp_obj.state_action_dict)

    @abstractmethod
    def get_value_func_dict(self, pol: Policy) -> VFDictType:
        pass

    def get_improved_det_policy(self, pol: Policy) -> DetPolicy:
        return get_det_policy_from_qf_dict(self.get_act_value_func_dict(pol))

    def get_act_value_func_dict(self, pol: Policy) -> QFDictType:
        v_dict = self.get_value_func_dict(pol)
        mo = self.mdp_obj
        return {s: {a: r + mo.gamma *
                sum(p * v_dict[s1] for s1, p in
                    mo.transitions[s][a].items()) for a, r in v.items()}
                for s, v in mo.rewards.items()}

    def get_optimal_policy_pi(self) -> DetPolicy:
        pol = self.get_init_policy()
        vf = self.get_value_func_dict(pol)
        epsilon = self.tol * 1e4
        while epsilon >= self.tol:
            pol = self.get_improved_det_policy(pol)
            new_vf = self.get_value_func_dict(pol)
            epsilon = max(abs(new_vf[s] - v) for s, v in vf.items())
            vf = new_vf
        return pol

    def get_optimal_policy_vi(self) -> DetPolicy:
        vf = {s: 0. for s in self.mdp_obj.all_states}
        epsilon = self.tol * 1e4
        mo = self.mdp_obj
        while epsilon >= self.tol:
            new_vf = {s: max(r + mo.gamma * sum(p * vf[s1] for s1, p in
                                                mo.transitions[s][a].items())
                             for a, r in v.items())
                      for s, v in mo.rewards.items()}
            epsilon = max(abs(new_vf[s] - v) for s, v in vf.items())
            vf = new_vf
        pol = DetPolicy({s: max(
            [(a, r + mo.gamma * sum(p * vf[s1]
                                    for s1, p in mo.transitions[s][a].items()))
             for a, r in v.items()],
            key=itemgetter(1)
        )[0] for s, v in mo.rewards.items()})
        return pol

    @abstractmethod
    def get_optimal_det_policy(self) -> DetPolicy:
        pass

