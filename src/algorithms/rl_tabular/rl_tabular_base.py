from typing import TypeVar, Mapping, Optional, Set
from abc import abstractmethod
from algorithms.tabular_base import TabularBase
from processes.mdp_rep_for_rl_tabular import MDPRepForRLTabular
from processes.policy import Policy
from processes.det_policy import DetPolicy
from algorithms.helper_funcs import get_vf_dict_from_qf_dict_and_policy
from algorithms.helper_funcs import get_uniform_policy
from algorithms.helper_funcs import get_det_policy_from_qf_dict

S = TypeVar('S')
A = TypeVar('A')
VFType = Mapping[S, float]
QVFType = Mapping[S, Mapping[A, float]]


class RLTabularBase(TabularBase):

    def __init__(
        self,
        mdp_rep_for_rl: MDPRepForRLTabular,
        softmax: bool,
        epsilon: float,
        num_episodes: int,
        max_steps: int
    ) -> None:

        self.mdp_rep: MDPRepForRLTabular = mdp_rep_for_rl
        self.softmax: bool = softmax
        self.epsilon: float = epsilon
        self.num_episodes: int = num_episodes
        self.max_steps: int = max_steps

    def get_state_action_dict(self) -> Mapping[S, Set[A]]:
        return self.mdp_rep.state_action_dict

    def get_init_policy(self) -> Policy:
        return get_uniform_policy(self.mdp_rep.state_action_dict)

    def get_value_func_dict(self, pol: Policy) -> VFType:
        return get_vf_dict_from_qf_dict_and_policy(
            self.get_qv_func_dict(pol),
            pol
        )

    @abstractmethod
    def get_qv_func_dict(self, pol: Optional[Policy]) -> QVFType:
        pass

    def get_act_value_func_dict(self, pol: Policy) -> QVFType:
        return self.get_qv_func_dict(pol)

    def get_optimal_det_policy(self) -> DetPolicy:
        return get_det_policy_from_qf_dict(self.get_qv_func_dict(None))
