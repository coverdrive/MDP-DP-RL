from typing import Mapping, Optional, Set, Callable
from abc import abstractmethod
from algorithms.tabular_base import TabularBase
from processes.mdp_rep_for_rl_tabular import MDPRepForRLTabular
from processes.policy import Policy
from processes.det_policy import DetPolicy
from algorithms.helper_funcs import get_vf_dict_from_qf_dict_and_policy
from algorithms.helper_funcs import get_uniform_policy
from algorithms.helper_funcs import get_det_policy_from_qf_dict
from algorithms.helper_funcs import get_epsilon_decay_func
from utils.generic_typevars import S, A
from utils.standard_typevars import VFDictType, QFDictType


class RLTabularBase(TabularBase):

    def __init__(
        self,
        mdp_rep_for_rl: MDPRepForRLTabular,
        exploring_start: bool,
        softmax: bool,
        epsilon: float,
        epsilon_half_life: float,
        num_episodes: int,
        max_steps: int
    ) -> None:

        self.mdp_rep: MDPRepForRLTabular = mdp_rep_for_rl
        self.exploring_start: bool = exploring_start
        self.softmax: bool = softmax
        self.epsilon_func: Callable[[int], float] = get_epsilon_decay_func(
            epsilon,
            epsilon_half_life
        )
        self.num_episodes: int = num_episodes
        self.max_steps: int = max_steps

    def get_state_action_dict(self) -> Mapping[S, Set[A]]:
        return self.mdp_rep.state_action_dict

    def get_init_policy(self) -> Policy:
        return get_uniform_policy(self.mdp_rep.state_action_dict)

    def get_value_func_dict(self, pol: Policy) -> VFDictType:
        return get_vf_dict_from_qf_dict_and_policy(
            self.get_qv_func_dict(pol),
            pol
        )

    @abstractmethod
    def get_qv_func_dict(self, pol: Optional[Policy]) -> QFDictType:
        pass

    def get_act_value_func_dict(self, pol: Policy) -> QFDictType:
        return self.get_qv_func_dict(pol)

    def get_optimal_det_policy(self) -> DetPolicy:
        return get_det_policy_from_qf_dict(self.get_qv_func_dict(None))
