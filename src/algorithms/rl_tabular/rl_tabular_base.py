from typing import TypeVar, Mapping, Callable, Tuple, Optional
from abc import abstractmethod
from algorithms.opt_base import OptBase
from processes.mdp_rep_for_rl_finite_sa import MDPRepForRLFiniteSA
from processes.policy import Policy
from processes.det_policy import DetPolicy
from algorithms.helper_funcs import get_vf_from_qf_and_policy
from algorithms.helper_funcs import get_uniform_policy
from algorithms.helper_funcs import get_det_policy_from_qf

S = TypeVar('S')
A = TypeVar('A')
Type1 = Mapping[S, Mapping[A, Callable[[], Tuple[S, float]]]]
VFType = Mapping[S, float]
QVFType = Mapping[S, Mapping[A, float]]


class TabularBase(OptBase):

    def __init__(
        self,
        mdp_rep_for_rl: MDPRepForRLFiniteSA,
        softmax: bool,
        epsilon: float,
        num_episodes: int,
        max_steps: int
    ) -> None:

        self.mdp_rep: MDPRepForRLFiniteSA = mdp_rep_for_rl
        self.softmax: bool = softmax
        self.epsilon: float = epsilon
        self.num_episodes: int = num_episodes
        self.max_steps: int = max_steps

    def get_init_policy(self) -> Policy:
        return get_uniform_policy(self.mdp_rep.state_action_dict)

    def get_value_func_dict(self, pol: Policy) -> VFType:
        return get_vf_from_qf_and_policy(
            self.get_act_value_func_dict(pol),
            pol
        )

    @abstractmethod
    def get_qv_func_dict(self, pol: Optional[Policy]) -> QVFType:
        pass

    def get_act_value_func_dict(self, pol: Policy) -> QVFType:
        return self.get_qv_func_dict(pol)

    def get_optimal_det_policy(self) -> DetPolicy:
        qf_dict = self.get_qv_func_dict(None)
        return get_det_policy_from_qf(qf_dict)
