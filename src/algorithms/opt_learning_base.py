from typing import TypeVar, Mapping, Callable, Tuple, Set
from abc import abstractmethod
from algorithms.opt_base import OptBase
from processes.mdp_refined import MDPRefined
from algorithms.helper_funcs import get_state_reward_gen_dict
from processes.policy import Policy
from processes.det_policy import DetPolicy
from algorithms.helper_funcs import get_vf_from_qf_and_policy
from algorithms.helper_funcs import get_uniform_policy

S = TypeVar('S')
A = TypeVar('A')
Type1 = Mapping[S, Mapping[A, Callable[[], Tuple[S, float]]]]
VFType = Mapping[S, float]
QVFType = Mapping[S, Mapping[A, float]]


class OptLearningBase(OptBase):

    def __init__(
        self,
        mdp_ref_obj: MDPRefined,
        softmax: bool,
        epsilon: float,
        num_episodes: int
    ) -> None:

        self.state_action_dict: Mapping[S, Set[A]] = mdp_ref_obj.state_action_dict
        self.gamma: float = mdp_ref_obj.gamma
        self.state_reward_gen_dict: Type1 = get_state_reward_gen_dict(
            mdp_ref_obj.rewards_refined,
            mdp_ref_obj.transitions
        )
        self.terminal_states: Set[S] = mdp_ref_obj.terminal_states
        self.softmax: bool = softmax
        self.epsilon: float = epsilon
        self.num_episodes: int = num_episodes

    def get_init_policy(self) -> Policy:
        return get_uniform_policy(self.state_action_dict)

    def get_value_func_dict(self, pol: Policy) -> VFType:
        return get_vf_from_qf_and_policy(
            self.get_act_value_func_dict(pol),
            pol
        )

    @abstractmethod
    def get_act_value_func_dict(self, pol: Policy) -> QVFType:
        pass

    @abstractmethod
    def get_optimal(self) -> Tuple[DetPolicy, VFType]:
        pass

