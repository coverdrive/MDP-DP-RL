from typing import TypeVar, Callable, Optional, Mapping
from abc import abstractmethod
from algorithms.opt_base import OptBase
from processes.mdp_rep_for_rl_fa import MDPRepForRLFA
from algorithms.func_approx_spec import FuncApproxSpec
from func_approx.func_approx_base import FuncApproxBase
from algorithms.helper_funcs import get_policy_func_for_fa
from algorithms.helper_funcs import get_uniform_policy_func
from operator import itemgetter

S = TypeVar('S')
A = TypeVar('A')
Type1 = Callable[[S], float]
Type2 = Callable[[S], Callable[[A], float]]
PolicyType = Callable[[S], Mapping[A, float]]


class RLFuncApproxBase(OptBase):

    def __init__(
        self,
        mdp_rep_for_rl: MDPRepForRLFA,
        softmax: bool,
        epsilon: float,
        num_episodes: int,
        max_steps: int,
        fa_spec: FuncApproxSpec
    ) -> None:

        self.mdp_rep: MDPRepForRLFA = mdp_rep_for_rl
        self.softmax: bool = softmax
        self.epsilon: float = epsilon
        self.num_episodes: int = num_episodes
        self.max_steps: int = max_steps
        self.vf_fa: FuncApproxBase = fa_spec.get_vf_func_approx_obj()
        self.qvf_fa: FuncApproxBase = fa_spec.get_qvf_func_approx_obj()
        self.state_action_func = self.mdp_rep.state_action_func

    def get_init_policy_func(self) -> PolicyType:
        return get_uniform_policy_func(self.state_action_func)

    def get_value_func_fa(self, polf: PolicyType) -> Type1:
        qv_func = self.get_qv_func_fa(polf)

        # noinspection PyShadowingNames
        def vf(s: S, polf=polf, qv_func=qv_func) -> float:
            return sum(polf(s)[a] * qv_func(s)(a) for a in
                       self.state_action_func(s))

        return vf

    def get_value_func(self, pol_func: Type2) -> Type1:
        return self.get_value_func_fa(
            get_policy_func_for_fa(pol_func, self.state_action_func)
        )

    @abstractmethod
    def get_qv_func_fa(self, polf: Optional[PolicyType]) -> Type2:
        pass

    def get_act_value_func(self, pol_func: Type2) -> Type2:
        return self.get_qv_func_fa(
            get_policy_func_for_fa(pol_func, self.state_action_func)
        )

    def get_optimal_det_policy_func(self) -> Callable[[S], A]:
        qv_func = self.get_qv_func_fa(None)

        # noinspection PyShadowingNames
        def detp_func(s: S, qv_func=qv_func) -> A:
            return max(
                [(a, qv_func(s)(a)) for a in self.state_action_func(s)],
                key=itemgetter(1)
            )[0]

        return detp_func
