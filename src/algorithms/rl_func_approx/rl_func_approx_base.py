from typing import Callable, Optional
from abc import abstractmethod
from algorithms.opt_base import OptBase
from processes.mdp_rep_for_rl_fa import MDPRepForRLFA
from algorithms.func_approx_spec import FuncApproxSpec
from func_approx.func_approx_base import FuncApproxBase
from algorithms.helper_funcs import get_uniform_policy_func
from algorithms.helper_funcs import get_epsilon_decay_func
from algorithms.helper_funcs import get_pdf_from_samples
from operator import itemgetter
from utils.generic_typevars import S, A
from utils.standard_typevars import VFType, QFType
from utils.standard_typevars import PolicyType, PolicyActDictType


class RLFuncApproxBase(OptBase):

    NUM_SAMPLES_PER_ACTION = 10

    def __init__(
        self,
        mdp_rep_for_rl: MDPRepForRLFA,
        exploring_start: bool,
        softmax: bool,
        epsilon: float,
        epsilon_half_life: float,
        num_episodes: int,
        max_steps: int,
        fa_spec: FuncApproxSpec
    ) -> None:

        self.mdp_rep: MDPRepForRLFA = mdp_rep_for_rl
        self.exploring_start: bool = exploring_start
        self.softmax: bool = softmax
        self.epsilon_func: Callable[[int], float] = get_epsilon_decay_func(
            epsilon,
            epsilon_half_life
        )
        self.num_episodes: int = num_episodes
        self.max_steps: int = max_steps
        self.vf_fa: FuncApproxBase = fa_spec.get_vf_func_approx_obj()
        self.qvf_fa: FuncApproxBase = fa_spec.get_qvf_func_approx_obj()
        self.state_action_func = self.mdp_rep.state_action_func

    def get_init_policy_func(self) -> PolicyActDictType:
        return get_uniform_policy_func(self.state_action_func)

    def get_value_func_fa(self, polf: PolicyActDictType) -> VFType:
        qv_func = self.get_qv_func_fa(polf)

        # noinspection PyShadowingNames
        def vf(s: S, polf=polf, qv_func=qv_func) -> float:
            return sum(polf(s)[a] * qv_func(s)(a) for a in
                       self.state_action_func(s))

        return vf

    # noinspection PyShadowingNames
    def get_value_func(self, pol_func: PolicyType) -> VFType:
        return self.get_value_func_fa(
            lambda s, pol_func=pol_func: get_pdf_from_samples(
                pol_func(s)(len(self.state_action_func(s)) *
                            RLFuncApproxBase.NUM_SAMPLES_PER_ACTION)
            )
        )

    @abstractmethod
    def get_qv_func_fa(self, polf: Optional[PolicyActDictType]) -> QFType:
        pass

    # noinspection PyShadowingNames
    def get_act_value_func(self, pol_func: PolicyType) -> QFType:
        return self.get_qv_func_fa(
            lambda s, pol_func=pol_func: get_pdf_from_samples(
                pol_func(s)(len(self.state_action_func(s)) *
                            RLFuncApproxBase.NUM_SAMPLES_PER_ACTION)
            )
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
