from typing import TypeVar, Mapping, Callable, Sequence, Set
from algorithms.helper_funcs import get_uniform_policy_func
from algorithms.opt_base import OptBase
from algorithms.helper_funcs import get_policy_func_for_fa
from algorithms.helper_funcs import get_epsilon_decay_func
from processes.mdp_rep_for_adp import MDPRepForADP
from algorithms.helper_funcs import get_soft_policy_func_from_qf
from algorithms.func_approx_spec import FuncApproxSpec
from func_approx.func_approx_base import FuncApproxBase
from processes.mp_funcs import mdp_func_to_mrp_func1, mdp_func_to_mrp_func2
from processes.mp_funcs import get_expected_action_value
from operator import itemgetter
import numpy as np


S = TypeVar('S')
A = TypeVar('A')
Type1 = Callable[[S], float]
Type2 = Callable[[S], Callable[[A], float]]
PolicyType = Callable[[S], Mapping[A, float]]


class ADP(OptBase):

    def __init__(
        self,
        mdp_rep_for_adp: MDPRepForADP,
        num_samples: int,
        softmax: bool,
        epsilon: float,
        epsilon_half_life: float,
        tol: float,
        learning_rate: float,
        fa_spec: FuncApproxSpec
    ) -> None:
        self.mdp_rep: MDPRepForADP = mdp_rep_for_adp
        self.num_samples: int = num_samples
        self.softmax: bool = softmax
        self.epsilon_func: Callable[[int], float] = get_epsilon_decay_func(
            epsilon,
            epsilon_half_life
        )
        self.tol: float = tol
        self.learning_rate: float = learning_rate
        self.fa: FuncApproxBase = fa_spec.get_vf_func_approx_obj()
        self.state_action_func: Callable[[S], Set[A]] =\
            self.mdp_rep.state_action_func

    @staticmethod
    def get_gradient_max(gradient: Sequence[np.ndarray]) -> float:
        return max(np.max(g) for g in gradient)

    def get_init_policy_func(self) -> PolicyType:
        return get_uniform_policy_func(self.state_action_func)

    # noinspection PyShadowingNames
    def get_improved_policy_func(
        self,
        polf: PolicyType,
        epsilon: float
    ) -> PolicyType:
        return get_soft_policy_func_from_qf(
            lambda sa, polf=polf: self.get_act_value_func_fa(polf)(sa[0])(sa[1]),
            self.state_action_func,
            self.softmax,
            epsilon
        )

    def get_value_func_fa(self, polf: PolicyType) -> Type1:
        epsilon = self.tol * 1e4
        mo = self.mdp_rep
        rew_func = mdp_func_to_mrp_func2(self.mdp_rep.reward_func, polf)
        prob_func = mdp_func_to_mrp_func1(self.mdp_rep.transitions_func, polf)
        samples_func = self.mdp_rep.sample_states_gen_func
        while epsilon >= self.tol:
            samples = samples_func(self.num_samples)
            values = [rew_func(s) + mo.gamma *
                      sum(p * self.fa.get_func_eval(s1) for s1, p in
                          prob_func(s).items())
                      for s in samples]
            avg_grad = [g / len(samples) for g in self.fa.get_sum_loss_gradient(
                samples,
                values
            )]
            self.fa.update_params_from_avg_loss_gradient(avg_grad)
            epsilon = ADP.get_gradient_max(avg_grad)
        return lambda s: self.fa.get_func_eval(s)

    def get_act_value_func_fa(self, polf: PolicyType) -> Type2:
        mo = self.mdp_rep
        v_func = self.get_value_func_fa(polf)

        # noinspection PyShadowingNames
        def state_func(s: S, mo=mo, v_func=v_func) -> Callable[[A], float]:

            # noinspection PyShadowingNames
            def act_func(a: A, mo=mo, v_func=v_func) -> float:
                return self.mdp_rep.reward_func(s, a) + mo.gamma *\
                       sum(p * v_func(s1) for s1, p in
                           self.mdp_rep.transitions_func(s, a))

            return act_func

        return state_func

    def get_value_func(self, pol_func: Type2) -> Type1:
        return self.get_value_func_fa(
            get_policy_func_for_fa(pol_func, self.state_action_func)
        )

    def get_act_value_func(self, pol_func: Type2) -> Type2:
        return self.get_act_value_func_fa(
            get_policy_func_for_fa(pol_func, self.state_action_func)
        )

    def get_optimal_policy_func_pi(self) -> Callable[[S], A]:
        this_polf = self.get_init_policy_func()
        eps = self.tol * 1e4
        iters = 0
        params = self.fa.params
        while eps >= self.tol:
            g_epsilon = self.epsilon_func(iters)
            this_polf = self.get_improved_policy_func(this_polf, g_epsilon)
            new_params = self.fa.params
            eps = ADP.get_gradient_max(
                [new_params[i] - p for i, p in enumerate(params)]
            )
            iters += 1

        # noinspection PyShadowingNames
        def det_pol(s: S, this_polf=this_polf) -> A:
            return max(this_polf(s).items(), key=itemgetter(1))[0]

        return det_pol

    def get_optimal_policy_func_vi(self) -> Callable[[S], A]:
        mo = self.mdp_rep
        samples_func = self.mdp_rep.sample_states_gen_func
        rew_func = self.mdp_rep.reward_func
        tr_func = self.mdp_rep.transitions_func
        eps = self.tol * 1e4
        iters = 0
        while eps >= self.tol:
            samples = samples_func(self.num_samples)
            values = [get_expected_action_value(
                {a: rew_func(s, a) + mo.gamma *
                    sum(p * self.fa.get_func_eval(s1)
                        for s1, p in tr_func(s, a).items())
                 for a in self.state_action_func(s)},
                self.softmax,
                self.epsilon_func(iters)
            ) for s in samples]
            avg_grad = [g / len(samples) for g in self.fa.get_sum_loss_gradient(
                samples,
                values
            )]
            self.fa.update_params_from_avg_loss_gradient(avg_grad)
            eps = ADP.get_gradient_max(avg_grad)
            iters += 1

        # noinspection PyShadowingNames
        def deter_func(s: S, rew_func=rew_func, tr_func=tr_func) -> A:
            return max(
                [(a, rew_func(s, a) +
                  sum(p * self.fa.get_func_eval(s1) for s1, p in
                      tr_func(s, a).items()))
                 for a in self.state_action_func(s)],
                key=itemgetter(1)
            )[0]

        return deter_func

    def get_optimal_det_policy_func(self) -> Callable[[S], A]:
        return self.get_optimal_policy_func_pi()


if __name__ == '__main__':
    print(0)
