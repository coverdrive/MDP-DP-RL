from typing import Mapping, Callable, Sequence, Tuple
from algorithms.opt_base import OptBase
from processes.mdp_rep_for_adp_pg import MDPRepForADPPG
from algorithms.func_approx_spec import FuncApproxSpec
from func_approx.func_approx_base import FuncApproxBase
from processes.mp_funcs import mdp_func_to_mrp_func1, mdp_func_to_mrp_func2
from algorithms.helper_funcs import get_policy_as_action_dict
from operator import itemgetter
from random import choices
from copy import deepcopy
import numpy as np
from utils.generic_typevars import S, A
from utils.standard_typevars import VFType, QFType
from utils.standard_typevars import PolicyType, PolicyActDictType


class ADPPolicyGradient(OptBase):

    NUM_ACTION_SAMPLES = 100

    def __init__(
        self,
        mdp_rep_for_adp_pg: MDPRepForADPPG,
        num_samples: int,
        max_steps: int,
        tol: float,
        actor_lambda: float,
        critic_lambda: float,
        score_func: Callable[[A, Sequence[float]], Sequence[float]],
        sample_actions_gen_func: Callable[[Sequence[float], int], Sequence[A]],
        vf_fa_spec: FuncApproxSpec,
        pol_fa_spec: Sequence[FuncApproxSpec]

    ) -> None:
        self.mdp_rep: MDPRepForADPPG = mdp_rep_for_adp_pg
        self.num_samples: int = num_samples
        self.max_steps: int = max_steps
        self.tol: float = tol
        self.actor_lambda: float = actor_lambda
        self.critic_lambda: float = critic_lambda
        self.score_func: Callable[[A, Sequence[float]], Sequence[float]] =\
            score_func
        self.sample_actions_gen_func: Callable[[Sequence[float], int], Sequence[A]] =\
            sample_actions_gen_func
        self.vf_fa: FuncApproxBase = vf_fa_spec.get_vf_func_approx_obj()
        self.pol_fa: Sequence[FuncApproxBase] =\
            [s.get_vf_func_approx_obj() for s in pol_fa_spec]

    @staticmethod
    def get_gradient_max(gradient: Sequence[np.ndarray]) -> float:
        return max(np.max(np.abs(g)) for g in gradient)

    def get_value_func_fa(self, polf: PolicyActDictType) -> VFType:
        epsilon = self.tol * 1e4
        mo = self.mdp_rep
        rew_func = mdp_func_to_mrp_func2(self.mdp_rep.reward_func, polf)
        prob_func = mdp_func_to_mrp_func1(self.mdp_rep.transitions_func, polf)
        init_samples_func = self.mdp_rep.init_states_gen_func
        while epsilon >= self.tol:
            samples = init_samples_func(self.num_samples)
            values = [rew_func(s) + mo.gamma *
                      sum(p * self.vf_fa.get_func_eval(s1) for s1, p in
                          prob_func(s).items())
                      for s in samples]
            avg_grad = [g / len(samples) for g in
                        self.vf_fa.get_sum_loss_gradient(samples, values)]
            self.vf_fa.update_params_from_gradient(avg_grad)
            epsilon = ADPPolicyGradient.get_gradient_max(avg_grad)
            # print(self.vf_fa.get_func_eval(1))
            # print(self.vf_fa.get_func_eval(2))
            # print(self.vf_fa.get_func_eval(3))
            # print("-----")

        return self.vf_fa.get_func_eval

    def get_act_value_func_fa(self, polf: PolicyActDictType) -> QFType:
        v_func = self.get_value_func_fa(polf)

        # noinspection PyShadowingNames
        def state_func(s: S, v_func=v_func) -> Callable[[A], float]:

            # noinspection PyShadowingNames
            def act_func(a: A, v_func=v_func) -> float:
                return self.mdp_rep.reward_func(s, a) + self.mdp_rep.gamma *\
                       sum(p * v_func(s1) for s1, p in
                           self.mdp_rep.transitions_func(s, a).items())

            return act_func

        return state_func

    def get_value_func(self, pol_func: PolicyType) -> VFType:
        return self.get_value_func_fa(
            get_policy_as_action_dict(
                pol_func,
                ADPPolicyGradient.NUM_ACTION_SAMPLES
            )
        )

    def get_act_value_func(self, pol_func: PolicyType) -> QFType:
        return self.get_act_value_func_fa(
            get_policy_as_action_dict(
                pol_func,
                ADPPolicyGradient.NUM_ACTION_SAMPLES
            )
        )

    def get_policy_as_policy_type(self) -> PolicyType:

        def pol(s: S) -> Callable[[int], Sequence[A]]:

            # noinspection PyShadowingNames
            def gen_func(samples: int, s=s) -> Sequence[A]:
                return self.sample_actions_gen_func(
                    [f.get_func_eval(s) for f in self.pol_fa],
                    samples
                )

            return gen_func

        return pol

    def get_path(
        self,
        start_state: S
    ) -> Sequence[Tuple[S, Sequence[float], A, float]]:
        res = []
        state = start_state
        steps = 0
        terminate = False

        while not terminate:
            pdf_params = [f.get_func_eval(state) for f in self.pol_fa]
            action = self.sample_actions_gen_func(pdf_params, 1)[0]
            reward = self.mdp_rep.reward_func(state, action)
            res.append((
                state,
                pdf_params,
                action,
                reward
            ))
            steps += 1
            terminate = steps >= self.max_steps or\
                self.mdp_rep.terminal_state_func(state)
            next_states, probs = zip(*self.mdp_rep.transitions_func(
                state,
                action
            ).items())
            state = choices(next_states, probs, k=1)[0]
        return res

    def get_optimal_stoch_policy_func(self) -> PolicyType:
        mo = self.mdp_rep
        init_samples_func = mo.init_states_gen_func
        eps = self.tol * 1e4
        params = deepcopy(self.vf_fa.params)
        tr_func = mo.transitions_func
        sc_func = self.score_func
        while eps >= self.tol:
            init_states = init_samples_func(self.num_samples)
            gamma_pow = 1.
            pol_grads = [
                [np.zeros_like(layer) for layer in this_pol_fa.params]
                for this_pol_fa in self.pol_fa
            ]
            for init_state in init_states:
                states = []
                deltas = []
                disc_scores = []
                this_path = self.get_path(init_state)

                for s, pp, a, r in this_path:
                    delta = r + mo.gamma * sum(
                        p * self.vf_fa.get_func_eval(s1) for s1, p in
                        tr_func(s, a).items()
                    ) - self.vf_fa.get_func_eval(s)
                    states.append(s)
                    deltas.append(delta)
                    disc_scores.append(
                        [gamma_pow * x for x in sc_func(a, pp)]
                    )
                    gamma_pow *= mo.gamma

                self.vf_fa.update_params_from_gradient(
                    self.vf_fa.get_el_tr_sum_objective_gradient(
                        states,
                        np.power(mo.gamma, np.arange(len(this_path))),
                        - np.array(deltas),
                        mo.gamma * self.critic_lambda
                    )
                )

                pg_arr = np.vstack(disc_scores)
                for i, pp_fa in enumerate(self.pol_fa):
                    this_pol_grad = pp_fa.get_el_tr_sum_objective_gradient(
                        states,
                        pg_arr[:, i],
                        - np.array(deltas),
                        mo.gamma * self.actor_lambda
                    )
                    for j in range(len(pol_grads[i])):
                        pol_grads[i][j] += this_pol_grad[j]

            for i, pp_fa in enumerate(self.pol_fa):
                pp_fa.update_params_from_gradient(
                    [pg / self.num_samples for pg in pol_grads[i]]
                )

            new_params = deepcopy(self.vf_fa.params)
            eps = ADPPolicyGradient.get_gradient_max(
                [new_params[i] - p for i, p in enumerate(params)]
            )
            params = new_params

            print(self.vf_fa.get_func_eval(1))
            print(self.vf_fa.get_func_eval(2))
            print(self.vf_fa.get_func_eval(3))
            print("----")

        return self.get_policy_as_policy_type()

    def get_optimal_det_policy_func(self) -> Callable[[S], A]:
        pol_func = get_policy_as_action_dict(
            self.get_optimal_stoch_policy_func(),
            ADPPolicyGradient.NUM_ACTION_SAMPLES
        )
        return lambda s: max(pol_func(s).items(), key=itemgetter(1))[0]


if __name__ == '__main__':
    from processes.mdp_refined import MDPRefined
    from func_approx.dnn_spec import DNNSpec
    from numpy.random import binomial

    mdp_refined_data = {
        1: {
            'a': {1: (0.3, 9.2), 2: (0.6, 4.5), 3: (0.1, 5.0)},
            'b': {2: (0.3, -0.5), 3: (0.7, 2.6)}
        },
        2: {
            'a': {1: (0.3, 9.8), 2: (0.6, 6.7), 3: (0.1, 1.8)},
            'b': {1: (0.3, 19.8), 2: (0.6, 16.7), 3: (0.1, 1.8)},
        },
        3: {
            'a': {3: (1.0, 0.0)},
            'b': {3: (1.0, 0.0)}
        }
    }
    gamma_val = 0.9
    mdp_ref_obj1 = MDPRefined(mdp_refined_data, gamma_val)
    mdp_rep_obj = mdp_ref_obj1.get_mdp_rep_for_adp_pg()

    num_samples_val = 10
    max_steps_val = 100
    tol_val = 1e-4
    actor_lambda_val = 0.95
    critic_lambda_val = 0.95
    vf_fa_spec_val = FuncApproxSpec(
        state_feature_funcs=[
            lambda s: 1. if s == 1 else 0.,
            lambda s: 1. if s == 2 else 0.,
            lambda s: 1. if s == 3 else 0.
        ],
        action_feature_funcs=[],
        dnn_spec=DNNSpec(
            neurons=[2],
            hidden_activation=DNNSpec.relu,
            hidden_activation_deriv=DNNSpec.relu_deriv,
            output_activation=DNNSpec.identity,
            output_activation_deriv=DNNSpec.identity_deriv
        )
    )
    pol_fa_spec_val = [FuncApproxSpec(
        state_feature_funcs=[
            lambda s: 1. if s == 1 else 0.,
            lambda s: 1. if s == 2 else 0.,
            lambda s: 1. if s == 3 else 0.
        ],
        action_feature_funcs=[],
        dnn_spec=DNNSpec(
            neurons=[2],
            hidden_activation=DNNSpec.relu,
            hidden_activation_deriv=DNNSpec.relu_deriv,
            output_activation=DNNSpec.sigmoid,
            output_activation_deriv=DNNSpec.sigmoid_deriv
        )
    )]
    # noinspection PyPep8
    this_score_func = lambda a, p: [1. / p[0] if a == 'a' else 1. / (p[0] - 1.)]
    # noinspection PyPep8
    sa_gen_func = lambda p, n: [('a' if x == 1 else 'b') for x in binomial(1, p[0], n)]
    adp_pg_obj = ADPPolicyGradient(
        mdp_rep_for_adp_pg=mdp_rep_obj,
        num_samples=num_samples_val,
        max_steps=max_steps_val,
        tol=tol_val,
        actor_lambda=actor_lambda_val,
        critic_lambda=critic_lambda_val,
        score_func=this_score_func,
        sample_actions_gen_func=sa_gen_func,
        vf_fa_spec=vf_fa_spec_val,
        pol_fa_spec=pol_fa_spec_val
    )

    def policy_func(i: int) -> Mapping[str, float]:
        if i == 1:
            ret = {'a': 0.4, 'b': 0.6}
        elif i == 2:
            ret = {'a': 0.7, 'b': 0.3}
        elif i == 3:
            ret = {'b': 1.0}
        else:
            raise ValueError
        return ret

    # print("Printing DP vf for a policy")
    # from processes.policy import Policy
    # true_vf_for_pol = mdp_ref_obj1.get_value_func_dict(Policy(
    #     {s: policy_func(s) for s in {1, 2, 3}}
    # ))
    # print(true_vf_for_pol)
    #
    # # this_qf = adp_pg_obj.get_act_value_func_fa(policy_func)
    # this_vf = adp_pg_obj.get_value_func_fa(policy_func)
    # print("Printing vf for a policy")
    # print(this_vf(1))
    # print(this_vf(2))
    # print(this_vf(3))

    true_opt = mdp_ref_obj1.get_optimal_policy(tol=tol_val)
    print("Printing DP Opt Policy")
    print(true_opt)
    true_vf = mdp_ref_obj1.get_value_func_dict(true_opt)
    print("Printing DP Opt VF")
    print(true_vf)

    opt_det_polf = adp_pg_obj.get_optimal_det_policy_func()

    # noinspection PyShadowingNames
    def opt_polf(s: S, opt_det_polf=opt_det_polf) -> Mapping[A, float]:
        return {opt_det_polf(s): 1.0}

    print("Printing Opt Policy")
    print(opt_polf(1))
    print(opt_polf(2))
    print(opt_polf(3))

    opt_vf = adp_pg_obj.get_value_func(adp_pg_obj.get_policy_as_policy_type())
    print("Printing Opt VF")
    print(opt_vf(1))
    print(opt_vf(2))
    print(opt_vf(3))
