from typing import Mapping, Callable, Sequence
from algorithms.opt_base import OptBase
from processes.mdp_rep_for_adp_pg import MDPRepForADPPG
from algorithms.func_approx_spec import FuncApproxSpec
from func_approx.func_approx_base import FuncApproxBase
from processes.mp_funcs import mdp_func_to_mrp_func1, mdp_func_to_mrp_func2
from algorithms.helper_funcs import get_policy_as_action_dict
from operator import itemgetter
from copy import deepcopy
import numpy as np
from utils.generic_typevars import S, A
from utils.standard_typevars import VFType, QFType
from utils.standard_typevars import PolicyType


class ADPPolicyGradient(OptBase):

    NUM_SAMPLES_PER_ACTION = 10

    def __init__(
        self,
        mdp_rep_for_adp_pg: MDPRepForADPPG,
        num_state_samples: int,
        num_action_samples: int,
        tol: float,
        score_func: Callable[[A, Sequence[float]], Sequence[float]],
        sample_actions_gen_func: Callable[[Sequence[float], int], Sequence[A]],
        vf_fa_spec: FuncApproxSpec,
        pol_fa_spec: Sequence[FuncApproxSpec]

    ) -> None:
        self.mdp_rep: MDPRepForADPPG = mdp_rep_for_adp_pg
        self.num_state_samples: int = num_state_samples
        self.num_action_samples: int = num_action_samples
        self.tol: float = tol
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

    def get_value_func(self, polf: PolicyType) -> VFType:
        epsilon = self.tol * 1e4
        mo = self.mdp_rep
        pol_func = get_policy_as_action_dict(
            polf,
            self.num_action_samples
        )
        rew_func = mdp_func_to_mrp_func2(self.mdp_rep.reward_func, pol_func)
        prob_func = mdp_func_to_mrp_func1(self.mdp_rep.transitions_func, pol_func)
        samples_func = self.mdp_rep.sample_states_gen_func
        while epsilon >= self.tol:
            samples = samples_func(self.num_state_samples)
            values = [rew_func(s) + mo.gamma *
                      sum(p * self.vf_fa.get_func_eval(s1) for s1, p in
                          prob_func(s).items())
                      for s in samples]
            avg_grad = [g / len(samples) for g in self.vf_fa.get_sum_loss_gradient(
                samples,
                values
            )]
            self.vf_fa.update_params_from_avg_loss_gradient(avg_grad)
            epsilon = ADPPolicyGradient.get_gradient_max(avg_grad)

        return self.vf_fa.get_func_eval

    def get_act_value_func(self, polf: PolicyType) -> QFType:
        v_func = self.get_value_func(polf)
        mo = self.mdp_rep

        # noinspection PyShadowingNames
        def state_func(s: S, mo=mo, v_func=v_func) -> Callable[[A], float]:

            # noinspection PyShadowingNames
            def act_func(a: A, mo=mo, v_func=v_func) -> float:
                return self.mdp_rep.reward_func(s, a) + mo.gamma *\
                       sum(p * v_func(s1) for s1, p in
                           self.mdp_rep.transitions_func(s, a).items())

            return act_func

        return state_func

    def get_policy_pdf_params_func(self) -> Callable[[S], Sequence[float]]:
        return lambda s: [f.get_func_eval(s) for f in self.pol_fa]

    def get_policy_as_policy_type(self) -> PolicyType:
        agf = self.sample_actions_gen_func
        pf = self.get_policy_pdf_params_func()

        # noinspection PyShadowingNames
        def pol(s: S) -> Callable[[int], Sequence[A]]:
            return lambda sps, s=s, pf=pf, agf=agf: agf(pf(s), sps)

        return pol

    def get_optimal_stoch_policy_func(self) -> PolicyType:
        mo = self.mdp_rep
        samples_func = mo.sample_states_gen_func
        eps = self.tol * 1e4
        params = [deepcopy(fap) for fap in self.vf_fa.params]
        papt = self.get_policy_as_policy_type()
        pol_func = get_policy_as_action_dict(
            papt,
            self.num_action_samples
        )
        rew_func = mo.reward_func
        tr_func = mo.transitions_func
        sc_func = self.score_func
        ppp_func = self.get_policy_pdf_params_func()
        while eps >= self.tol:
            samples = samples_func(self.num_state_samples)
            values = []
            pol_grads = []
            for s in samples:
                ppp = ppp_func(s)
                prob_score_ret = [(
                    ap,
                    np.array(sc_func(av, ppp)),
                    rew_func(s, av) + mo.gamma * sum(
                        p * self.vf_fa.get_func_eval(s1) for s1, p in
                        tr_func(s, av).items()
                    )
                ) for av, ap in pol_func(s).items()]
                values.append(sum(p * r for p, _, r in prob_score_ret))
                pol_grads.append(sum(p * r * sc for p, sc, r in prob_score_ret))

            avg_value_grad = [g / len(samples) for g in
                              self.vf_fa.get_sum_loss_gradient(samples, values)]
            self.vf_fa.update_params_from_avg_loss_gradient(avg_value_grad)

            pol_grads_arr = np.vstack(pol_grads)
            for i, pp_fa in enumerate(self.pol_fa):
                avg_pol_grad = [h / len(samples) for h in
                                pp_fa.get_sum_objective_gradient(
                                    samples,
                                    pol_grads_arr[:, i]
                                )]
                pp_fa.update_params_from_avg_loss_gradient(avg_pol_grad)

            new_params = [deepcopy(fap) for fap in self.vf_fa.params]
            eps = max(ADPPolicyGradient.get_gradient_max(
                [new_params[i][j] - p for j, p in enumerate(this_params)]
            ) for i, this_params in enumerate(params))
            params = new_params

        return papt

    def get_optimal_det_policy_func(self) -> Callable[[S], A]:
        pol_func = get_policy_as_action_dict(
            self.get_optimal_stoch_policy_func(),
            self.num_action_samples
        )
        return lambda s: max(pol_func(s).items(), key=itemgetter(1))[0]


if __name__ == '__main__':
    from processes.mdp_refined import MDPRefined
    from func_approx.dnn_spec import DNNSpec
    from numpy.random import binomial
    from processes.mp_funcs import get_sampling_func_from_prob_dict

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
    mdp_rep_obj = mdp_ref_obj1.get_mdp_rep_for_adp()

    num_state_samples_val = 100
    num_action_samples_val = 100
    tol_val = 1e-4
    vf_fa_spec_val = FuncApproxSpec(
        state_feature_funcs=[
            lambda s: 1. if s == 1 else 0.,
            lambda s: 1. if s == 2 else 0.,
            lambda s: 1. if s == 3 else 0.
        ],
        action_feature_funcs=[],
        dnn_spec=DNNSpec(
            neurons=[2, 4],
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
            neurons=[2, 4],
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
        num_state_samples=num_state_samples_val,
        num_action_samples=num_action_samples_val,
        tol=tol_val,
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

    def pf_as_policy_type(i: int) -> Callable[[int], Sequence[str]]:
        return get_sampling_func_from_prob_dict(policy_func(i))

    this_qf = adp_pg_obj.get_act_value_func(pf_as_policy_type)
    this_vf = adp_pg_obj.get_value_func(pf_as_policy_type)
    print("Printing vf for a policy")
    print(this_vf(1))
    print(this_vf(2))
    print(this_vf(3))
    print("Printing DP vf for a policy")
    from processes.policy import Policy
    true_vf_for_pol = mdp_ref_obj1.get_value_func_dict(Policy(
        {s: policy_func(s) for s in {1, 2, 3}}
    ))
    print(true_vf_for_pol)

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
    true_opt = mdp_ref_obj1.get_optimal_policy(tol=tol_val)
    print("Printing DP Opt Policy")
    print(true_opt)
    true_vf = mdp_ref_obj1.get_value_func_dict(true_opt)
    print("Printing DP Opt VF")
    print(true_vf)
