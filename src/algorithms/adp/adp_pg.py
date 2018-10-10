from typing import Mapping, Callable, Sequence, Tuple
from algorithms.opt_base import OptBase
from processes.mdp_rep_for_adp_pg import MDPRepForADPPG
from algorithms.func_approx_spec import FuncApproxSpec
from func_approx.func_approx_base import FuncApproxBase
from algorithms.helper_funcs import get_policy_as_action_dict
import numpy as np
from utils.generic_typevars import S, A
from utils.standard_typevars import VFType, QFType
from utils.standard_typevars import PolicyType, PolicyActDictType


class ADPPolicyGradient(OptBase):
    """
    This class is meant for continuous state spaces and continuous action
    spaces. Here we blend ADP (i.e., DP with function approximation for
    value function) with PG (i.e., gradient of policy drives policy
    improvement). See the class ADP to understand how plain ADP is done
    (i.e., without PG) - this class is an extension/variant of that
    technique where the policy improvement is policy gradient instead
    of argmax. Note that both ADP and this class are model-based
    methods, meaning we have access to state transition probabilities
    and expected rewards, and we take advantage of that information
    in the algorithm. On the other hand, the class PolicyGradient is
    a model-free algorithm based on PolicyGradient (also meant for
    continuous states/actions).
    """

    def __init__(
        self,
        mdp_rep_for_adp_pg: MDPRepForADPPG,
        reinforce: bool,
        num_state_samples: int,
        num_next_state_samples: int,
        num_action_samples: int,
        num_batches: int,
        max_steps: int,
        actor_lambda: float,
        critic_lambda: float,
        score_func: Callable[[A, Sequence[float]], Sequence[float]],
        sample_actions_gen_func: Callable[[Sequence[float], int], Sequence[A]],
        vf_fa_spec: FuncApproxSpec,
        pol_fa_spec: Sequence[FuncApproxSpec]

    ) -> None:
        self.mdp_rep: MDPRepForADPPG = mdp_rep_for_adp_pg
        self.reinforce: bool = reinforce
        self.num_state_samples: int = num_state_samples
        self.num_next_state_samples: int = num_next_state_samples
        self.num_action_samples: int = num_action_samples
        self.num_batches: int = num_batches
        self.max_steps: int = max_steps
        self.actor_lambda: float = actor_lambda
        self.critic_lambda: float = critic_lambda
        self.score_func: Callable[[A, Sequence[float]], Sequence[float]] =\
            score_func
        self.sample_actions_gen_func: Callable[[Sequence[float], int], Sequence[A]] =\
            sample_actions_gen_func
        self.vf_fa: FuncApproxBase = vf_fa_spec.get_vf_func_approx_obj()
        self.pol_fa: Sequence[FuncApproxBase] =\
            [s.get_vf_func_approx_obj() for s in pol_fa_spec]

    def get_value_func_fa(self, polf: PolicyActDictType) -> VFType:
        mo = self.mdp_rep
        sr_func = self.mdp_rep.state_reward_gen_func
        # rew_func = mdp_func_to_mrp_func2(self.mdp_rep.reward_func, polf)
        # prob_func = mdp_func_to_mrp_func1(self.mdp_rep.transitions_func, polf)
        init_samples_func = self.mdp_rep.init_states_gen_func
        for _ in range(self.num_batches):
            samples = init_samples_func(self.num_state_samples)
            values = [sum(ap * (r + mo.gamma * self.vf_fa.get_func_eval(s1))
                          for a, ap in polf(s).items() for s1, r in
                          sr_func(s, a, self.num_next_state_samples)) /
                      self.num_next_state_samples for s in samples]
            avg_grad = [g / len(samples) for g in
                        self.vf_fa.get_sum_loss_gradient(samples, values)]
            self.vf_fa.update_params_from_gradient(avg_grad)
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
                return sum(r + self.mdp_rep.gamma * v_func(s1) for
                           s1, r in self.mdp_rep.state_reward_gen_func(
                    s,
                    a,
                    self.num_next_state_samples
                )
                           ) / self.num_next_state_samples

            return act_func

        return state_func

    def get_value_func(self, pol_func: PolicyType) -> VFType:
        return self.get_value_func_fa(
            get_policy_as_action_dict(
                pol_func,
                self.num_action_samples
            )
        )

    def get_act_value_func(self, pol_func: PolicyType) -> QFType:
        return self.get_act_value_func_fa(
            get_policy_as_action_dict(
                pol_func,
                self.num_action_samples
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
    ) -> Sequence[Tuple[S, Sequence[float], A]]:
        res = []
        state = start_state
        steps = 0
        terminate = False

        while not terminate:
            pdf_params = [f.get_func_eval(state) for f in self.pol_fa]
            action = self.sample_actions_gen_func(pdf_params, 1)[0]
            res.append((
                state,
                pdf_params,
                action
            ))
            steps += 1
            terminate = steps >= self.max_steps or\
                self.mdp_rep.terminal_state_func(state)
            state = self.mdp_rep.state_reward_gen_func(state, action, 1)[0][0]
        return res

    def get_optimal_reinforce_func(self) -> PolicyType:
        mo = self.mdp_rep
        init_samples_func = mo.init_states_gen_func
        sc_func = self.score_func

        for _ in range(self.num_batches):
            init_states = init_samples_func(self.num_state_samples)
            pol_grads = [
                [np.zeros_like(layer) for layer in this_pol_fa.params]
                for this_pol_fa in self.pol_fa
            ]
            for init_state in init_states:
                states = []
                disc_return_scores = []
                return_val = 0.
                this_path = self.get_path(init_state)

                for i, (s, pp, a) in enumerate(this_path[::-1]):
                    i1 = len(this_path) - 1 - i
                    states.append(s)
                    reward = sum(r for _, r in mo.state_reward_gen_func(
                        s,
                        a,
                        self.num_next_state_samples
                    )) / self.num_next_state_samples
                    return_val = return_val * mo.gamma + reward
                    disc_return_scores.append(
                        [return_val * mo.gamma ** i1 * x for x in sc_func(a, pp)]
                    )
                    # print(s)
                    # print(pp)

                pg_arr = np.vstack(disc_return_scores)
                for i, pp_fa in enumerate(self.pol_fa):
                    this_pol_grad = pp_fa.get_sum_objective_gradient(
                        states,
                        - pg_arr[:, i]
                    )
                    for j in range(len(pol_grads[i])):
                        pol_grads[i][j] += this_pol_grad[j]

            # print("--------")
            for i, pp_fa in enumerate(self.pol_fa):
                gradient = [pg / self.num_state_samples for pg in pol_grads[i]]
                # print(gradient)
                pp_fa.update_params_from_gradient(gradient)

        return self.get_policy_as_policy_type()

    def get_optimal_tdl_func(self) -> PolicyType:
        mo = self.mdp_rep
        init_samples_func = mo.init_states_gen_func
        sc_func = self.score_func
        for _ in range(self.num_batches):
            init_states = init_samples_func(self.num_state_samples)
            pol_grads = [
                [np.zeros_like(layer) for layer in this_pol_fa.params]
                for this_pol_fa in self.pol_fa
            ]
            for init_state in init_states:
                gamma_pow = 1.
                states = []
                deltas = []
                disc_scores = []
                this_path = self.get_path(init_state)

                for s, pp, a in this_path:
                    target = sum(r + mo.gamma * self.vf_fa.get_func_eval(s1)
                                 for s1, r in mo.state_reward_gen_func(
                        s,
                        a,
                        self.num_next_state_samples
                    )) / self.num_next_state_samples
                    delta = target - self.vf_fa.get_func_eval(s)
                    states.append(s)
                    deltas.append(delta)
                    disc_scores.append(
                        [gamma_pow * x for x in sc_func(a, pp)]
                    )
                    gamma_pow *= mo.gamma
                    # print(s)
                    # print(pp)
                    # print(a)
                    # print(target)

                # print(list(zip(states, deltas)))

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
                gradient = [pg / self.num_state_samples for pg in pol_grads[i]]
                # print(gradient)
                pp_fa.update_params_from_gradient(gradient)

            # print(self.vf_fa.get_func_eval(1))
            # print(self.vf_fa.get_func_eval(2))
            # print(self.vf_fa.get_func_eval(3))
            # print("----")

        return self.get_policy_as_policy_type()

    def get_optimal_stoch_policy_func(self) -> PolicyType:
        return self.get_optimal_reinforce_func() if self.reinforce\
            else self.get_optimal_tdl_func()

    def get_optimal_det_policy_func(self) -> Callable[[S], A]:
        papt = self.get_optimal_stoch_policy_func()

        def opt_det_pol_func(s: S) -> A:
            return tuple(np.mean(
                papt(s)(self.num_action_samples),
                axis=0
            ))

        return opt_det_pol_func


if __name__ == '__main__':
    from processes.mdp_refined import MDPRefined
    from func_approx.dnn_spec import DNNSpec
    from numpy.random import binomial

    mdp_refined_data = {
        1: {
            (10,): {1: (0.3, 9.2), 2: (0.6, 4.5), 3: (0.1, 5.0)},
            (-10,): {2: (0.3, -0.5), 3: (0.7, 2.6)}
        },
        2: {
            (10,): {1: (0.3, 9.8), 2: (0.6, 6.7), 3: (0.1, 1.8)},
            (-10,): {1: (0.3, 19.8), 2: (0.6, 16.7), 3: (0.1, 1.8)},
        },
        3: {
            (10,): {3: (1.0, 0.0)},
            (-10,): {3: (1.0, 0.0)}
        }
    }
    gamma_val = 0.9
    mdp_ref_obj1 = MDPRefined(mdp_refined_data, gamma_val)
    mdp_rep_obj = mdp_ref_obj1.get_mdp_rep_for_adp_pg()

    reinforce_val = False

    num_state_samples_val = 100
    num_next_state_samples_val = 25
    num_action_samples_val = 20
    num_batches_val = 100
    max_steps_val = 100
    actor_lambda_val = 0.95
    critic_lambda_val = 0.95
    state_ff = [
        lambda s: 1. if s == 1 else 0.,
        lambda s: 1. if s == 2 else 0.,
        lambda s: 1. if s == 3 else 0.
    ]
    vf_fa_spec_val = FuncApproxSpec(
        state_feature_funcs=state_ff,
        sa_feature_funcs=[(lambda x, f=f: f(x[0])) for f in state_ff],
        dnn_spec=DNNSpec(
            neurons=[2],
            hidden_activation=DNNSpec.relu,
            hidden_activation_deriv=DNNSpec.relu_deriv,
            output_activation=DNNSpec.identity,
            output_activation_deriv=DNNSpec.identity_deriv
        )
    )
    pol_fa_spec_val = [FuncApproxSpec(
        state_feature_funcs=state_ff,
        sa_feature_funcs=[(lambda x, f=f: f(x[0])) for f in state_ff],
        dnn_spec=DNNSpec(
            neurons=[3],
            hidden_activation=DNNSpec.relu,
            hidden_activation_deriv=DNNSpec.relu_deriv,
            output_activation=DNNSpec.sigmoid,
            output_activation_deriv=DNNSpec.sigmoid_deriv
        )
    )]
    # noinspection PyPep8
    this_score_func = lambda a, p: [1. / p[0] if a == (10,) else 1. / (p[0] - 1.)]
    # noinspection PyPep8
    sa_gen_func = lambda p, n: [((10,) if x == 1 else (-10,)) for x in binomial(1, p[0], n)]
    adp_pg_obj = ADPPolicyGradient(
        mdp_rep_for_adp_pg=mdp_rep_obj,
        reinforce=reinforce_val,
        num_state_samples=num_state_samples_val,
        num_next_state_samples=num_next_state_samples_val,
        num_action_samples=num_action_samples_val,
        num_batches=num_batches_val,
        max_steps=max_steps_val,
        actor_lambda=actor_lambda_val,
        critic_lambda=critic_lambda_val,
        score_func=this_score_func,
        sample_actions_gen_func=sa_gen_func,
        vf_fa_spec=vf_fa_spec_val,
        pol_fa_spec=pol_fa_spec_val
    )

    def policy_func(i: int) -> Mapping[Tuple[int], float]:
        if i == 1:
            ret = {(10,): 0.4, (-10,): 0.6}
        elif i == 2:
            ret = {(10,): 0.7, (-10,): 0.3}
        elif i == 3:
            ret = {(-10,): 1.0}
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

    tol_val = 1e-6
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
