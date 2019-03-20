from typing import Mapping, Optional
from algorithms.td_algo_enum import TDAlgorithm
from algorithms.rl_func_approx.rl_func_approx_base import RLFuncApproxBase
from algorithms.func_approx_spec import FuncApproxSpec
from processes.mdp_rep_for_rl_fa import MDPRepForRLFA
from processes.mp_funcs import get_rv_gen_func_single
from algorithms.helper_funcs import get_soft_policy_func_from_qf
from processes.mp_funcs import get_expected_action_value
import numpy as np
from utils.generic_typevars import S, A
from utils.standard_typevars import VFType, QFType, PolicyActDictType


class TDLambda(RLFuncApproxBase):

    def __init__(
        self,
        mdp_rep_for_rl: MDPRepForRLFA,
        exploring_start: bool,
        algorithm: TDAlgorithm,
        softmax: bool,
        epsilon: float,
        epsilon_half_life: float,
        lambd: float,
        num_episodes: int,
        batch_size: int,
        max_steps: int,
        fa_spec: FuncApproxSpec,
        offline: bool
    ) -> None:

        super().__init__(
            mdp_rep_for_rl=mdp_rep_for_rl,
            exploring_start=exploring_start,
            softmax=softmax,
            epsilon=epsilon,
            epsilon_half_life=epsilon_half_life,
            num_episodes=num_episodes,
            max_steps=max_steps,
            fa_spec=fa_spec
        )
        self.algorithm: TDAlgorithm = algorithm
        self.gamma_lambda: float = self.mdp_rep.gamma * lambd
        self.batch_size: int = batch_size
        self.offline: bool = offline

    def get_value_func_fa(self, polf: PolicyActDictType) -> VFType:
        episodes = 0

        while episodes < self.num_episodes:
            et = [np.zeros_like(p) for p in self.vf_fa.params]
            state = self.mdp_rep.init_state_gen()
            steps = 0
            terminate = False

            states = []
            targets = []
            while not terminate:
                action = get_rv_gen_func_single(polf(state))()
                next_state, reward =\
                    self.mdp_rep.state_reward_gen_func(state, action)
                target = reward + self.mdp_rep.gamma *\
                    self.vf_fa.get_func_eval(next_state)
                delta = target - self.vf_fa.get_func_eval(state)
                if self.offline:
                    states.append(state)
                    targets.append(target)
                else:
                    et = [et[i] * self.gamma_lambda + g for i, g in
                          enumerate(self.vf_fa.get_sum_objective_gradient(
                              [state],
                              np.ones(1)
                          )
                          )]
                    self.vf_fa.update_params_from_gradient(
                        [-e * delta for e in et]
                    )
                steps += 1
                terminate = steps >= self.max_steps or\
                    self.mdp_rep.terminal_state_func(state)
                state = next_state

            if self.offline:
                avg_grad = [g / len(states) for g in
                            self.vf_fa.get_el_tr_sum_loss_gradient(
                                states,
                                targets,
                                self.gamma_lambda
                            )]
                self.vf_fa.update_params_from_gradient(avg_grad)
            episodes += 1

        return self.vf_fa.get_func_eval

    # noinspection PyShadowingNames
    def get_qv_func_fa(self, polf: Optional[PolicyActDictType]) -> QFType:
        control = polf is None
        this_polf = polf if polf is not None else self.get_init_policy_func()
        episodes = 0

        while episodes < self.num_episodes:
            et = [np.zeros_like(p) for p in self.qvf_fa.params]
            if self.exploring_start:
                state, action = self.mdp_rep.init_state_action_gen()
            else:
                state = self.mdp_rep.init_state_gen()
                action = get_rv_gen_func_single(this_polf(state))()

            # print((episodes, max(self.qvf_fa.get_func_eval((state, a)) for a in
            #        self.mdp_rep.state_action_func(state))))
            # print(self.qvf_fa.params)

            steps = 0
            terminate = False

            states_actions = []
            targets = []
            while not terminate:
                next_state, reward = \
                    self.mdp_rep.state_reward_gen_func(state, action)
                next_action = get_rv_gen_func_single(this_polf(next_state))()
                if self.algorithm == TDAlgorithm.QLearning and control:
                    next_qv = max(self.qvf_fa.get_func_eval((next_state, a)) for a in
                                  self.state_action_func(next_state))
                elif self.algorithm == TDAlgorithm.ExpectedSARSA and control:
                    # next_qv = sum(this_polf(next_state).get(a, 0.) *
                    #               self.qvf_fa.get_func_eval((next_state, a))
                    #               for a in self.state_action_func(next_state))
                    next_qv = get_expected_action_value(
                        {a: self.qvf_fa.get_func_eval((next_state, a)) for a in
                         self.state_action_func(next_state)},
                        self.softmax,
                        self.epsilon_func(episodes)
                    )
                else:
                    next_qv = self.qvf_fa.get_func_eval((next_state, next_action))

                target = reward + self.mdp_rep.gamma * next_qv
                delta = target - self.qvf_fa.get_func_eval((state, action))

                if self.offline:
                    states_actions.append((state, action))
                    targets.append(target)
                else:
                    et = [et[i] * self.gamma_lambda + g for i, g in
                          enumerate(self.qvf_fa.get_sum_objective_gradient(
                              [(state, action)],
                              np.ones(1)
                          )
                          )]
                    self.qvf_fa.update_params_from_gradient(
                        [-e * delta for e in et]
                    )
                if control and self.batch_size == 0:
                    this_polf = get_soft_policy_func_from_qf(
                        self.qvf_fa.get_func_eval,
                        self.state_action_func,
                        self.softmax,
                        self.epsilon_func(episodes)
                    )
                steps += 1
                terminate = steps >= self.max_steps or \
                    self.mdp_rep.terminal_state_func(state)

                state = next_state
                action = next_action

            if self.offline:
                avg_grad = [g / len(states_actions) for g in
                            self.qvf_fa.get_el_tr_sum_loss_gradient(
                                states_actions,
                                targets,
                                self.gamma_lambda
                            )]
                self.qvf_fa.update_params_from_gradient(avg_grad)

            episodes += 1

            if control and self.batch_size != 0 and\
                    episodes % self.batch_size == 0:
                this_polf = get_soft_policy_func_from_qf(
                    self.qvf_fa.get_func_eval,
                    self.state_action_func,
                    self.softmax,
                    self.epsilon_func(episodes - 1)
                )

        return lambda st: lambda act, st=st: self.qvf_fa.get_func_eval((st, act))


if __name__ == '__main__':
    from processes.mdp_refined import MDPRefined
    mdp_refined_data = {
        1: {
            'a': {1: (0.3, 9.2), 2: (0.6, 4.5), 3: (0.1, 5.0)},
            'b': {2: (0.3, -0.5), 3: (0.7, 2.6)},
            'c': {1: (0.2, 4.8), 2: (0.4, -4.9), 3: (0.4, 0.0)}
        },
        2: {
            'a': {1: (0.3, 9.8), 2: (0.6, 6.7), 3: (0.1, 1.8)},
            'c': {1: (0.2, 4.8), 2: (0.4, 9.2), 3: (0.4, -8.2)}
        },
        3: {
            'a': {3: (1.0, 0.0)},
            'b': {3: (1.0, 0.0)}
        }
    }
    gamma_val = 0.9
    mdp_ref_obj1 = MDPRefined(mdp_refined_data, gamma_val)
    mdp_rep_obj = mdp_ref_obj1.get_mdp_rep_for_rl_tabular()

    exploring_start_val = False
    algorithm_type = TDAlgorithm.ExpectedSARSA
    softmax_flag = False
    epsilon_val = 0.1
    epsilon_half_life_val = 1000
    learning_rate_val = 0.1
    lambda_val = 0.7
    episodes_limit = 10000
    batch_size_val = 20
    max_steps_val = 1000
    offline_val = True
    state_ff = [lambda s: float(s)]
    sa_ff = [
        lambda x: float(x[0]),
        lambda x: 1. if x[1] == 'a' else 0.,
        lambda x: 1. if x[1] == 'b' else 0.,
        lambda x: 1. if x[1] == 'c' else 0.,
    ]
    fa_spec_val = FuncApproxSpec(
        state_feature_funcs=state_ff,
        sa_feature_funcs=sa_ff,
        dnn_spec=None,
        learning_rate=learning_rate_val
    )
    esl_obj = TDLambda(
        mdp_rep_obj,
        exploring_start_val,
        algorithm_type,
        softmax_flag,
        epsilon_val,
        epsilon_half_life_val,
        lambda_val,
        episodes_limit,
        batch_size_val,
        max_steps_val,
        fa_spec_val,
        offline_val
    )

    def policy_func(i: int) -> Mapping[str, float]:
        if i == 1:
            ret = {'a': 0.4, 'b': 0.6}
        elif i == 2:
            ret = {'a': 0.7, 'c': 0.3}
        elif i == 3:
            ret = {'b': 1.0}
        else:
            raise ValueError
        return ret

    this_qf = esl_obj.get_qv_func_fa(policy_func)
    this_vf = esl_obj.get_value_func_fa(policy_func)
    print(this_vf(1))
    print(this_vf(2))
    print(this_vf(3))

    opt_det_polf = esl_obj.get_optimal_det_policy_func()

    # noinspection PyShadowingNames
    def opt_polf(s: S, opt_det_polf=opt_det_polf) -> Mapping[A, float]:
        return {opt_det_polf(s): 1.0}

    opt_vf = esl_obj.get_value_func_fa(opt_polf)
    print(opt_polf(1))
    print(opt_polf(2))
    print(opt_polf(3))
    print(opt_vf(1))
    print(opt_vf(2))
    print(opt_vf(3))
