from typing import TypeVar, Mapping, Optional, Callable
from algorithms.td_algo_enum import TDAlgorithm
from algorithms.rl_func_approx.rl_func_approx_base import RLFuncApproxBase
from algorithms.func_approx_spec import FuncApproxSpec
from processes.mdp_rep_for_rl_fa import MDPRepForRLFA
from processes.mp_funcs import get_rv_gen_func_single
from algorithms.helper_funcs import get_soft_policy_func_from_qf
import numpy as np

S = TypeVar('S')
A = TypeVar('A')
Type1 = Callable[[S], float]
Type2 = Callable[[S], Callable[[A], float]]
PolicyType = Callable[[S], Mapping[A, float]]


class TDLambda(RLFuncApproxBase):

    def __init__(
        self,
        mdp_rep_for_rl: MDPRepForRLFA,
        algorithm: TDAlgorithm,
        softmax: bool,
        epsilon: float,
        alpha: float,
        lambd: float,
        num_episodes: int,
        max_steps: int,
        fa_spec: FuncApproxSpec,
        offline: bool
    ) -> None:

        super().__init__(
            mdp_rep_for_rl=mdp_rep_for_rl,
            softmax=softmax,
            epsilon=epsilon,
            num_episodes=num_episodes,
            max_steps=max_steps,
            fa_spec=fa_spec
        )
        self.algorithm: TDAlgorithm = algorithm
        self.alpha: float = alpha
        self.lambd: float = lambd
        self.gamma_lambda = self.mdp_rep.gamma * self.lambd
        self.offline = offline

    def get_value_func_fa(self, polf: PolicyType) -> Type1:
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
                          enumerate(self.vf_fa.get_sum_func_gradient([state]))]
                    self.vf_fa.update_params_from_avg_loss_gradient(
                        [e * delta for e in et]
                    )
                steps += 1
                terminate = steps >= self.max_steps or\
                    self.mdp_rep.terminal_state_func(state)
                state = next_state

            if self.offline:
                print(states)
                print(targets)
                avg_grad = [g / len(states) for g in
                            self.vf_fa.get_el_tr_sum_gradient(
                                states,
                                targets,
                                self.gamma_lambda
                            )]
                print(avg_grad[0])
                self.vf_fa.update_params_from_avg_loss_gradient(avg_grad)
                print(self.vf_fa.params[0])
            episodes += 1

        return self.vf_fa.get_func_eval

    # noinspection PyShadowingNames
    def get_qv_func_fa(self, polf: Optional[PolicyType]) -> Type2:
        control = polf is None
        this_polf = polf if polf is not None else self.get_init_policy_func()
        episodes = 0

        while episodes < self.num_episodes:
            et = [np.zeros_like(p) for p in self.qvf_fa.params]
            state, action = self.mdp_rep.init_state_action_gen()
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
                    next_qv = sum(this_polf(next_state).get(a, 0.) *
                                  self.qvf_fa.get_func_eval((next_state, a))
                                  for a in self.state_action_func(next_state))
                else:
                    next_qv = self.qvf_fa.get_func_eval((next_state, next_action))

                target = reward + self.mdp_rep.gamma * next_qv
                delta = target - self.qvf_fa.get_func_eval((state, action))
                if self.offline:
                    states_actions.append((state, action))
                    targets.append(target)
                else:
                    et = [et[i] * self.gamma_lambda + g for i, g in
                          enumerate(self.qvf_fa.get_sum_func_gradient([(state, action)]))]
                    self.qvf_fa.update_params_from_avg_loss_gradient(
                        [e * delta for e in et]
                    )
                if control:
                    this_polf = get_soft_policy_func_from_qf(
                        self.qvf_fa.get_func_eval,
                        self.state_action_func,
                        self.softmax,
                        self.epsilon
                    )
                steps += 1
                terminate = steps >= self.max_steps or \
                    self.mdp_rep.terminal_state_func(state)
                state = next_state
                action = next_action

            if self.offline:
                avg_grad = [g / len(states_actions) for g in
                            self.qvf_fa.get_el_tr_sum_gradient(
                                states_actions,
                                targets,
                                self.gamma_lambda
                            )]
                self.qvf_fa.update_params_from_avg_loss_gradient(avg_grad)
            episodes += 1

        return lambda st: lambda act, st=st: self.qvf_fa.get_func_eval((st, act))


if __name__ == '__main__':
    from processes.mdp_rep_for_rl_tabular import MDPRepForRLTabular
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
    mdp_rep_obj = MDPRepForRLTabular(mdp_ref_obj1)

    algorithm_type = TDAlgorithm.ExpectedSARSA
    softmax_flag = True
    epsilon_val = 0.1
    alpha_val = 0.1
    lambda_val = 0.2
    episodes_limit = 1000
    max_steps_val = 1000
    offline_val = True
    fa_spec_val = FuncApproxSpec(
        state_feature_funcs=[lambda s: float(s)],
        action_feature_funcs=[
            lambda a: 1. if a == 'a' else 0.,
            lambda a: 1. if a == 'b' else 0.,
            lambda a: 1. if a == 'c' else 0.,
        ],
        dnn_spec=None
    )
    esl_obj = TDLambda(
        mdp_rep_obj,
        algorithm_type,
        softmax_flag,
        epsilon_val,
        alpha_val,
        lambda_val,
        episodes_limit,
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
