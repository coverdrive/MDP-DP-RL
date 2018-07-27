from typing import TypeVar, Mapping
from algorithms.opt_learning_tdl_base import OptLearningTDLBase
from processes.mdp_refined import MDPRefined
from processes.policy import Policy
from processes.det_policy import DetPolicy
from algorithms.helper_funcs import get_rv_gen_func
from algorithms.helper_funcs import get_soft_policy_from_qf
from algorithms.helper_funcs import get_det_policy_from_qf

S = TypeVar('S')
A = TypeVar('A')
VFType = Mapping[S, float]
QVFType = Mapping[S, Mapping[A, float]]


class OptLearningExpSARSALambda(OptLearningTDLBase):

    def __init__(
        self,
        mdp_ref_obj: MDPRefined,
        softmax: bool,
        epsilon: float,
        alpha: float,
        lambd: float,
        num_episodes: int,
        max_steps: int
    ) -> None:

        super().__init__(
            mdp_ref_obj,
            softmax,
            epsilon,
            alpha,
            lambd,
            num_episodes,
            max_steps
        )

    def get_optimal_det_policy(self) -> DetPolicy:
        pol = self.get_init_policy()
        sa_dict = self.state_action_dict
        s_uniform_dict = {s: 1. / len(sa_dict) for s in sa_dict.keys()}
        start_gen_f = get_rv_gen_func(s_uniform_dict)
        qf_dict = {s: {a: 0.0 for a in v} for s, v in sa_dict.items()}
        episodes = 0

        while episodes < self.num_episodes:
            state = start_gen_f(1)[0]
            steps = 0
            terminate = False

            while not terminate:
                et_dict = {s: {a: 0.0 for a in v} for s, v in sa_dict.items()}
                action = get_rv_gen_func(pol.get_state_probabilities(state))(1)[0]
                next_state, reward = self.state_reward_gen_dict[state][action]()
                delta = reward + self.gamma * sum(
                    pol.get_state_action_probability(next_state, a) *
                    qf_dict[next_state][a] for a in sa_dict[next_state])\
                    - qf_dict[state][action]
                et_dict[state][action] += 1
                for s, a_set in self.state_action_dict.items():
                    for a in a_set:
                        qf_dict[s][a] += self.alpha * delta * et_dict[s][a]
                        et_dict[s][a] *= self.gamma * self.lambd
                pol = get_soft_policy_from_qf(qf_dict, self.softmax, self.epsilon)
                state = next_state
                steps += 1
                terminate = steps >= self.max_steps or\
                    state in self.terminal_states

            episodes += 1

        return get_det_policy_from_qf(qf_dict)


if __name__ == '__main__':
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

    softmax_flag = True
    epsilon_val = 0.1
    alpha_val = 0.1
    lambda_val = 0.2
    episodes_limit = 1000
    max_steps_val = 1000
    esl_obj = OptLearningExpSARSALambda(
        mdp_ref_obj1,
        softmax_flag,
        epsilon_val,
        alpha_val,
        lambda_val,
        episodes_limit,
        max_steps_val
    )

    policy_data = {
        1: {'a': 0.4, 'b': 0.6},
        2: {'a': 0.7, 'c': 0.3},
        3: {'b': 1.0}
    }
    pol_obj = Policy(policy_data)

    this_qf_dict = esl_obj.get_act_value_func_dict(pol_obj)
    print(this_qf_dict)
    this_vf_dict = esl_obj.get_value_func_dict(pol_obj)
    print(this_vf_dict)

    opt_pol = esl_obj.get_optimal_det_policy()
    print(opt_pol)
    opt_vf_dict = esl_obj.get_value_func_dict(opt_pol)
    print(opt_vf_dict)
