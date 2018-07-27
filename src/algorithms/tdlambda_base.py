from typing import TypeVar, Mapping
from algorithms.learning_base import LearningBase
from abc import abstractmethod
from processes.mdp_refined import MDPRefined
from processes.policy import Policy
from processes.det_policy import DetPolicy
from algorithms.helper_funcs import get_rv_gen_func

S = TypeVar('S')
A = TypeVar('A')
VFType = Mapping[S, float]
QVFType = Mapping[S, Mapping[A, float]]


class TDLambdaBase(LearningBase):

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
            num_episodes,
            max_steps
        )
        self.alpha: float = alpha
        self.lambd: float = lambd

    def get_value_func_dict(self, pol: Policy) -> VFType:
        sa_dict = self.state_action_dict
        s_uniform_dict = {s: 1. / len(sa_dict) for s in sa_dict.keys()}
        start_gen_f = get_rv_gen_func(s_uniform_dict)
        vf_dict = {s: 0. for s in sa_dict.keys()}
        act_gen_dict = {s: get_rv_gen_func(pol.get_state_probabilities(s))
                        for s in self.state_action_dict.keys()}
        episodes = 0

        while episodes < self.num_episodes:
            et_dict = {s: 0. for s in sa_dict.keys()}
            state = start_gen_f(1)[0]
            steps = 0
            terminate = False

            while not terminate:
                action = act_gen_dict[state](1)[0]
                next_state, reward = self.state_reward_gen_dict[state][action]()
                delta = reward + self.gamma * vf_dict[next_state] - vf_dict[state]
                et_dict[state] += 1
                for s in self.state_action_dict.keys():
                    vf_dict[s] += self.alpha * delta * et_dict[s]
                    et_dict[s] *= self.gamma * self.lambd
                state = next_state
                steps += 1
                terminate = steps >= self.max_steps or\
                    state in self.terminal_states

            episodes += 1

        return vf_dict

    def get_act_value_func_dict(self, pol: Policy) -> QVFType:
        sa_dict = self.state_action_dict
        sa_uniform_dict = {(s, a): 1. / sum(len(v) for v in sa_dict.values())
                           for s, v1 in sa_dict.items() for a in v1}
        start_gen_f = get_rv_gen_func(sa_uniform_dict)
        qf_dict = {s: {a: 0.0 for a in v} for s, v in sa_dict.items()}
        act_gen_dict = {s: get_rv_gen_func(pol.get_state_probabilities(s))
                        for s in self.state_action_dict.keys()}
        episodes = 0

        while episodes < self.num_episodes:
            et_dict = {s: {a: 0.0 for a in v} for s, v in sa_dict.items()}
            state, action = start_gen_f(1)[0]
            steps = 0
            terminate = False

            while not terminate:
                next_state, reward = self.state_reward_gen_dict[state][action]()
                next_action = act_gen_dict[next_state](1)[0]
                delta = reward + self.gamma * qf_dict[next_state][next_action] -\
                    qf_dict[state][action]
                et_dict[state][action] += 1
                for s, a_set in self.state_action_dict.items():
                    for a in a_set:
                        qf_dict[s][a] += self.alpha * delta * et_dict[s][a]
                        et_dict[s][a] *= self.gamma * self.lambd
                state = next_state
                action = next_action
                steps += 1
                terminate = steps >= self.max_steps or\
                    state in self.terminal_states

            episodes += 1

        return qf_dict

    @abstractmethod
    def get_optimal_det_policy(self) -> DetPolicy:
        pass
