from typing import Optional, Tuple, Sequence
from algorithms.rl_tabular.rl_tabular_base import RLTabularBase
from processes.policy import Policy
from processes.mp_funcs import get_rv_gen_func_single
from processes.mdp_rep_for_rl_tabular import MDPRepForRLTabular
from algorithms.helper_funcs import get_returns_from_rewards_terminating
from algorithms.helper_funcs import get_returns_from_rewards_non_terminating
from algorithms.helper_funcs import get_soft_policy_from_qf_dict
from algorithms.helper_funcs import get_nt_return_eval_steps
import numpy as np
from utils.generic_typevars import S, A
from utils.standard_typevars import VFDictType, QFDictType


class MonteCarlo(RLTabularBase):

    def __init__(
        self,
        mdp_rep_for_rl: MDPRepForRLTabular,
        exploring_start: bool,
        first_visit: bool,
        softmax: bool,
        epsilon: float,
        epsilon_half_life: float,
        num_episodes: int,
        max_steps: int
    ) -> None:

        super().__init__(
            mdp_rep_for_rl=mdp_rep_for_rl,
            exploring_start=exploring_start,
            softmax=softmax,
            epsilon=epsilon,
            epsilon_half_life=epsilon_half_life,
            num_episodes=num_episodes,
            max_steps=max_steps
        )
        self.first_visit: bool = first_visit
        self.nt_return_eval_steps = get_nt_return_eval_steps(
            max_steps,
            mdp_rep_for_rl.gamma,
            1e-4
        )

    def get_mc_path(
        self,
        pol: Policy,
        start_state: S,
        start_action: Optional[A] = None,
    ) -> Sequence[Tuple[S, A, float, bool]]:

        res = []
        state = start_state
        steps = 0
        terminate = False
        occ_states = set()
        act_gen_dict = {s: get_rv_gen_func_single(pol.get_state_probabilities(s))
                        for s in self.mdp_rep.state_action_dict.keys()}

        while not terminate:
            first = state not in occ_states
            occ_states.add(state)
            action = act_gen_dict[state]()\
                if (steps > 0 or start_action is None) else start_action
            next_state, reward =\
                self.mdp_rep.state_reward_gen_dict[state][action]()
            res.append((state, action, reward, first))
            steps += 1
            terminate = steps >= self.max_steps or\
                state in self.mdp_rep.terminal_states
            state = next_state
        return res

    def get_value_func_dict(self, pol: Policy) -> VFDictType:
        sa_dict = self.mdp_rep.state_action_dict
        counts_dict = {s: 0 for s in sa_dict.keys()}
        vf_dict = {s: 0.0 for s in sa_dict.keys()}
        episodes = 0

        while episodes < self.num_episodes:
            start_state = self.mdp_rep.init_state_gen()
            mc_path = self.get_mc_path(
                pol,
                start_state,
                start_action=None
            )

            rew_arr = np.array([x for _, _, x, _ in mc_path])
            if mc_path[-1][0] in self.mdp_rep.terminal_states:
                returns = get_returns_from_rewards_terminating(
                    rew_arr,
                    self.mdp_rep.gamma
                )
            else:
                returns = get_returns_from_rewards_non_terminating(
                    rew_arr,
                    self.mdp_rep.gamma,
                    self.nt_return_eval_steps
                )
            for i, r in enumerate(returns):
                s, _, _, f = mc_path[i]
                if not self.first_visit or f:
                    counts_dict[s] += 1
                    c = counts_dict[s]
                    vf_dict[s] = (vf_dict[s] * (c - 1) + r) / c
            episodes += 1

        return vf_dict

    def get_qv_func_dict(self, pol: Optional[Policy]) -> QFDictType:
        control = pol is None
        this_pol = pol if pol is not None else self.get_init_policy()
        sa_dict = self.mdp_rep.state_action_dict
        counts_dict = {s: {a: 0 for a in v} for s, v in sa_dict.items()}
        qf_dict = {s: {a: 0.0 for a in v} for s, v in sa_dict.items()}
        episodes = 0

        while episodes < self.num_episodes:
            if self.exploring_start:
                start_state, start_action = self.mdp_rep.init_state_action_gen()
            else:
                start_state = self.mdp_rep.init_state_gen()
                start_action = None
            mc_path = self.get_mc_path(
                this_pol,
                start_state,
                start_action
            )
            rew_arr = np.array([x for _, _, x, _ in mc_path])
            if mc_path[-1][0] in self.mdp_rep.terminal_states:
                returns = get_returns_from_rewards_terminating(
                    rew_arr,
                    self.mdp_rep.gamma
                )
            else:
                returns = get_returns_from_rewards_non_terminating(
                    rew_arr,
                    self.mdp_rep.gamma,
                    self.nt_return_eval_steps
                )
            for i, r in enumerate(returns):
                s, a, _, f = mc_path[i]
                if not self.first_visit or f:
                    counts_dict[s][a] += 1
                    c = counts_dict[s][a]
                    qf_dict[s][a] = (qf_dict[s][a] * (c - 1) + r) / c
            if control:
                this_pol = get_soft_policy_from_qf_dict(
                    qf_dict,
                    self.softmax,
                    self.epsilon_func(episodes)
                )
            episodes += 1

        return qf_dict


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
    gamma_val = 1.0
    mdp_ref_obj1 = MDPRefined(mdp_refined_data, gamma_val)
    mdp_rep_obj = mdp_ref_obj1.get_mdp_rep_for_rl_tabular()

    exploring_start_val = False
    first_visit_flag = True
    softmax_flag = False
    episodes_limit = 1000
    epsilon_val = 0.1
    epsilon_half_life_val = 100
    max_steps_val = 1000
    mc_obj = MonteCarlo(
        mdp_rep_obj,
        exploring_start_val,
        first_visit_flag,
        softmax_flag,
        epsilon_val,
        epsilon_half_life_val,
        episodes_limit,
        max_steps_val
    )

    policy_data = {
        1: {'a': 0.4, 'b': 0.6},
        2: {'a': 0.7, 'c': 0.3},
        3: {'b': 1.0}
    }
    pol_obj = Policy(policy_data)

    this_mc_path = mc_obj.get_mc_path(pol_obj, 1)
    print(this_mc_path)

    this_qf_dict = mc_obj.get_act_value_func_dict(pol_obj)
    print(this_qf_dict)
    this_vf_dict = mc_obj.get_value_func_dict(pol_obj)
    print(this_vf_dict)

    opt_pol = mc_obj.get_optimal_det_policy()
    print(opt_pol)
    opt_vf_dict = mc_obj.get_value_func_dict(opt_pol)
    print(opt_vf_dict)
