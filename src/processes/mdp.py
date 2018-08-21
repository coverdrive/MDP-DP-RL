from typing import Mapping, Set, Tuple, Generic
from utils.gen_utils import zip_dict_of_tuple, is_approx_eq
from processes.mp_funcs import get_all_states, get_actions_for_states
from processes.mp_funcs import verify_mdp, get_lean_transitions
from processes.policy import Policy
from processes.det_policy import DetPolicy
from processes.mp_funcs import mdp_rep_to_mrp_rep1, mdp_rep_to_mrp_rep2
from operator import itemgetter
from processes.mrp import MRP
from processes.mp_funcs import get_rv_gen_func
from processes.mdp_rep_for_adp import MDPRepForADP
from utils.generic_typevars import S, A


class MDP(Generic[S, A]):

    def __init__(
        self,
        info: Mapping[S, Mapping[A, Tuple[Mapping[S, float], float]]],
        gamma: float
    ) -> None:
        if verify_mdp(info):
            d = {k: zip_dict_of_tuple(v) for k, v in info.items()}
            d1, d2 = zip_dict_of_tuple(d)
            self.all_states: Set[S] = get_all_states(info)
            self.state_action_dict: Mapping[S, Set[A]] = \
                get_actions_for_states(info)
            self.transitions: Mapping[S, Mapping[A, Mapping[S, float]]] = \
                {s: {a: get_lean_transitions(v1) for a, v1 in v.items()}
                 for s, v in d1.items()}
            self.rewards: Mapping[S, Mapping[A, float]] = d2
            self.gamma: float = gamma
            self.terminal_states: Set[S] = self.get_terminal_states()
        else:
            raise ValueError

    def get_sink_states(self) -> Set[S]:
        return {k for k, v in self.transitions.items() if
                all(len(v1) == 1 and k in v1.keys() for _, v1 in v.items())
                }

    def get_terminal_states(self) -> Set[S]:
        """
        A terminal state is a sink state (100% probability to going back
        to itself, FOR EACH ACTION) and the rewards on those transitions back
        to itself are zero.
        """
        sink = self.get_sink_states()
        return {s for s in sink if
                all(is_approx_eq(r, 0.0) for _, r in self.rewards[s].items())}

    def get_mrp(self, pol: Policy) -> MRP:
        tr = mdp_rep_to_mrp_rep1(self.transitions, pol.policy_data)
        rew = mdp_rep_to_mrp_rep2(self.rewards, pol.policy_data)
        return MRP({s: (v, rew[s]) for s, v in tr.items()}, self.gamma)

    def get_value_func_dict(self, pol: Policy)\
            -> Mapping[S, float]:
        mrp_obj = self.get_mrp(pol)
        value_func_vec = mrp_obj.get_value_func_vec()
        nt_vf = {mrp_obj.nt_states_list[i]: value_func_vec[i]
                 for i in range(len(mrp_obj.nt_states_list))}
        t_vf = {s: 0. for s in self.terminal_states}
        return {**nt_vf, **t_vf}

    def get_act_value_func_dict(self, pol: Policy)\
            -> Mapping[S, Mapping[A, float]]:
        v_dict = self.get_value_func_dict(pol)
        return {s: {a: r + self.gamma * sum(p * v_dict[s1] for s1, p in
                                            self.transitions[s][a].items())
                    for a, r in v.items()}
                for s, v in self.rewards.items()}

    def get_improved_policy(self, pol: Policy) -> DetPolicy:
        q_dict = self.get_act_value_func_dict(pol)
        return DetPolicy({s: max(v.items(), key=itemgetter(1))[0]
                          for s, v in q_dict.items()})

    def get_optimal_policy(self, tol=1e-4) -> DetPolicy:
        pol = Policy({s: {a: 1. / len(v) for a in v} for s, v in
                      self.state_action_dict.items()})
        vf = self.get_value_func_dict(pol)
        epsilon = tol * 1e4
        while epsilon >= tol:
            pol = self.get_improved_policy(pol)
            new_vf = self.get_value_func_dict(pol)
            epsilon = max(abs(new_vf[s] - v) for s, v in vf.items())
            vf = new_vf
        return pol

    def get_mdp_rep_for_adp(self) -> MDPRepForADP:
        return MDPRepForADP(
            state_action_func=lambda s: self.state_action_dict[s],
            gamma=self.gamma,
            sample_states_gen_func=get_rv_gen_func(
                {s: 1. / len(self.state_action_dict) for s in
                 self.state_action_dict.keys()}
            ),
            reward_func=lambda s, a: self.rewards[s][a],
            transitions_func=lambda s, a: self.transitions[s][a]
        )


if __name__ == '__main__':
    data = {
        1: {
            'a': ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
            'b': ({2: 0.3, 3: 0.7}, 2.8),
            'c': ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2)
        },
        2: {
            'a': ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
            'c': ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2)
        },
        3: {
            'a': ({3: 1.0}, 0.0),
            'b': ({3: 1.0}, 0.0)
        }
    }
    mdp_obj = MDP(data, 0.95)
    print(mdp_obj.all_states)
    print(mdp_obj.transitions)
    print(mdp_obj.rewards)
    terminal = mdp_obj.get_terminal_states()
    print(terminal)
    policy_data = {
        1: {'a': 0.4, 'b': 0.6},
        2: {'a': 0.7, 'c': 0.3},
        3: {'b': 1.0}
    }
    pol_obj = Policy(policy_data)
    mdp_data = {
        1: {
            'a': ({1: 0.2, 2: 0.6, 3: 0.2}, 7.0),
            'b': ({1: 0.6, 2: 0.3, 3: 0.1}, -2.0),
            'c': ({1: 0.1, 2: 0.2, 3: 0.7}, 10.0)
        },
        2: {
            'a': ({1: 0.1, 2: 0.6, 3: 0.3}, 1.0),
            'c': ({1: 0.6, 2: 0.2, 3: 0.2}, -1.2)
        },
        3: {
            'b': ({3: 1.0}, 0.0)
        }
    }
    mdp1_obj = MDP(mdp_data, gamma=0.9)
    mrp1_obj = mdp1_obj.get_mrp(pol_obj)
    print(mrp1_obj.transitions)
    print(mrp1_obj.rewards)
    print(mrp1_obj.trans_matrix)
    print(mrp1_obj.rewards_vec)
    print(mrp1_obj.get_value_func_vec())
    opt_policy = mdp1_obj.get_optimal_policy()
    print(opt_policy.policy_data)
    opt_vf_dict = mdp1_obj.get_value_func_dict(opt_policy)
    print(opt_vf_dict)
