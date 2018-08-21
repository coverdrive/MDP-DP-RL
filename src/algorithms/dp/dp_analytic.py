from algorithms.dp.dp_base import DPBase
from processes.policy import Policy
from processes.mdp import MDP
from processes.det_policy import DetPolicy
from utils.standard_typevars import VFDictType


class DPAnalytic(DPBase):

    def __init__(self, mdp_obj: MDP, tol: float) -> None:
        super().__init__(mdp_obj, tol)

    def get_value_func_dict(self, pol: Policy) -> VFDictType:
        mrp_obj = self.mdp_obj.get_mrp(pol)
        value_func_vec = mrp_obj.get_value_func_vec()
        nt_vf = {mrp_obj.nt_states_list[i]: value_func_vec[i]
                 for i in range(len(mrp_obj.nt_states_list))}
        t_vf = {s: 0. for s in self.mdp_obj.terminal_states}
        return {**nt_vf, **t_vf}

    def get_optimal_det_policy(self) -> DetPolicy:
        return self.get_optimal_policy_pi()


if __name__ == '__main__':
    from processes.mdp import MDP
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
    gamma_val = 0.9
    mdp1_obj = MDP(mdp_data, gamma_val)
    mrp1_obj = mdp1_obj.get_mrp(pol_obj)
    print(mrp1_obj.transitions)
    print(mrp1_obj.rewards)
    print(mrp1_obj.trans_matrix)
    print(mrp1_obj.rewards_vec)
    print(mrp1_obj.get_value_func_vec())
    tol_val = 1e-4
    opn = DPAnalytic(mdp1_obj, tol_val)
    opt_policy_pi = opn.get_optimal_policy_pi()
    print(opt_policy_pi)
    opt_vf_dict_pi = opn.get_value_func_dict(opt_policy_pi)
    print(opt_vf_dict_pi)
    opt_policy_vi = opn.get_optimal_policy_vi()
    print(opt_policy_vi)
    opt_vf_dict_vi = opn.get_value_func_dict(opt_policy_vi)
    print(opt_vf_dict_vi)
