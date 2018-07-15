from typing import Mapping, TypeVar, Generic
from processes.mp_funcs import mdp_rep_to_mrp_rep1, mdp_rep_to_mrp_rep2
from processes.mdp import MDP
from processes.mdp_refined import MDPRefined
from processes.mrp import MRP
from processes.mrp_refined import MRPRefined

S = TypeVar('S')
A = TypeVar('A')


class Policy(Generic[S, A]):

    def __init__(self, data: Mapping[S, Mapping[A, float]]) -> None:
        self.policy_data = data

    def get_mrp(self, mdp: MDP) -> MRP:
        tr = mdp_rep_to_mrp_rep1(mdp.transitions, self.policy_data)
        rew = mdp_rep_to_mrp_rep2(mdp.rewards, self.policy_data)
        return MRP({s: (v, rew[s]) for s, v in tr.items()})

    def get_mrp_refined(self, mdp_refined: MDPRefined) -> MRPRefined:
        tr = mdp_rep_to_mrp_rep1(mdp_refined.transitions, self.policy_data)
        rew_ref = mdp_rep_to_mrp_rep1(
            mdp_refined.rewards_refined,
            self.policy_data
        )
        return MRPRefined({s: {s1: (v1, rew_ref[s][s1]) for s1, v1 in v.items()}
                           for s, v in tr.items()})


if __name__ == '__main__':
    policy_data = {
        1: {'a': 0.4, 'b': 0.6},
        2: {'a': 0.7, 'c': 0.3},
        3: {'b': 1.0}
    }
    policy = Policy(policy_data)
    mdp_data = {
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
    mdp1 = MDP(mdp_data)
    mrp = policy.get_mrp(mdp1)
    print(mrp.transitions)
    print(mrp.rewards)
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
    mdp2 = MDPRefined(mdp_refined_data)
    mrp_refined = policy.get_mrp_refined(mdp2)
    print(mrp_refined.transitions)
    print(mrp_refined.rewards_refined)
