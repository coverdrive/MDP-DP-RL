from typing import Mapping, TypeVar
from processes.policy import Policy
from processes.mdp import MDP
from processes.mdp_refined import MDPRefined

S = TypeVar('S')
A = TypeVar('A')


class DetPolicy(Policy):

    def __init__(self, det_policy_data: Mapping[S, A]) -> None:
        super().__init__({s: {a: 1.0} for s, a in det_policy_data.items()})


if __name__ == '__main__':
    policy_data = {1: 'a', 2: 'c', 3: 'b'}
    policy = DetPolicy(policy_data)
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
