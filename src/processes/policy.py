from typing import Mapping, TypeVar, Generic
from processes.mp_funcs import verify_policy

S = TypeVar('S')
A = TypeVar('A')


class Policy(Generic[S, A]):

    def __init__(self, data: Mapping[S, Mapping[A, float]]) -> None:
        if verify_policy(data):
            self.policy_data = data
        else:
            raise ValueError

    def get_state_probabilities(self, state: S) -> Mapping[A, float]:
        return self.policy_data[state]

    def get_state_action_probability(self, state: S, action: A) -> float:
        return self.get_state_probabilities(state).get(action, 0.)


