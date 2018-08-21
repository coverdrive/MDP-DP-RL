from typing import Mapping, Generic, Dict
from processes.mp_funcs import verify_policy
from processes.mp_funcs import get_epsilon_action_probs
from processes.mp_funcs import get_softmax_action_probs
from utils.generic_typevars import S, A


class Policy(Generic[S, A]):

    def __init__(self, data: Dict[S, Mapping[A, float]]) -> None:
        if verify_policy(data):
            self.policy_data = data
        else:
            raise ValueError

    def get_state_probabilities(self, state: S) -> Mapping[A, float]:
        return self.policy_data[state]

    def get_state_action_probability(self, state: S, action: A) -> float:
        return self.get_state_probabilities(state).get(action, 0.)

    def edit_state_action_to_epsilon_greedy(
        self,
        state: S,
        action_value_dict: Mapping[A, float],
        epsilon: float
    ) -> None:
        self.policy_data[state] = get_epsilon_action_probs(
            action_value_dict,
            epsilon
        )

    def edit_state_action_to_softmax(
            self,
            state: S,
            action_value_dict: Mapping[A, float]
    ) -> None:
        self.policy_data[state] = get_softmax_action_probs(
            action_value_dict
        )

    def __repr__(self):
        return self.policy_data.__repr__()

    def __str__(self):
        return self.policy_data.__str__()


