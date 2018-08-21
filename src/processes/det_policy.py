from typing import Mapping
from processes.policy import Policy
from utils.generic_typevars import S, A


class DetPolicy(Policy):

    def __init__(self, det_policy_data: Mapping[S, A]) -> None:
        super().__init__({s: {a: 1.0} for s, a in det_policy_data.items()})

    def get_action_for_state(self, state: S) -> A:
        return list(self.get_state_probabilities(state).keys())[0]

    def get_state_to_action_map(self) -> Mapping[S, A]:
        return {s: self.get_action_for_state(s) for s in self.policy_data}

    def __repr__(self) -> str:
        return self.get_state_to_action_map().__repr__()

    def __str__(self) -> str:
        return self.get_state_to_action_map().__str__()

