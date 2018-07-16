from typing import Mapping, TypeVar
from processes.policy import Policy

S = TypeVar('S')
A = TypeVar('A')


class DetPolicy(Policy):

    def __init__(self, det_policy_data: Mapping[S, A]) -> None:
        super().__init__({s: {a: 1.0} for s, a in det_policy_data.items()})

