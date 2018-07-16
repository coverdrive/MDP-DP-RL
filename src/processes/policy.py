from typing import Mapping, TypeVar, Generic

S = TypeVar('S')
A = TypeVar('A')


class Policy(Generic[S, A]):

    def __init__(self, data: Mapping[S, Mapping[A, float]]) -> None:
        self.policy_data = data


