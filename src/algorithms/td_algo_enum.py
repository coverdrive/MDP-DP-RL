from enum import Enum, auto


class TDAlgorithm(Enum):
    SARSA = auto()
    QLearning = auto()
    ExpectedSARSA = auto()
