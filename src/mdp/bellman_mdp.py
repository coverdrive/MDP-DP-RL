from typing import Dict


class BellmanMDP(object):
    def __init__(self, tr, rew):
        self.transitions = tr
        self.rewards = rew


if __name__ == '__main__':
    transitions = {1: {1: 0.3, 2: 0.6, 3: 0.1}, 2: {1: 0.4, 2: 0.2, 3: 0.4}, 3: {1: 0.6, 2: 0.4}}
    rewards = {1: 3.0, 2: 0.4, 3: -0.3}
