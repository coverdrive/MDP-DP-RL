from typing import Tuple, Sequence, NamedTuple, Set, Mapping
from enum import Enum
from scipy.stats import norm
from processes.mdp_refined import MDPRefined

Node = Tuple[int, int]
NodeSet = Set[Node]
WindSpec = Sequence[Tuple[float, float]]


class Move(Enum):
    U = (0, 1)
    D = (0, -1)
    L = (-1, 0)
    R = (1, 0)
    S = (0, 0)


class WindyGrid(NamedTuple):
    x_len: int
    y_len: int
    blocks: NodeSet
    terminals: NodeSet
    wind: WindSpec
    edge_bump_cost: float
    block_bump_cost: float

    def validate_spec(self) -> bool:
        b1 = self.x_len >= 2
        b2 = self.y_len >= 2
        b3 = all(0 <= x < self.x_len and 0 <= y < self.y_len
                 for x, y in self.blocks)
        b4 = len(self.terminals) >= 1
        b5 = all(0 <= x < self.x_len and 0 <= y < self.y_len
                 for x, y in self.terminals)
        b6 = len(self.wind) == self.x_len
        b7 = all(y >= 0 for _, y in self.wind)
        b8 = self.edge_bump_cost > 1
        b9 = self.block_bump_cost > 1
        return all([b1, b2, b3, b4, b5, b6, b7, b8, b9])

    @staticmethod
    def add_tuples(a: Node, b: Node) -> Node:
        return a[0] + b[0], a[1] + b[1]

    def is_valid_state(self, state: Node) -> bool:
        return 0 <= state[0] < self.x_len\
               and 0 <= state[1] < self.y_len\
               and state not in self.blocks

    def get_all_nt_states(self) -> NodeSet:
        return {(i, j) for i in range(self.x_len) for j in range(self.y_len)
                if (i, j) not in set.union(self.blocks, self.terminals)}

    def get_actions_and_next_states(self, state: Node)\
            -> Set[Tuple[Move, Node]]:
        temp = {(a.name, WindyGrid.add_tuples(state, a.value)) for a in Move}
        return {(a, s) for a, s in temp if self.is_valid_state(s)}

    def get_state_probs_and_rewards(self, state: Node)\
            -> Mapping[Node, Tuple[float, float]]:
        state_x, state_y = state
        barriers = set.union(
            {-1, self.y_len},
            {y for x, y in self.blocks if x == state_x}
        )
        lower = max(y for y in barriers if y < state_y) + 1
        upper = min(y for y in barriers if y > state_y) - 1
        mu, sigma = self.wind[state_x]
        if sigma == 0:
            only_state = round(state_y + mu)
            if lower <= only_state <= upper:
                cost = 0.
            elif only_state < lower:
                cost = self.edge_bump_cost if lower == 0\
                    else self.block_bump_cost
            else:
                cost = self.edge_bump_cost if upper == self.y_len - 1\
                    else self.block_bump_cost
            ret = {(state_x, max(lower, min(upper, only_state))):
                   (1., -(1. + cost))}
        else:
            rv = norm(loc=mu, scale=sigma)
            temp_data = []
            for y in range(lower, upper + 1):
                if y == lower:
                    pr = rv.cdf(lower - state_y + 0.5)
                    pr1 = rv.cdf(lower - state_y - 0.5)
                    cost = pr1 / pr * (self.edge_bump_cost if lower == 0
                                       else self.block_bump_cost)\
                        if pr != 0. else 0.
                elif y == upper:
                    pr = 1. - rv.cdf(upper - state_y - 0.5)
                    pr1 = 1. - rv.cdf(upper - state_x + 0.5)
                    cost = pr1 / pr * (self.edge_bump_cost
                                       if upper == self.y_len - 1
                                       else self.block_bump_cost)\
                        if pr != 0. else 0.
                else:
                    pr = rv.cdf(y - state_y + 0.5) - rv.cdf(y - state_y - 0.5)
                    cost = 0.
                temp_data.append((y, pr, cost))
            sum_pr = sum(p for _, p, _ in temp_data)
            ret = {(state_x, y): (p / sum_pr, -(1. + c))
                   for y, p, c in temp_data}
        return ret

    def get_non_terminals_dict(self)\
            -> Mapping[Node, Mapping[Move, Mapping[Node, Tuple[float, float]]]]:
        return {s: {a: ({s1: (1., -1.)} if s1 in self.terminals else
                        self.get_state_probs_and_rewards(s1))
                    for a, s1 in self.get_actions_and_next_states(s)}
                for s in self.get_all_nt_states()}

    def get_mdp_refined_dict(self)\
            -> Mapping[Node, Mapping[Move, Mapping[Node, Tuple[float, float]]]]:
        d1 = self.get_non_terminals_dict()
        d2 = {s: {Move.S.name: {s: (1.0, 0.0)}} for s in self.terminals}
        return {**d1, **d2}

    def get_mdp_refined(self) -> MDPRefined:
        return MDPRefined(self.get_mdp_refined_dict(), gamma=1.)


if __name__ == '__main__':
    wg = WindyGrid(
        x_len=4,
        y_len=4,
        blocks={(2, 1)},
        terminals={(3, 2)},
        wind=[(0.5, 0.5), (0.3, 0.8), (0.2, 1.7), (-0.6, 0.2)],
        edge_bump_cost=3.,
        block_bump_cost=2.
    )
    valid = wg.validate_spec()
    print(valid)
    all_nt_states = wg.get_all_nt_states()
    print(all_nt_states)
    d = {s: wg.get_actions_and_next_states(s) for s in wg.get_all_nt_states()}
    print(d)
    print(d[(2, 0)])
    print(d[(1, 2)])
    print(d[(3, 3)])
    print(d[(0, 1)])
    spr = wg.get_state_probs_and_rewards((0, 2))
    print(spr)
    mdp_refined_dict = wg.get_mdp_refined_dict()
    print(mdp_refined_dict)
    mdp_refined_obj = wg.get_mdp_refined()
    opt_pol = mdp_refined_obj.get_optimal_policy()
    print(opt_pol)
    vf_dict = mdp_refined_obj.get_value_func_dict(opt_pol)
    print(vf_dict)
