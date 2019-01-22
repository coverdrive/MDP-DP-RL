from typing import Tuple, Sequence, NamedTuple, Set, Mapping
from enum import Enum
from scipy.stats import norm
from processes.mdp_refined import MDPRefined
from processes.det_policy import DetPolicy
from func_approx.func_approx_base import FuncApproxBase
from algorithms.func_approx_spec import FuncApproxSpec
from examples.run_all_algorithms import RunAllAlgorithms

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
        return 0 <= state[0] < self.x_len \
               and 0 <= state[1] < self.y_len \
               and state not in self.blocks

    def get_all_nt_states(self) -> NodeSet:
        return {(i, j) for i in range(self.x_len) for j in range(self.y_len)
                if (i, j) not in set.union(self.blocks, self.terminals)}

    def get_actions_and_next_states(self, nt_state: Node) \
            -> Set[Tuple[Move, Node]]:
        temp = {(a.name, WindyGrid.add_tuples(nt_state, a.value))
                for a in Move if a != Move.S}
        return {(a, s) for a, s in temp if self.is_valid_state(s)}

    def get_state_probs_and_rewards(self, state: Node) \
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
                cost = self.edge_bump_cost if lower == 0 \
                    else self.block_bump_cost
            else:
                cost = self.edge_bump_cost if upper == self.y_len - 1 \
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
                                       else self.block_bump_cost) \
                        if pr != 0. else 0.
                elif y == upper:
                    pr = 1. - rv.cdf(upper - state_y - 0.5)
                    pr1 = 1. - rv.cdf(upper - state_x + 0.5)
                    cost = pr1 / pr * (self.edge_bump_cost
                                       if upper == self.y_len - 1
                                       else self.block_bump_cost) \
                        if pr != 0. else 0.
                else:
                    pr = rv.cdf(y - state_y + 0.5) - rv.cdf(y - state_y - 0.5)
                    cost = 0.
                temp_data.append((y, pr, cost))
            sum_pr = sum(p for _, p, _ in temp_data)
            ret = {(state_x, y): (p / sum_pr, -(1. + c))
                   for y, p, c in temp_data}
        return ret

    def get_non_terminals_dict(self) \
            -> Mapping[Node, Mapping[Move, Mapping[Node, Tuple[float, float]]]]:
        return {s: {a: ({s1: (1., -1.)} if s1 in self.terminals else
                        self.get_state_probs_and_rewards(s1))
                    for a, s1 in self.get_actions_and_next_states(s)}
                for s in self.get_all_nt_states()}

    def get_mdp_refined_dict(self) \
            -> Mapping[Node, Mapping[Move, Mapping[Node, Tuple[float, float]]]]:
        d1 = self.get_non_terminals_dict()
        d2 = {s: {Move.S.name: {s: (1.0, 0.0)}} for s in self.terminals}
        return {**d1, **d2}

    def get_mdp_refined(self) -> MDPRefined:
        return MDPRefined(self.get_mdp_refined_dict(), gamma=1.)

    def print_vf(self, vf_dict, chars: int, decimals: int) -> None:
        display = "%%%d.%df" % (chars, decimals)
        display1 = "%%%dd" % chars
        display2 = "%%%dd " % 2
        blocks_dict = {s: 'X' * chars for s in self.blocks}
        non_blocks_dict = {s: display % -v for s, v in vf_dict.items()}
        full_dict = {**non_blocks_dict, **blocks_dict}
        print("   " + " ".join([display1 % j for j in range(0, self.x_len)]))
        for i in range(self.y_len - 1, -1, -1):
            print(display2 % i + " ".join(full_dict[(j, i)]
                                          for j in range(0, self.x_len)))

    def print_policy(self, pol: DetPolicy) -> None:
        display1 = "%%%dd" % 2
        display2 = "%%%dd  " % 2
        blocks_dict = {s: 'X' for s in self.blocks}
        full_dict = {**pol.get_state_to_action_map(), **blocks_dict}
        print("   " + " ".join([display1 % j for j in range(0, self.x_len)]))
        for i in range(self.y_len - 1, -1, -1):
            print(display2 % i + "  ".join(full_dict[(j, i)]
                                           for j in range(0, self.x_len)))

    def print_wind_and_bumps(self, chars: int, decimals: int) -> None:
        display = "%%%d.%df" % (chars, decimals)
        print("mu " + " ".join(display % m for m, _ in self.wind))
        print("sd " + " ".join(display % s for _, s in self.wind))
        print("Block Bump Cost = %5.2f" % self.block_bump_cost)
        print("Edge Bump Cost = %5.2f" % self.edge_bump_cost)


if __name__ == '__main__':
    wg = WindyGrid(
        x_len=6,
        y_len=9,
        blocks={(1, 5), (2, 1), (2, 2), (2, 3), (4, 4), (4, 5), (4, 6), (4, 7)},
        terminals={(5, 7)},
        wind=[(0., 0.), (0., 0.), (-1.2, 0.3), (-1.7, 0.7), (0.6, 0.4), (0.5, 1.2)],
        edge_bump_cost=3.,
        block_bump_cost=4.
    )
    valid = wg.validate_spec()
    mdp_ref_obj = wg.get_mdp_refined()
    this_tolerance = 1e-3
    exploring_start = False
    this_first_visit_mc = True
    this_num_samples = 30
    this_softmax = False
    this_epsilon = 0.0
    this_epsilon_half_life = 100
    this_learning_rate = 0.1
    this_learning_rate_decay = 1e6
    this_lambd = 0.8
    this_num_episodes = 1000
    this_batch_size = 10
    this_max_steps = 1000
    this_td_offline = True
    state_ffs = FuncApproxBase.get_indicator_feature_funcs(mdp_ref_obj.all_states)
    sa_ffs = [(lambda x, f=f: f(x[0])) for f in state_ffs] +\
        [(lambda x, f=f: f(x[1])) for f in FuncApproxBase.get_indicator_feature_funcs(
            {m.name for m in Move}
        )]
    this_fa_spec = FuncApproxSpec(
        state_feature_funcs=state_ffs,
        sa_feature_funcs=sa_ffs,
        dnn_spec=None
        # dnn_spec=DNNSpec(
        #     neurons=[2, 4],
        #     hidden_activation=DNNSpec.relu,
        #     hidden_activation_deriv=DNNSpec.relu_deriv,
        #     output_activation=DNNSpec.identity,
        #     output_activation_deriv=DNNSpec.identity_deriv
        # )
    )

    raa = RunAllAlgorithms(
        mdp_refined=mdp_ref_obj,
        tolerance=this_tolerance,
        exploring_start=exploring_start,
        first_visit_mc=this_first_visit_mc,
        num_samples=this_num_samples,
        softmax=this_softmax,
        epsilon=this_epsilon,
        epsilon_half_life=this_epsilon_half_life,
        learning_rate=this_learning_rate,
        learning_rate_decay=this_learning_rate_decay,
        lambd=this_lambd,
        num_episodes=this_num_episodes,
        batch_size=this_batch_size,
        max_steps=this_max_steps,
        tdl_fa_offline=this_td_offline,
        fa_spec=this_fa_spec
    )
    for name, algo in raa.get_all_algorithms().items():
        print(name)
        opt_pol_func = algo.get_optimal_det_policy_func()
        opt_pol = DetPolicy({s: opt_pol_func(s) for s in mdp_ref_obj.all_states})
        opt_vf_func = algo.get_optimal_value_func()
        opt_vf_dict = {s: opt_vf_func(s) for s in mdp_ref_obj.all_states}
        wg.print_policy(opt_pol)
        chars_count = 5
        decimals_count = 2
        print()
        wg.print_vf(opt_vf_dict, chars_count, decimals_count)
        print()
        wg.print_wind_and_bumps(chars_count, decimals_count)
        print()
        print()

