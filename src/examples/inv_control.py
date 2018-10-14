from typing import Tuple, NamedTuple, Set, Mapping, Sequence
from itertools import chain, product, groupby
import numpy as np
from numpy.core.multiarray import ndarray
from scipy.stats import poisson
from processes.mdp_refined import MDPRefined
from func_approx.dnn_spec import DNNSpec
from func_approx.func_approx_base import FuncApproxBase
from algorithms.func_approx_spec import FuncApproxSpec
from copy import deepcopy
from operator import itemgetter
from processes.det_policy import DetPolicy
from examples.run_all_algorithms import RunAllAlgorithms

StateType = Tuple[int, ...]


class InvControl(NamedTuple):
    demand_lambda: float
    lead_time: int
    stockout_cost: float
    fixed_order_cost: float
    epoch_disc_factor: float
    order_limit: int
    space_limit: int
    throwout_cost: float
    stockout_limit: int
    stockout_limit_excess_cost: float

    def validate_spec(self) -> bool:
        b1 = self.demand_lambda > 0.
        b2 = self.lead_time >= 0
        b3 = self.stockout_cost > 1.
        b4 = self.fixed_order_cost >= 0.
        b5 = 0. <= self.epoch_disc_factor <= 1.
        b6 = self.order_limit > 0
        b7 = self.space_limit > 0
        b8 = self.throwout_cost > 1.
        b9 = self.stockout_limit > 0.
        b10 = self.stockout_limit_excess_cost > 0.
        return all([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10])

    def get_all_states(self) -> Set[StateType]:
        on_hand_range = range(-self.stockout_limit, self.space_limit + 1)
        on_order_range = range(self.order_limit + 1)
        return set(product(
            *chain([on_hand_range], [on_order_range] * self.lead_time)
        ))

    # Order of operations in an epoch are:
    # 1) Order Placement (Action)
    # 2) Receipt
    # 3) Throwout Space-Limited-Excess Inventory
    # 4) Demand
    # 5) Adjust (Negative) Inventory to not fall below stockout limit

    # In the following func, the input "state" is represented by
    # the on-hand and on-order right before an order is placed (the very
    # first event in the epoch) and the "state"s in the output are represented
    # by the  on-hand and on-order just before the next order is placed (in the
    # next epoch).  Both the input and output "state"s are arrays of length (L+1).

    def get_next_states_probs_rewards(
            self,
            state: StateType,
            action: int,
            demand_probs: Sequence[float]
    ) -> Mapping[StateType, Tuple[float, float]]:
        next_state_arr: ndarray = np.array(state)
        # The next line represents state change due to Action and Receipt
        next_state_arr = np.insert(
            np.zeros(len(next_state_arr) - 1),
            0,
            next_state_arr[0]
        ) + np.append(next_state_arr[1:], action)
        excess = max(0, next_state_arr[0] - self.space_limit)
        cost = (self.fixed_order_cost if action > 0 else 0.) + \
            excess * self.throwout_cost
        # The next line represents throwing out excess inventory
        next_state_arr[0] -= excess
        # The next line represents state change due to demand
        temp_list = []
        for demand, prob in enumerate(demand_probs):
            ns = deepcopy(next_state_arr)
            ns[0] -= demand
            excess_stockout = max(0, -self.stockout_limit - ns[0])
            this_cost = cost + excess_stockout * \
                (self.stockout_cost + self.stockout_limit_excess_cost)
            # the next line represents adjustment of negative inventory
            # to not fall below stockout limit
            ns[0] += excess_stockout
            inv = ns[0]
            onhand = max(0., inv)
            stockout = max(0., -inv)
            this_cost += (onhand + self.stockout_cost * stockout)
            ns_tup = tuple(int(x) for x in ns)
            temp_list.append((ns_tup, prob, -this_cost))

        ret = {}
        crit = itemgetter(0)
        for s, v in groupby(sorted(temp_list, key=crit), key=crit):
            tl = [(p, r) for _, p, r in v]
            sum_p = sum(p for p, _ in tl)
            avg_r = sum(p * r for p, r in tl) / sum_p if sum_p != 0. else 0.
            ret[s] = (sum_p, avg_r)
        return ret

    def get_mdp_refined_dict(self) \
            -> Mapping[StateType,
                       Mapping[int,
                               Mapping[StateType,
                                       Tuple[float, float]]]]:
        rv = poisson(mu=self.demand_lambda)
        raw_probs = [rv.pmf(i) for i in range(int(rv.ppf(0.9999)))]
        pp = [p / sum(raw_probs) for p in raw_probs]
        return {s: {a: self.get_next_states_probs_rewards(s, a, pp)
                    for a in range(self.order_limit + 1)}
                for s in self.get_all_states()}

    def get_mdp_refined(self) -> MDPRefined:
        return MDPRefined(self.get_mdp_refined_dict(), self.epoch_disc_factor)

    def get_optimal_policy(self) -> DetPolicy:
        return self.get_mdp_refined().get_optimal_policy()

    def get_ips_orders_dict(self) -> Mapping[int, Sequence[int]]:
        sa_pairs = self.get_optimal_policy().get_state_to_action_map().items()

        def crit(x: Tuple[Tuple[int, ...], int]) -> int:
            return sum(x[0])

        return {ip: [y for _, y in v] for ip, v in
                groupby(sorted(sa_pairs, key=crit), key=crit)}


if __name__ == '__main__':

    ic = InvControl(
        demand_lambda=0.5,
        lead_time=1,
        stockout_cost=49.,
        fixed_order_cost=0.0,
        epoch_disc_factor=0.98,
        order_limit=7,
        space_limit=8,
        throwout_cost=30.,
        stockout_limit=5,
        stockout_limit_excess_cost=30.
    )
    valid = ic.validate_spec()
    mdp_ref_obj = ic.get_mdp_refined()
    this_tolerance = 1e-3
    exploring_start = False
    this_first_visit_mc = True
    num_samples = 30
    this_softmax = True
    this_epsilon = 0.05
    this_epsilon_half_life = 30
    this_learning_rate = 0.1
    this_learning_rate_decay = 1e6
    this_lambd = 0.8
    this_num_episodes = 3000
    this_max_steps = 1000
    this_tdl_fa_offline = True
    state_ffs = FuncApproxBase.get_identity_feature_funcs(ic.lead_time + 1)
    sa_ffs = [(lambda x, f=f: f(x[0])) for f in state_ffs] + [lambda x: x[1]]
    this_fa_spec = FuncApproxSpec(
        state_feature_funcs=state_ffs,
        sa_feature_funcs=sa_ffs,
        dnn_spec=DNNSpec(
            neurons=[2, 4],
            hidden_activation=DNNSpec.relu,
            hidden_activation_deriv=DNNSpec.relu_deriv,
            output_activation=DNNSpec.identity,
            output_activation_deriv=DNNSpec.identity_deriv
        )
    )

    raa = RunAllAlgorithms(
        mdp_refined=mdp_ref_obj,
        tolerance=this_tolerance,
        exploring_start=exploring_start,
        first_visit_mc=this_first_visit_mc,
        num_samples=num_samples,
        softmax=this_softmax,
        epsilon=this_epsilon,
        epsilon_half_life=this_epsilon_half_life,
        learning_rate=this_learning_rate,
        learning_rate_decay=this_learning_rate_decay,
        lambd=this_lambd,
        num_episodes=this_num_episodes,
        max_steps=this_max_steps,
        tdl_fa_offline=this_tdl_fa_offline,
        fa_spec=this_fa_spec
    )

    def criter(x: Tuple[Tuple[int, ...], int]) -> int:
        return sum(x[0])

    for st, mo in raa.get_all_algorithms().items():
        print("Starting %s" % st)
        opt_pol_func = mo.get_optimal_det_policy_func()
        opt_pol = {s: opt_pol_func(s) for s in mdp_ref_obj.all_states}
        print(sorted(
            [(ip, np.mean([float(y) for _, y in v])) for ip, v in
             groupby(sorted(opt_pol.items(), key=criter), key=criter)],
            key=itemgetter(0)
        ))
