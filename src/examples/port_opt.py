from typing import Tuple, Generic, Mapping, Sequence, Callable
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

StateType = Tuple[int, float]
ActionType = Tuple[float, ...]


class PortOpt(Generic[StateType, ActionType]):

    @staticmethod
    def validate_spec(
        num_risky_assets: int,
        riskless_returns_seq: Sequence[float],
        returns_gen: Sequence[Callable[int], np.ndarray],
        disc_fac: float
    ) -> bool:
        b1 = num_risky_assets >= 1
        b2 = all(x > 0 for x in riskless_returns_seq)
        b3 = len(riskless_returns_seq) == len(returns_gen)
        b4 = 0. <= disc_fac <= 1.
        return all([b1, b2, b3, b4])

    def __init__(
        self,
        num_risky: int,
        riskless_returns: Sequence[float],
        returns_gen_funcs: Sequence[Callable[int], np.ndarray],
        cons_util_func: Callable[[float], float],
        beq_util_func: Callable[[float], float],
        discount_factor: float
    ) -> None:
        if PortOpt.validate_spec(
            num_risky,
            riskless_returns,
            discount_factor
        ):
            self.num_risky = num_risky
            self.riskless_returns = riskless_returns
            self.epochs = len(riskless_returns)
            self.returns_gen_funcs = returns_gen_funcs
            self.cons_util_func = cons_util_func
            self.beq_utils_func = beq_util_func

    # Epoch t is from time t to time (t+1), for 0 <= t < T
    # where T = number of epochs. At time T (i.e., at end of epoch
    # (T-1)), the process ends and W_T is the quantity of bequest.
    # The order of operations in an epoch t (0 <= t < T) is:
    # 1) Observe the state (t, W_t).
    # 2) Consume C_t . W_t so wealth drops to W_t . (1 - C_t)
    # 3) Allocate W_t . (1 - C_t) to n risky assets and 1 riskless asset
    # 4) Riskless asset grows by r_t and risky assets grow stochastically
    #    with wealth growing from W_t . (1 - C_t) to W_{t+1}
    # 5) At the end of final epoch (T-1) (i.e., at time T), bequest W_T.
    #
    # U_Beq(W_T) is the utility of bequest, U_Cons(W_t) is the utility
    # State at the start of epoch t is (t, W_t)
    # Action upon observation of state (t, W_t) is (C_t, A_1, .. A_n)
    # where 0 <= C_t <= 1 is the consumption and A_i, i = 1 .. n, is the
    # allocation to risky assets. Allocation to riskless asset will be set to
    # A_0 = 1 - \sum_{i=1}^n A_i. If stochastic return of risky asset i is R_i:
    # W_{t+1} = W_t . (1 - C_t) . (A_0 . (1+r) + \sum_{i=1}^n A_i . (1 + R_i))

    def init_state(self) -> StateType:
        return (0, 1.)

    def state_reward_gen(
        self,
        state: StateType,
        action: ActionType,
        num_samples: int
    ) -> Sequence[Tuple[StateType, float]]:
        t, W = state
        if t == self.epochs:
            ret = [((t, 0.), self.beq_utils_func(W))] * num_samples
        else:
            cons = action[0]
            risky_alloc = action[1:]
            riskless_alloc = 1. - sum(risky_alloc)
            alloc = np.insert(np.array(risky_alloc), 0, riskless_alloc)
            ret_samples = np.hstack((
                np.full((num_samples, 1), self.riskless_returns[t]),
                self.returns_gen_funcs[t](num_samples)
            ))
            W1 = W * (1 - cons)
            ret = [((t + 1, W1 * alloc.dot(1 + rs)), self.cons_util_func(cons))
                   for rs in ret_samples]
        return ret


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
    this_fa_spec = FuncApproxSpec(
        state_feature_funcs=FuncApproxBase.get_identity_feature_funcs(
            ic.lead_time + 1
        ),
        action_feature_funcs=[lambda x: x],
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