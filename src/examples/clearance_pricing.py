from scipy.stats import poisson
from algorithms.backward_dp import BackwardDP
from typing import List, Tuple, Mapping
import numpy as np


def get_clearance_backward_dp(
    time_steps: int,
    init_inv: int,
    base_price: float,
    base_demand: float,
    el: List[Tuple[float, float]],  # (price, poisson mean) pairs
) -> BackwardDP:

    aug_el = [(0., 0.)] + el
    rvs = [poisson(base_demand * (1 + l)) for _, l in aug_el]
    num_el = len(aug_el)

    tr_rew_dict = {
        (s, p): {
            p1: {
                (s - d, p1): (
                    rvs[p1].pmf(d) if d < s else 1. - rvs[p1].cdf(s - 1),
                    d * base_price * (1 - aug_el[p1][0])
                ) for d in range(s + 1)
            } for p1 in range(p, num_el)
        } for s in range(init_inv + 1) for p in range(num_el)
    }
    return BackwardDP(
        transitions_rewards=[tr_rew_dict] * time_steps,
        terminal_opt_val={(s, p): 0. for s in range(init_inv + 1)
                          for p in range(num_el)},
        gamma=1.
    )


def get_performance(
    time_steps: int,
    init_inv: int,
    base_price: float,
    base_demand: float,
    el: List[Tuple[float, float]],
    num_traces: int
) -> Mapping[str, float]:
    vf_and_pol = get_clearance_backward_dp(
        time_steps,
        init_inv,
        base_price,
        base_demand,
        el
    ).vf_and_policy
    opt_vf = vf_and_pol[0][(init_inv, 0)][0]

    aug_el = [(0., 0.)] + el
    rvs = [poisson(base_demand * (1 + l)) for _, l in aug_el]

    all_revs = np.zeros(num_traces)
    all_leftovers = np.zeros(num_traces)
    for i in range(num_traces):
        rev = 0.
        state = (init_inv, 0)
        for t in range(time_steps):
            action = vf_and_pol[t][state][1]
            price = base_price * (1 - aug_el[action][0])
            demand = rvs[action].rvs()
            rev += min(state[0], demand) * price
            state = (max(0, state[0] - demand), action)
        all_revs[i] = rev
        all_leftovers[i] = state[0]

    total_value = init_inv * base_price
    salvage = np.mean(all_leftovers) * base_price
    revenue = np.mean(all_revs)
    a_markdown = total_value - salvage - revenue

    return {
        "Optimal VF": opt_vf,
        "Total Value": total_value,
        "Revenue": revenue,
        "A Markdown": a_markdown,
        "Salvage": salvage
    }


if __name__ == '__main__':
    ts: int = 6  # time steps
    ii: int = 20  # initial inventory
    bp: float = 10.0  # base price
    bd: float = 1.1  # base demand
    this_el: List[Tuple[float, float]] = [
        (0.3, 1.2), (0.5, 2.1), (0.7, 2.8)
    ]
    # bdp = get_clearance_backward_dp(ts, ii, bp, bd, this_el)
    #
    # for i in range(ts):
    #     print([(x, y) for x, (y, _) in bdp.vf_and_policy[i].items()])
    # for i in range(ts):
    #     print([(x, z) for x, (_, z) in bdp.vf_and_policy[i].items()])

    traces = 10000
    perf = get_performance(ts, ii, bp, bd, this_el, traces)
    print(perf)
