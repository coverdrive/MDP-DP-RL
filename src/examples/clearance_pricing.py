from scipy.stats import poisson
from algorithms.backward_dp import BackwardDP
from typing import List, Tuple, Mapping, Any, Sequence
from matplotlib.ticker import PercentFormatter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def get_clearance_backward_dp(
    time_steps: int,
    init_inv: int,
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
                    d * (1 - aug_el[p1][0])
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
    base_demand: float,
    el: List[Tuple[float, float]],
    num_traces: int
) -> Mapping[str, Any]:
    vf_and_pol = get_clearance_backward_dp(
        time_steps,
        init_inv,
        base_demand,
        el
    ).vf_and_policy
    opt_vf = vf_and_pol[0][(init_inv, 0)][0]

    aug_el = [(0., 0.)] + el
    rvs = [poisson(base_demand * (1 + l)) for _, l in aug_el]

    all_revs = np.empty(num_traces)
    all_rem = np.empty((num_traces, time_steps))
    all_actions = np.empty((num_traces, time_steps))
    for i in range(num_traces):
        rev = 0.
        state = (init_inv, 0)
        for t in range(time_steps):
            action = vf_and_pol[t][state][1]
            price = 1 - aug_el[action][0]
            demand = rvs[action].rvs()
            rev += (min(state[0], demand) * price)
            state = (max(0, state[0] - demand), action)
            all_rem[i, t] = state[0]
            all_actions[i, t] = aug_el[action][0]
        all_revs[i] = rev

    remaining = np.mean(all_rem, axis=0)
    salvage = remaining[-1]
    revenue = np.mean(all_revs)
    a_markdown = init_inv - salvage - revenue
    actions = np.mean(all_actions, axis=0)

    return {
        "Optimal VF": opt_vf,
        "Revenue": revenue / init_inv,
        "AMarkdown": a_markdown / init_inv,
        "Salvage": salvage / init_inv,
        "Remaining": [r / init_inv for r in remaining],
        "Actions": actions
    }


def graph_perf(
    time_steps: int,
    demand: float,
    inv: Sequence[int],
    elasticity: Tuple[float, float, float]
) -> None:
    revs = []
    ams = []
    sals = []
    for initial_inv in inv:
        perf = get_performance(
            time_steps,
            initial_inv,
            demand,
            list(zip((0.3, 0.5, 0.7), elasticity)),
            10000
        )
        revs.append(perf["Revenue"] * 100)
        ams.append(perf["AMarkdown"] * 100)
        sals.append(perf["Salvage"] * 100)
    plt.grid()
    plt.plot(inv, revs, "k", label="Revenue")
    plt.plot(inv, ams, "b", label="A-Markdown")
    plt.plot(inv, sals, "r", label="Salvage")
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.xlabel("Initial Inventory", fontsize=10)
    plt.ylabel("Percentage of Initial Value", fontsize=10)
    tup = (
        time_steps,
        demand,
        elasticity[0] * 100,
        elasticity[1] * 100,
        elasticity[2] * 100
    )
    plt.title(
        "Weeks=%d,WeeklyDemand=%.1f,Elasticity=[%d,%d,%d]" % tup,
        fontsize=10
    )
    plt.legend(loc="upper right")
    file_name = str(Path.home()) + ("/wks=%d&dem=%d&el=%d-%d-%d.png" % tup)
    print(file_name)
    plt.savefig(file_name)
    plt.close()


if __name__ == '__main__':
    # ts: int = 8  # time steps
    # ii: int = 12  # initial inventory
    # bd: float = 1.0  # base demand
    # this_el: List[Tuple[float, float]] = [
    #     (0.3, 0.5), (0.5, 1.1), (0.7, 1.4)
    # ]
    # bdp = get_clearance_backward_dp(ts, ii, bd, this_el)
    #
    # for i in range(ts):
    #     print([(x, y) for x, (y, _) in bdp.vf_and_policy[i].items()])
    # for i in range(ts):
    #     print([(x, z) for x, (_, z) in bdp.vf_and_policy[i].items()])

    # traces = 10000
    # per = get_performance(ts, ii, bd, this_el, traces)
    # print(per)

    ts: int = 8  # time steps
    bd: float = 1.0  # base demand
    invs: Sequence[int] = list(range(2, 30, 2))

    elasticities = [
        (0.1, 0.3, 0.5),
        (0.3, 0.7, 1.0),
        (0.5, 0.8, 1.1),
        (0.7, 1.2, 1.5),
        (0.8, 1.3, 1.7),
        (1.0, 1.5, 2.0),
        (1.0, 2.0, 2.5),
        (1.5, 2.5, 3.5),
        (2.0, 4.0, 6.0)
    ]
    for els in elasticities:
        graph_perf(ts, bd, invs, els)
