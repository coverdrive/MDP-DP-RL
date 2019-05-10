from scipy.stats import poisson
from algorithms.backward_dp import BackwardDP
from typing import List, Tuple


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
                    d * base_price * (1. - aug_el[p1][0])
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


if __name__ == '__main__':
    ts: int = 6  # time steps
    ii: int = 5  # initial inventory
    bp: float = 10.0  # base price
    bd: float = 1.1  # base demand
    this_el: List[Tuple[float, float]] = [
        (0.3, 0.5), (0.5, 0.8), (0.7, 1.0)
    ]
    bdp = get_clearance_backward_dp(ts, ii, bp, bd, this_el)

    for i in range(ts):
        print([(x, y) for x, (y, _) in bdp.vf_and_policy[i].items()])
    for i in range(ts):
        print([(x, z) for x, (_, z) in bdp.vf_and_policy[i].items()])

