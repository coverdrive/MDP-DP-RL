from scipy.stats import poisson
from algorithms.backward_dp import BackwardDP
from typing import Sequence, Tuple


def get_clearance_backward_dp(
    time_steps: int,
    init_inv: int,
    el: Sequence[Tuple[float, float]],  # (price, poisson mean) pairs
) -> BackwardDP:

    rvs = [(p, poisson(l)) for p, l in el]

    tr_rew_dict = {
        s: {
            p: {
                s - d: (
                    rv.pmf(d) if d < s else 1. - rv.cdf(s - 1),
                    d * p
                ) for d in range(s + 1)
            } for p, rv in rvs
        } for s in range(init_inv + 1)
    }
    return BackwardDP(
        transitions_rewards=[tr_rew_dict] * time_steps,
        terminal_opt_val={s: 0. for s in range(init_inv + 1)},
        gamma=1.
    )


if __name__ == '__main__':
    ts: int = 50  # time steps
    ii: int = 10  # initial inventory
    this_el: Sequence[Tuple[float, float]] = [
        (10.0, 0.1), (9.0, 0.16), (8.0, 0.22),
        (7.0, 0.28), (6.0, 0.38), (5.0, 0.5)
    ]
    bdp = get_clearance_backward_dp(ts, ii, this_el)

    for i in range(ts):
        print([(x, y) for x, (y, _) in bdp.vf_and_policy[i].items()])
    for i in range(ts):
        print([(x, z) for x, (_, z) in bdp.vf_and_policy[i].items()])

