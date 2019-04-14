from typing import Mapping, Sequence, Tuple, Generic
from utils.gen_utils import is_approx_eq
from utils.generic_typevars import S, A
from utils.standard_typevars import SASTff
from operator import itemgetter
from scipy.stats import poisson


class BackwardDP(Generic[S, A]):

    def __init__(
        self,
        transitions_rewards: Sequence[SASTff],
        terminal_opt_val: Mapping[S, float],
        gamma: float
    ) -> None:
        if BackwardDP.verify_data(transitions_rewards, terminal_opt_val, gamma):
            self.transitions_rewards = transitions_rewards
            self.terminal_opt_val = terminal_opt_val
            self.gamma = gamma
            self.vf_and_policy = self.get_vf_and_policy()
        else:
            raise ValueError

    @staticmethod
    def verify_data(
        transitions_rewards: Sequence[SASTff],
        terminal_opt_val: Mapping[S, float],
        gamma: float
    ) -> bool:
        valid = 0. <= gamma <= 1.
        time_len = len(transitions_rewards)
        i = 0
        while valid and i < time_len:
            this_d = transitions_rewards[i]
            check_actions = all(len(v) > 0 for _, v in this_d.items())
            next_dict = [{k: v for k, (v, _) in d1.items()}
                         for _, d in this_d.items() for _, d1 in d.items()]
            check_pos = all(all(x >= 0 for x in d1.values()) for d1 in next_dict)
            check_sum = all(is_approx_eq(sum(d1.values()), 1.0) for d1 in next_dict)
            states = set((transitions_rewards[i+1]
                         if i < time_len - 1 else terminal_opt_val).keys())
            subset = all(set(d1.keys()).issubset(states) for d1 in next_dict)
            valid = valid and check_actions and check_pos and check_sum and subset
            i = i + 1
        return valid

    def get_vf_and_policy(self) -> Sequence[Mapping[S, Tuple[float, A]]]:
        vf_pol = {s: (v, None) for s, v in self.terminal_opt_val.items()}
        ret = []
        for tr in self.transitions_rewards[::-1]:
            vf_pol = {s: max(
                [(
                    sum(p * (r + self.gamma * vf_pol[s1][0])
                        for s1, (p, r) in d1.items()),
                    a
                ) for a, d1 in d.items()],
                key=itemgetter(0)
            ) for s, d in tr.items()}
            ret.append(vf_pol)
        return ret[::-1]


if __name__ == '__main__':
    T: int = 10  # time steps
    M: int = 200  # initial inventory
    # the following are (price, poisson mean) pairs, i.e., elasticity
    el: Sequence[Tuple[float, float]] = [
        (10.0, 10.0), (9.0, 16.0), (8.0, 20.0),
        (7.0, 23.0), (6.0, 25.0), (5.0, 26.0)
    ]
    rvs: Sequence = [poisson(l) for _, l in el]

    tr_rew_dict = {
        s: {
            p: {
                s - d: (
                    rvs[i].pmf(d) if d < s else 1. - rvs[i].cdf(s - 1),
                    d * p
                ) for d in range(s + 1)
            } for i, (p, l) in enumerate(el)
        } for s in range(M + 1)
    }

    bdp = BackwardDP(
        transitions_rewards=[tr_rew_dict] * T,
        terminal_opt_val={s: 0. for s in range(M + 1)},
        gamma=1.
    )
    print(bdp.vf_and_policy[0])

