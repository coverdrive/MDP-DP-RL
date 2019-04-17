from typing import Mapping, Set, Sequence, Tuple, Generic, Callable
from utils.gen_utils import is_approx_eq
from utils.generic_typevars import S, A
from operator import itemgetter
from algorithms.func_approx_spec import FuncApproxSpec
from func_approx.func_approx_base import FuncApproxBase


class BackwardADP(Generic[S, A]):

    def __init__(
        self,
        state_actions_funcs: Sequence[Callable[[S], Set[A]]],
        sample_states_gen_funcs: Sequence[Callable[[int], Sequence[S]]],
        transitions_rewards_funcs: Sequence[Callable[[S, A], Mapping[S, Tuple[float, float]]]],
        terminal_opt_val_func: Callable[[S], float],
        gamma: float,
        fa_specs: Sequence[FuncApproxSpec]
    ) -> None:
        if (len(state_actions_funcs) == len(sample_states_gen_funcs)\
            == len(transitions_rewards_funcs) == len(fa_specs))\
            and 0. <= gamma <= 1.:
            self.state_actions_funcs = state_actions_funcs
            self.sample_states_gen_funcs = sample_states_gen_funcs
            self.transitions_rewards_funcs = transitions_rewards_funcs
            self.terminal_opt_val_func = terminal_opt_val_func
            self.gamma = gamma
            self.fas: Sequence[FuncApproxBase] = [x.get_vf_func_approx_obj() for x in fa_specs]
            self.vf_and_policy_func = self.get_vf_and_policy_func()
        else:
            raise ValueError


    def get_vf_and_policy_func(self) -> Sequence[Mapping[S, Tuple[float, A]]]:
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
    from scipy.stats import poisson
    T: int = 10  # time steps
    M: int = 200  # initial inventory
    # the following are (price, poisson mean) pairs, i.e., elasticity
    el: Sequence[Tuple[float, float]] = [
        (10.0, 10.0), (9.0, 16.0), (8.0, 20.0),
        (7.0, 23.0), (6.0, 25.0), (5.0, 26.0)
    ]
    rvs = [(p, poisson(l)) for p, l in el]

    tr_rew_dict = {
        s: {
            p: {
                s - d: (
                    rv.pmf(d) if d < s else 1. - rv.cdf(s - 1),
                    d * p
                ) for d in range(s + 1)
            } for p, rv in rvs
        } for s in range(M + 1)
    }

    bdp = BackwardADP(
        transitions_rewards=[tr_rew_dict] * T,
        terminal_opt_val={s: 0. for s in range(M + 1)},
        gamma=1.
    )
    print(bdp.vf_and_policy[0])
