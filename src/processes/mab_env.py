from typing import Tuple, Callable, Sequence, NamedTuple
import numpy as np


class MabEnv(NamedTuple):

    arms_sampling_funcs: Sequence[Callable[[], float]]

    @staticmethod
    def get_gaussian_mab_env(means_vars: Sequence[Tuple[float, float]]) -> 'MabEnv':
        return MabEnv([lambda m=m, s=s: np.random.normal(m, s, 1)[0] for m, s in means_vars])

    @staticmethod
    def get_bernoulli_mab_env(probs: Sequence[float]) -> 'MabEnv':
        return MabEnv([lambda p=p: float(np.random.binomial(1, p, 1)[0]) for p in probs])


if __name__ == '__main__':
    mean_vars_data = [(5., 2.), (10., 3.), (0., 4.)]
    me = MabEnv.get_gaussian_mab_env(mean_vars_data)
    asf = me.arms_sampling_funcs
    res = [[asf[i]() for _ in range(10000)] for i in range(len(asf))]
    for i in range(len(mean_vars_data)):
        nums = res[i]
        print("Mean = %.3f" % np.mean(nums))
        print("Stdev = %.3f" % np.std(nums))
