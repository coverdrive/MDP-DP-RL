from typing import Sequence, NamedTuple
import numpy as np


class WageMax(NamedTuple):

    probs: Sequence[float]
    wages: Sequence[float]
    gamma: float
    alpha: float
    risk_aversion: float

    def validate_inputs(self) -> bool:
        b1 = abs(sum(self.probs) - 1) <= 1e-8
        b2 = len(self.probs) + 1 == len(self.wages)
        b3 = all(self.wages[0] < w for w in self.wages[1:])
        b4 = 0. <= self.gamma < 1.
        b5 = 0. <= self.alpha <= 1.
        b6 = self.risk_aversion > 0.
        return all([b1, b2, b3, b4, b5, b6])

    # noinspection PyShadowingNames
    def get_wages_utility(self) -> Sequence[float]:
        a = self.risk_aversion
        f = (lambda x, a=a: (pow(x, 1 - a) - 1) / (1 - a)) \
            if a != 1 else (lambda x: np.log(x))
        return [f(w) for w in self.wages]

    def get_opt_vf(self) -> Sequence[float]:
        jobs = len(self.probs)
        utils = self.get_wages_utility()
        vf = [0.] * (jobs + 1)
        tol = 1e-6
        epsilon = tol * 1e6
        while epsilon >= tol:
            old_vf = [v for v in vf]
            vf[0] = sum(self.probs[i] * max(
                vf[i + 1],
                utils[0] + self.gamma * vf[0]
            ) for i in range(jobs))
            for i in range(1, jobs + 1):
                vf[i] = utils[i] + self.gamma *\
                            (self.alpha * vf[0] + (1 - self.alpha) * vf[i])
            epsilon = max(abs(old_vf[i] - v) for i, v in enumerate(vf))
        return vf

    def get_opt_policy(self) -> Sequence[str]:
        jobs = len(self.probs)
        utils = self.get_wages_utility()
        vf = self.get_opt_vf()
        return ["Accept" if vf[i] > utils[0] + self.gamma * vf[0]
                else "Decline" for i in range(1, jobs + 1)]


if __name__ == '__main__':
    this_probs: Sequence[float] = [0.5, 0.3, 0.2]
    this_wages: Sequence[float] = [1.0, 1.8, 2.8, 5.2]
    this_gamma: float = 0.9
    this_alpha: float = 0.2
    this_risk_aversion: float = 1.0
    # all_jobs = 10
    # this_probs: Sequence[float] = [1. / all_jobs] * all_jobs
    # this_wages: Sequence[float] = [i + 1 for i in range(all_jobs + 1)]
    # this_gamma: float = 0.5
    # this_alpha: float = 0.1
    # this_risk_aversion: float = 0.5
    wm = WageMax(
        probs=this_probs,
        wages=this_wages,
        gamma=this_gamma,
        alpha=this_alpha,
        risk_aversion=this_risk_aversion
    )
    if not wm.validate_inputs():
        raise ValueError
    opt_vf = wm.get_opt_vf()
    opt_policy = wm.get_opt_policy()
    print(opt_vf)
    print(opt_policy)