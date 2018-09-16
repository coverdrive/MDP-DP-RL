import numpy as np
from typing import Tuple
from scipy.special import digamma


class BetaDistribution:

    SMALL_POS = 1e-8
    """
    Beta Distribution is normally defined by parameters alpha and beta
    with alpha, beta > 0. Here we define the beta distribution in terms
    of parameters mu (for mean of beta distribution and nu (= alpha + beta).

    So,  mu = alpha / (alpha + beta) = alpha / nu
    alpha = mu * nu, beta = (1-mu) * nu

    p(x) = Gamma(alpa + beta) / (Gamma(alpha) * Gamma(beta)) *
    x^{alpha-1) * (1-x)^{beta-1)

    Score_mu(x) = d(log(p(x)))/d(mu) = Score_alpha * d(alpha)/d(mu)
    + Score_beta * d(beta)/d(mu)
    = (digamma(beta) - digamma(alpha) + log(x) - log(1-x)) * nu

    Score_nu(x) = d(log(p(x)))/d(nu) = Score_alpha * d(alpha)/d(nu)
    + Score_beta * d(beta)/d(nu)
    = (digamma(beta) - digamma(alpha) + log(x) - log(1-x)) * mu +
    digamma(nu) - digamma(beta) + log(1-x)
    """

    def __init__(self, mu, nu) -> None:
        if 0 < mu < 1 and nu > 0:
            self.mu = mu
            self.nu = nu
            self.alpha = mu * nu
            self.beta = (1. - mu) * nu
        else:
            raise ValueError("mu = %.3f, nu = %.3f" % (mu, nu))

    def get_samples(self, n: int) -> np.ndarray:
        sp = BetaDistribution.SMALL_POS
        return np.vectorize(lambda x: min(1. - sp, max(sp, x)))(
            np.random.beta(a=self.alpha, b=self.beta, size=n)
        )

    def get_mu_nu_scores(self, x: float) -> Tuple[float, float]:
        diga = digamma(self.alpha)
        digb = digamma(self.beta)
        dign = digamma(self.nu)
        lx = np.log(x)
        l1x = np.log(1. - x)
        temp = digb - diga + lx - l1x
        r1 = temp * self.nu
        r2 = temp * self.mu + dign - digb + l1x
        return r1, r2


