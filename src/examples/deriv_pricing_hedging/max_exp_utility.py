from typing import NamedTuple, Callable, Mapping
from scipy.optimize import minimize_scalar, root_scalar
from scipy.integrate import quad
import numpy as np


class MaxExpUtility(NamedTuple):
    """
    The goal is to compute the price and hedges for a derivative
    for a single underlying in a single time-step setting. We
    assume that the underlying takes on a continuum of values
    at t=1 (hedge of underlying and risk-free security established
    at t = 0). This corresponds to an incomplete market scenario
    and so, there is no unique price. We determine pricing
    and hedging using the Maximum Expected Utility method and
    assume that the Utility function is CARA (-e^{-ax}/a) where
    a is the risk-aversion parameter. We assume the underlying
    follows a normal distribution at t-1.
    Formal Details in Appendix 4 of
    https://github.com/coverdrive/technical-documents/blob/master/finance/ArbitrageCompleteness.pdf
    """
    underlying_spot: float  # underlying value at t=0
    risk_free_rate: float  # risk-free security grows as e^{risk-free-rate}
    underlying_mean: float  # mean of underlying at t=1
    underlying_stdev: float  # standard deviation of underlying at t=1
    payoff_func: Callable[[float], float]  # derivative payoff at t=1

    def validate_spec(self) -> bool:
        b1 = self.risk_free_rate >= 0.
        b2 = self.underlying_stdev > 0.
        x = self.underlying_spot * np.exp(self.risk_free_rate)
        b3 = self.underlying_mean > x >\
            self.underlying_mean - self.underlying_stdev
        return all([b1, b2, b3])

    def complete_mkt_price_and_hedges(self) -> Mapping[str, float]:
        """
        This computes the price and hedges assuming a complete
        market, which means the underlying takes on two values
        at t=1. 1) mean + stdev 2) mean - stdev, with equal
        probabilities. This situation can be perfectly hedged
        with underlying and risk-free security. The following
        code provides the solution for the 2 equations and 2
        variables system
        alpha is the hedge in the underlying units and beta
        is the hedge in the risk-free security units
        """
        x = self.underlying_mean + self.underlying_stdev
        z = self.underlying_mean - self.underlying_stdev
        v1 = self.payoff_func(x)
        v2 = self.payoff_func(z)
        alpha = (v1 - v2) / (z - x)
        beta = - np.exp(-self.risk_free_rate) * (v1 + alpha * x)
        price = - (beta + alpha * self.underlying_spot)
        return {"price": price, "alpha": alpha, "beta": beta}

    def max_exp_util_for_zero(
        self,
        c: float,
        risk_aversion_param: float
    ) -> Mapping[str, float]:
        ra = risk_aversion_param
        er = np.exp(self.risk_free_rate)
        mu = self.underlying_mean
        sigma = self.underlying_stdev
        s0 = self.underlying_spot
        alpha = (mu - s0 * er) / (ra * sigma * sigma)
        beta = - (c + alpha * self.underlying_spot)
        max_val = - np.exp(-ra * (-er * c + alpha * (mu - s0 * er))
                           + (ra * alpha * sigma) ** 2 / 2) / ra
        return {"alpha": alpha, "beta": beta, "max_val": max_val}

    def max_exp_util(
        self,
        c: float,
        pf: Callable[[float], float],
        risk_aversion_param: float
    ) -> Mapping[str, float]:
        sigma2 = self.underlying_stdev * self.underlying_stdev
        mu = self.underlying_mean
        s0 = self.underlying_spot
        er = np.exp(self.risk_free_rate)
        factor = 1. / np.sqrt(2. * np.pi * sigma2)

        integral_lb = self.underlying_mean - self.underlying_stdev * 6
        integral_ub = self.underlying_mean + self.underlying_stdev * 6

        def eval_expectation(alpha: float, c=c) -> float:

            def integrand(rand: float, alpha=alpha, c=c) -> float:
                payoff = pf(rand) - er * c\
                         + alpha * (rand - er * s0)
                exponent = -(0.5 * (rand - mu) * (rand - mu) / sigma2
                             + risk_aversion_param * payoff)
                return - factor / risk_aversion_param * np.exp(exponent)

            return -quad(integrand, integral_lb, integral_ub)[0]

        res = minimize_scalar(eval_expectation)
        alpha_star = res["x"]
        max_val = - res["fun"]
        beta_star = - (c + alpha_star * s0)
        return {"alpha": alpha_star, "beta": beta_star, "max_val": max_val}

    def max_exp_util_price_and_hedge(
        self,
        risk_aversion_param: float
    ) -> Mapping[str, float]:
        meu_for_zero = self.max_exp_util_for_zero(0., risk_aversion_param)["max_val"]

        def prep_func(pr: float) -> float:
            return self.max_exp_util(
                pr,
                self.payoff_func,
                risk_aversion_param
            )["max_val"] - meu_for_zero

        lb = self.underlying_mean - self.underlying_stdev * 6
        ub = self.underlying_mean + self.underlying_stdev * 6
        payoff_vals = [self.payoff_func(x) for x in np.linspace(lb, ub, 1001)]
        lb_payoff = min(payoff_vals)
        ub_payoff = max(payoff_vals)

        opt_price = root_scalar(
            prep_func,
            bracket=[lb_payoff, ub_payoff],
            method="brentq"
        ).root

        hedges = self.max_exp_util(
            opt_price,
            self.payoff_func,
            risk_aversion_param
        )
        alpha = hedges["alpha"]
        beta = hedges["beta"]
        return {"price": opt_price, "alpha": alpha, "beta": beta}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    meu = MaxExpUtility(
        underlying_spot=100.,
        risk_free_rate=0.0,
        underlying_mean=105.,
        underlying_stdev=10.,
        payoff_func=lambda x: np.abs(x - 100.)
    )

    if meu.validate_spec():
        plt.xlabel("Underlying Price")
        plt.ylabel("Derivative Payoff and Hedges")
        plt.title("Hedging in Incomplete Market")
        lb = meu.underlying_mean - meu.underlying_stdev * 1.3
        ub = meu.underlying_mean + meu.underlying_stdev * 1.3
        x_plot_pts = np.linspace(lb, ub, 1001)
        payoff_plot_pts = np.array([meu.payoff_func(x) for x in x_plot_pts])
        plt.plot(x_plot_pts, payoff_plot_pts, "r", linewidth=3, label="Derivative Payoff")
        cm_ph = meu.complete_mkt_price_and_hedges()
        cm_plot_pts = - (cm_ph["beta"] + cm_ph["alpha"] * x_plot_pts)
        plt.plot(x_plot_pts, cm_plot_pts, "b", linestyle="dashed", label="Complete Market Hedge")
        print("Complete Market Price = %.3f" % cm_ph["price"])
        print("Complete Market Alpha = %.3f" % cm_ph["alpha"])
        print("Complete Market Beta = %.3f" % cm_ph["beta"])
        for risk_aversion_param, color in [(0.2, "g"), (1.0, "y"), (5.0, "m")]:
            print("------ Risk Aversion Param = %.2f ----------" % risk_aversion_param)
            meu_for_zero = meu.max_exp_util_for_zero(0., risk_aversion_param)
            print("MEU for Zero Alpha = %.3f" % meu_for_zero["alpha"])
            print("MEU for Zero Beta = %.3f" % meu_for_zero["beta"])
            print("MEU for Zero Max Val = %.3f" % meu_for_zero["max_val"])
            res2 = meu.max_exp_util_price_and_hedge(risk_aversion_param)
            print(res2)
            im_plot_pts = - (res2["beta"] + res2["alpha"] * x_plot_pts)
            plt.plot(x_plot_pts, im_plot_pts, color, label="Hedge for Risk-Aversion = %.1f" % risk_aversion_param)

        plt.xlim(lb, ub)
        plt.ylim(min(payoff_plot_pts), max(payoff_plot_pts))
        plt.grid(True)
        plt.legend()
        plt.show()


