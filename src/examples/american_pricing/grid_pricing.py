from typing import Callable
from scipy.integrate import trapz
from scipy.interpolate import splrep, BSpline
from scipy.stats import norm
import numpy as np
from src.examples.american_pricing.num_utils import get_future_price_mean_var

MIN_STOCK_PRICE = 1e-4


class GridPricing:
    """
    In the risk-neutral measure, the underlying price x_t
    follows the Ito process: dx_t = r(t) x_t dt + dispersion(t, x_t) dz_t
    spot_price is x_0
    In this module, we only allow two types of dispersion functions,
    Type 1 (a.k.a. "lognormal") : dx_t = r(t) x_t dt + sigma(t) x_t dz_t
    Type 2 (a.k.a. "normal"): dx_t = r(t) x_t dt + sigma(t) dz_t
    payoff is a function from (t, x_t) to payoff (eg: x_t - K)
    expiry is the time to expiry of american option (in years)
    lognormal is a bool that defines whether our dispersion function
    amounts to Type 1(lognormal) or Type 2(normal)
    r(t) is a function from (time t) to risk-free rate
    sigma(t) is a function from (time t) to (sigma at time t)
    We don't provide r(t) and sigma(t) as arguments
    Instead we provide their appropriate integrals as arguments
    Specifically, we provide ir(t) and isig(t) as arguments (as follows):
    ir(t) = \int_0^t r(u) du, so discount D_t = e^{- ir(t)}
    isig(t) = \int 0^t sigma^2(u) du if lognormal == True
    else \int_0^t sigma^2(u) e^{-\int_0^u 2 r(s) ds} du
    """

    def __init__(
        self,
        spot_price: float,
        payoff: Callable[[float, float], float],
        expiry: float,
        lognormal: bool,
        ir: Callable[[float], float],
        isig: Callable[[float], float]
    ) -> None:
        self.spot_price: float = spot_price
        self.payoff: Callable[[float, float], float] = payoff
        self.expiry: float = expiry
        self.lognormal: bool = lognormal
        self.ir: Callable[[float], float] = ir
        self.isig: Callable[[float, float], float] = isig

    def get_price(
        self,
        num_dt: int,
        num_dx: int,
        center: float,
        width: float
    ) -> float:
        """
        :param num_dt: represents number of discrete time steps
        :param num_dx: represents number of discrete state-space steps
        (on each side of the center)
        :param center: represents the center of the state space grid. For
        the case of lognormal == True, it should be Mean[log(x_{expiry}].
        For the case of lognormal == False, it should be Mean[x_{expiry}].
        :param width: represents the width of the state space grid. For the
        case of lognormal == True, it should be a multiple of
        Stdev[log(x_{expiry})]. For the case of lognormal == True, it
        should be a multiple of Stdev[log(x_{expiry})].
        :return: the price of the American option (this is the discounted
        expected payoff at time 0 at current stock price.
        """
        dt = self.expiry / num_dt
        x_pts = 2 * num_dx + 1
        lsp = np.linspace(center - width, center + width, x_pts)
        prices = np.exp(lsp) if self.lognormal else lsp
        res = np.empty([num_dt, x_pts])
        res[-1, :] = [max(self.payoff(self.expiry, p), 0.) for p in prices]
        sample_points = 201
        for i in range(num_dt - 2, -1, -1):
            t = (i + 1) * dt
            knots, coeffs, order = splrep(prices, res[i + 1, :], k=3)
            spline_func = BSpline(knots, coeffs, order)
            disc = np.exp(self.ir(t) - self.ir(t + dt))
            for j in range(x_pts):
                m, v = get_future_price_mean_var(
                    prices[j],
                    t,
                    dt,
                    self.lognormal,
                    self.ir,
                    self.isig
                )
                stdev = np.sqrt(v)
                norm_dist = norm(loc=m, scale=stdev)

                # noinspection PyShadowingNames
                def integr_func(
                    x: float,
                    spline_func=spline_func,
                    norm_dist=norm_dist
                ) -> float:
                    val = np.exp(x) if self.lognormal else x
                    return max(spline_func(val), 0.) * norm_dist.pdf(x)

                low, high = (m - 4 * stdev, m + 4 * stdev)
                disc_exp_payoff = disc * trapz(
                    np.vectorize(integr_func)(np.linspace(low, high, sample_points)),
                    dx=(high - low) / (sample_points - 1)
                ) / (norm_dist.cdf(high) - norm_dist.cdf(low))
                res[i, j] = max(self.payoff(t, prices[j]), disc_exp_payoff)

        knots, coeffs, order = splrep(prices, res[0, :], k=3)
        spline_func = BSpline(knots, coeffs, order)
        disc = np.exp(-self.ir(dt))
        m, v = get_future_price_mean_var(
            self.spot_price,
            0.,
            dt,
            self.lognormal,
            self.ir,
            self.isig
        )
        stdev = np.sqrt(v)
        norm_dist = norm(loc=m, scale=stdev)

        # noinspection PyShadowingNames
        def integr_func0(
            x: float,
            spline_func=spline_func,
            norm_dist=norm_dist
        ) -> float:
            val = np.exp(x) if self.lognormal else x
            return max(spline_func(val), 0.) * norm_dist.pdf(x)

        low, high = (m - 4 * stdev, m + 4 * stdev)
        disc_exp_payoff = disc * trapz(
            np.vectorize(integr_func0)(np.linspace(low, high, sample_points)),
            dx=(high - low) / (sample_points - 1)
        ) / (norm_dist.cdf(high) - norm_dist.cdf(low))
        return max(self.payoff(0., self.spot_price), disc_exp_payoff)


if __name__ == '__main__':
    spot_price_val = 80.0
    strike_val = 78.0
    payoff_func = lambda _, x: strike_val - x
    expiry_val = 2.0
    rr = 0.05
    lognormal_val = True
    sigma_val = 0.25

    from examples.american_pricing.bs_pricing import EuropeanBSPricing
    ebsp = EuropeanBSPricing(
        is_call=False,
        spot_price=spot_price_val,
        strike=strike_val,
        expiry=expiry_val,
        r=rr,
        sigma=sigma_val
    )
    print(ebsp.option_price)
    # noinspection PyShadowingNames
    ir_func = lambda t, rr=rr: rr * t
    # noinspection PyShadowingNames
    isig_func = lambda t, sigma_val=sigma_val: sigma_val * sigma_val * t

    gp = GridPricing(
        spot_price=spot_price_val,
        payoff=payoff_func,
        expiry=expiry_val,
        lognormal=lognormal_val,
        ir=ir_func,
        isig=isig_func,
    )
    num_dt_val = 10
    num_dx_val = 100
    expiry_mean, expiry_var = get_future_price_mean_var(
        spot_price_val,
        0.,
        expiry_val,
        lognormal_val,
        ir_func,
        isig_func
    )
    print(gp.get_price(
        num_dt=num_dt_val,
        num_dx=num_dx_val,
        center=expiry_mean,
        width=np.sqrt(expiry_var) * 4.
    ))

