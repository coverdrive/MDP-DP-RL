from typing import Callable
from scipy.integrate import trapz
from scipy.interpolate import splrep, BSpline
from scipy.stats import norm
import numpy as np
from src.examples.american_pricing.num_utils import get_future_price_mean_var


class GridPricing:
    """
    In the risk-neutral measure, the underlying price x_t
    follows the Ito process: dx_t = r_t x_t dt + dispersion(t, x_t) dz_t
    spot_price is x_0
    payoff is a function from (t, x_t) to payoff (eg: x_t - K)
    expiry is the time to expiry of american option (in years)
    dispersion(t, x_t) is a function from (t, x_t) to dispersion
    r_t is a function from (time t) to risk-free rate
    We define ir_t = \int_0^t r_u du, so discount D_t = e^{- ir_t}
    """

    def __init__(
        self,
        spot_price: float,
        payoff: Callable[[float, float], float],
        expiry: float,
        dispersion: Callable[[float, float], float],
        r: Callable[[float], float],
        ir: Callable[[float], float]
    ) -> None:
        self.spot_price: float = spot_price
        self.payoff: Callable[[float, float], float] = payoff
        self.expiry: float = expiry
        self.dispersion: Callable[[float, float], float] = dispersion
        self.r: Callable[[float], float] = r
        self.ir: Callable[[float], float] = ir

    def get_price(
        self,
        num_dt: int,
        dx: float,
        num_dx: int
    ) -> float:
        dt = self.expiry / num_dt
        x_pts = 2 * num_dx + 1
        x_limit = dx * num_dx
        res = np.empty([num_dt + 1, x_pts])
        prices = np.linspace(-x_limit, x_limit, x_pts) + self.spot_price
        res[-1, :] = [max(self.payoff(self.expiry, p), 0.) for p in prices]
        for i in range(num_dt - 1, -1, -1):
            t = i * dt
            knots, coeffs, order = splrep(prices, res[i + 1, :], k=3)
            spline_func = BSpline(knots, coeffs, order)
            disc = np.exp(self.ir(t) - self.ir(t + dt))
            for j in range(x_pts):
                m, v = get_future_price_mean_var(
                    prices[j],
                    t,
                    dt,
                    self.ir,
                    self.dispersion
                )
                stdev = np.sqrt(v)
                norm_dist = norm(loc=m, scale=stdev)
                sample_points = 201
                integr_func = lambda x: max(spline_func(x), 0.) * norm_dist.pdf(x)
                low, high = (m - 4 * stdev, m + 4 * stdev)
                disc_exp_payoff = disc * trapz(
                    np.vectorize(integr_func)(np.linspace(low, high, sample_points)),
                    dx=(high - low) / (sample_points - 1)
                ) / (norm_dist.cdf(high) - norm_dist.cdf(low))
                res[i, j] = max(self.payoff(t, prices[j]), disc_exp_payoff)
        return res[0, num_dx]


if __name__ == '__main__':
    spot_price_val = 80.0
    strike_val = 75.0
    payoff_func = lambda _, x: x - strike_val
    expiry_val = 4.0
    rr = 0.03
    sigma_val = 0.25

    from examples.american_pricing.bs_pricing import EuropeanBSPricing
    ebsp = EuropeanBSPricing(
        is_call=True,
        spot_price=spot_price_val,
        strike=strike_val,
        expiry=expiry_val,
        r=rr,
        sigma=sigma_val
    )
    print(ebsp.option_price)
    # noinspection PyShadowingNames
    dispersion_func = lambda t, x, sigma_val=sigma_val: sigma_val * x
    # noinspection PyShadowingNames
    r_func = lambda t, rr=rr: rr
    # noinspection PyShadowingNames
    ir_func = lambda t, rr=rr: rr * t

    gp = GridPricing(
        spot_price=spot_price_val,
        payoff=payoff_func,
        expiry=expiry_val,
        dispersion=dispersion_func,
        r=r_func,
        ir=ir_func
    )
    dt_val = 0.1
    num_dt_val = int(expiry_val / dt_val)
    x_lim = 4. * sigma_val * spot_price_val * np.sqrt(expiry_val)
    num_dx_val = 200
    dx_val = x_lim / num_dx_val
    print(gp.get_price(
        num_dt=num_dt_val,
        dx=dx_val,
        num_dx=num_dx_val
    ))

