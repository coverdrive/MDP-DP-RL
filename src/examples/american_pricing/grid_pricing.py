from typing import Callable, Tuple
from scipy.integrate import trapz
from scipy.interpolate import splrep, BSpline
from scipy.stats import norm
import numpy as np


class GridPricing:
    """
    In the risk-neutral measure, the underlying price x_t
    follows the Ito process: dx_t = r_t x_t dt + dispersion(t, x_t) dz_t
    spot_price is x_0
    payoff is a function from (t, x_t) to payoff (eg: (x_t - K)^+)
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

    @staticmethod
    def get_future_price_mean_var(
        x,
        t,
        delta_t,
        rate_int_func,  # ir
        disp_func  # dispersion
    ) -> Tuple[float, float]:
        """
        :param x: represents underlying price at time t, x_t
        :param t: represents current time t
        :param delta_t: represents interval of time beyond t at which
        we want the future price, i.e., at time t + delta_t
        :param rate_int_func: this is ir_t func
        :param disp_func: this is dispersion func
        :return: mean and variance of x_{t+delta_t}

        Treat dx_t = r_t x_t dt + dispersion(t, x_t) dz_t
        as Ornstein-Uhlenbeck (OU) even though dispersion is dependent
        on x_t. Solve it like it is OU. Then, (with t1 = t + delta_t),
        x_{t1} = e^{ir(t1) - ir(t)} (x_t + int_t^{t1} e^{ir(u) - ir(t)}^{-1}
         . dispersion(u, x_u) dz
         Mean(x_{t1}) = x_t e^{ir(t1) - ir(t)}
         Variance(x_{t1}) = e^{ir(t1) - ir(t)}^2 int_t^{t1}
         e^{ir(u) - ir(t)}^{-2} . dispersion^2(u, x_u) du
         and in variance formula, replace x_u with x_t e^{ir(u) - ir(t)}
        """
        temp = np.exp(rate_int_func(t + delta_t) - rate_int_func(t))
        mean = x * temp

        # noinspection PyShadowingNames
        def int_func(u: float, x=x, t=t) -> float:
            temp1 = np.exp(rate_int_func(t + u) - rate_int_func(t))
            return (disp_func(t + u, x * temp1) / temp1) ** 2

        num_samples = 11

        var = temp ** 2 * trapz(
            np.vectorize(int_func)(np.linspace(0., delta_t, num_samples)),
            dx=delta_t / (num_samples - 1)
        )
        return mean, var

    def get_price(
        self,
        num_dt: int,
        dx: float,
        num_dx: int
    ) -> np.ndarray:
        dt = self.expiry / num_dt
        x_pts = 2 * num_dx + 1
        x_limit = dx * num_dx
        res = np.empty([num_dt + 1, x_pts])
        prices = np.linspace(-x_limit, x_limit, x_pts) + self.spot_price
        res[-1, :] = [self.payoff(self.expiry, p) for p in prices]
        for i in range(num_dt - 1, -1, -1):
            t = i * dt
            knots, coeffs, order = splrep(prices, res[i + 1, :], k=3)
            spline_func = BSpline(knots, coeffs, order)
            disc = np.exp(self.ir(t) - self.ir(t+dt))
            for j in range(x_pts):
                m, v = GridPricing.get_future_price_mean_var(
                    prices[j],
                    t,
                    dt,
                    self.ir,
                    self.dispersion
                )
                stdev = np.sqrt(v)
                norm_dist = norm(loc=m, scale=stdev)
                sample_points = 201
                integr_func = lambda x: spline_func(x) * norm_dist.pdf(x)
                low, high = (m - 4 * stdev, m + 4 * stdev)
                disc_exp_payoff = disc * trapz(
                    np.vectorize(integr_func)(np.linspace(low, high, sample_points)),
                    dx=(high - low) / (sample_points - 1)
                ) / (norm_dist.cdf(high) - norm_dist.cdf(low))
                res[i, j] = max(self.payoff(t, prices[j]), disc_exp_payoff)
        return res


if __name__ == '__main__':
    spot_price_val = 80.0
    strike_val = 78.2
    payoff_func = lambda _, x: max(strike_val - x, 0.)
    expiry_val = 10.0
    rr = 0.03
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
    drift_func = lambda t, x, rr=rr: rr * x
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
    dt = 0.1
    num_dt_val = int(expiry_val / dt)
    x_lim = 4. * sigma_val * spot_price_val * np.sqrt(expiry_val)
    num_dx_val = 200
    dx_val = x_lim / num_dx_val
    print(gp.get_price(
        num_dt=num_dt_val,
        dx=dx_val,
        num_dx=num_dx_val
    )[0][num_dx_val])

