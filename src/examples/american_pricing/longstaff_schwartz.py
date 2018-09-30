from typing import Callable, Sequence
import numpy as np
from src.examples.american_pricing.num_utils import get_future_price_mean_var


class LongstaffSchwartz:
    """
    In the risk-neutral measure, the underlying price x_t
    follows the Ito process: dx_t = r_t x_t dt + dispersion(t, x_t) dz_t
    spot_price is x_0
    payoff is a function from (t, (x_0, ..., x_t) to payoff (
    eg: \sum_{i=0}^t x_i / (t+1) - K)
    expiry is the time to expiry of american option (in years)
    dispersion(t, x_t) is a function from (t, x_t) to dispersion
    r_t is a function from (time t) to risk-free rate
    We define ir_t = \int_0^t r_u du, so discount D_t = e^{- ir_t}
    """

    def __init__(
        self,
        spot_price: float,
        payoff: Callable[[float, np.ndarray], float],
        expiry: float,
        dispersion: Callable[[float, float], float],
        r: Callable[[float], float],
        ir: Callable[[float], float]
    ) -> None:
        self.spot_price: float = spot_price
        self.payoff: Callable[[float, np.ndarray], float] = payoff
        self.expiry: float = expiry
        self.dispersion: Callable[[float, float], float] = dispersion
        self.r: Callable[[float], float] = r
        self.ir: Callable[[float], float] = ir

    def get_price(
        self,
        num_dt: int,
        num_paths: int,
        feature_funcs: Sequence[Callable[[float, np.ndarray], float]]
    ) -> float:
        dt = self.expiry / num_dt
        paths = np.empty([num_paths, num_dt + 1])
        paths[:, 0] = self.spot_price
        for i in range(num_paths):
            price = self.spot_price
            for t in range(num_dt):
                m, v = get_future_price_mean_var(
                    price,
                    t,
                    dt,
                    self.ir,
                    self.dispersion
                )
                price = np.random.normal(m, np.sqrt(v))
                paths[i, t + 1] = price
        cashflow = np.array([max(self.payoff(self.expiry, paths[i, :]), 0.)
                             for i in range(num_paths)])
        for t in range(num_dt - 1, 0, -1):
            """
            For each time slice t
            Step 1: collect X as features of (t, [S_0,.., S_t]) for those paths
            for which payoff(t, [S_0, ...., S_t]) > 0, and corresponding Y as
            the time-t discounted future actual cash flow on those paths.
            Step 2: Do the (X,Y) regression. Denote Y^ as regression-prediction.
            Compare Y^ versus payoff(t, [S_0, ..., S_t]). If payoff is higher,
            set cashflow at time t on that path to be the payoff, else set 
            cashflow at time t on that path to be the time-t discounted future
            actual cash flow on that path.
            """
            disc = np.exp(self.ir(t) - self.ir(t + dt))
            cashflow = cashflow * disc
            payoff = np.array([self.payoff(t, paths[i, :(t + 1)]) for
                               i in range(num_paths)])
            indices = [i for i in range(num_paths) if payoff[i] > 0]
            if len(indices) > 0:
                x_vals = np.array([[f(t, paths[i, :(t + 1)]) for f in
                                    feature_funcs] for i in indices])
                y_vals = np.array([cashflow[i] for i in indices])
                estimate = x_vals.dot(
                    np.linalg.lstsq(x_vals, y_vals, rcond=None)[0]
                )
                for i, ind in enumerate(indices):
                    if payoff[ind] > estimate[i]:
                        cashflow[ind] = payoff[ind]

        return max(
            self.payoff(0, np.array([self.spot_price])),
            np.average(cashflow * np.exp(-self.ir(dt)))
        )


if __name__ == '__main__':
    spot_price_val = 80.0
    strike_val = 74.8
    payoff_func = lambda _, x: strike_val - x[-1]
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

    gp = LongstaffSchwartz(
        spot_price=spot_price_val,
        payoff=payoff_func,
        expiry=expiry_val,
        dispersion=dispersion_func,
        r=r_func,
        ir=ir_func
    )
    dt_val = 0.1
    num_dt_val = int(expiry_val / dt_val)
    num_paths_val = 10000
    from numpy.polynomial.laguerre import lagval
    num_laguerre = 10
    ident = np.eye(num_laguerre)
    print(gp.get_price(
        num_dt=num_dt_val,
        num_paths=num_paths_val,
        feature_funcs=[lambda t, x: np.exp(-x[-1] / 2) * lagval(x[-1], ident[i])
                       for i in range(num_laguerre)]
    ))

