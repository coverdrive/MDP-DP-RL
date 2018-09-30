from typing import Tuple
import numpy as np
from scipy.integrate import trapz
from src.examples.american_pricing.bs_pricing import EuropeanBSPricing
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


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


def plot_fitted_call_prices(
    is_call: bool,
    strike: float,
    expiry: float,
    r: float,
    sigma: float
) -> None:
    spot_prices = np.linspace(strike * 0.5, strike * 1.5, 1001)
    option_prices = [EuropeanBSPricing(
        is_call,
        s,
        strike,
        expiry,
        r,
        sigma
    ).get_option_price() for s in spot_prices]

    def fit_func(
        x: np.ndarray,
        a: float,
        b: float,
        c: float
    ) -> np.ndarray:
        return a * np.exp(b * x + c)

    def jac_func(
        x: np.ndarray,
        a: float,
        b: float,
        c: float
    ) -> np.ndarray:
        t = np.exp(b * x + c)
        da = t
        db = a * t * x
        dc = a * t
        return np.transpose([da, db, dc])

    fp = curve_fit(
        f=fit_func,
        xdata=spot_prices,
        ydata=option_prices,
        jac=jac_func
    )[0]
    print(fp)
    pred_option_prices = fit_func(spot_prices, fp[0], fp[1], fp[2])

    plt.plot(spot_prices, option_prices, 'r')
    plt.plot(spot_prices, pred_option_prices, 'b')
    plt.show()


if __name__ == '__main__':
    is_call_val = True
    strike_val = 80.0
    expiry_val = 0.1
    r_val = 0.02
    sigma_val = 0.3

    plot_fitted_call_prices(
        is_call=is_call_val,
        strike=strike_val,
        expiry=expiry_val,
        r=r_val,
        sigma=sigma_val
    )
