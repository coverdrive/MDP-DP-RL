from typing import Mapping, Any
import numpy as np
from algorithms.td_algo_enum import TDAlgorithm
from numpy.polynomial.laguerre import lagval
from examples.american_pricing.american_pricing import AmericanPricing
from examples.american_pricing.grid_pricing import GridPricing
from src.examples.american_pricing.num_utils import get_future_price_mean_var

LARGENUM = 1e8


# noinspection PyShadowingNames
def get_vanilla_american_price(
    is_call: bool,
    spot_price: float,
    strike: float,
    expiry: float,
    lognormal: bool,
    r: float,
    sigma: float,
    num_dt: int,
    num_paths: int,
    num_laguerre: int,
    params_bag: Mapping[str, Any]
) -> Mapping[str, float]:
    opt_payoff = lambda _, x, is_call=is_call, strike=strike:\
        max(x - strike, 0.) if is_call else max(strike - x, 0.)
    # noinspection PyShadowingNames
    ir_func = lambda t, r=r: r * t
    isig_func = lambda t, sigma=sigma: sigma * sigma * t

    num_dx = 200
    expiry_mean, expiry_var = get_future_price_mean_var(
        spot_price,
        0.,
        expiry,
        lognormal,
        ir_func,
        isig_func
    )
    grid_price = GridPricing(
        spot_price=spot_price,
        payoff=opt_payoff,
        expiry=expiry,
        lognormal=lognormal,
        ir=ir_func,
        isig=isig_func
    ).get_price(
        num_dt=num_dt,
        num_dx=num_dx,
        center=expiry_mean,
        width=np.sqrt(expiry_var) * 4
    )

    gp = AmericanPricing(
        spot_price=spot_price,
        payoff=(lambda t, x, opt_payoff=opt_payoff: opt_payoff(t, x[-1])),
        expiry=expiry,
        lognormal=lognormal,
        ir=ir_func,
        isig=isig_func
    )
    ident = np.eye(num_laguerre)

    # noinspection PyShadowingNames
    def laguerre_feature_func(
        x: float,
        i: int,
        ident=ident,
        strike=strike
    ) -> float:
        # noinspection PyTypeChecker
        xp = x / strike
        return np.exp(-xp / 2) * lagval(xp, ident[i])

    ls_price = gp.get_ls_price(
        num_dt=num_dt,
        num_paths=num_paths,
        feature_funcs=[lambda _, x: 1.] +
                      [(lambda _, x, i=i: laguerre_feature_func(x[-1], i)) for i in
                       range(num_laguerre)]
    )

    # noinspection PyShadowingNames
    def rl_feature_func(
        ind: int,
        x: float,
        a: bool,
        i: int,
        num_laguerre: int = num_laguerre,
        num_dt: int = num_dt,
        expiry: float = expiry
    ) -> float:
        dt = expiry / num_dt
        t = ind * dt
        if i < num_laguerre + 4:
            if ind < num_dt and not a:
                if i == 0:
                    ret = 1.
                elif i < num_laguerre + 1:
                    ret = laguerre_feature_func(x, i - 1)
                elif i == num_laguerre + 1:
                    ret = np.sin(-t * np.pi / (2. * expiry) + np.pi / 2.)
                elif i == num_laguerre + 2:
                    ret = np.log(expiry - t)
                else:
                    rat = t / expiry
                    ret = rat * rat
            else:
                ret = 0.
        else:
            if ind <= num_dt and a:
                ret = np.exp(-r * (ind * dt)) * opt_payoff(ind * dt, x)
            else:
                ret = 0.

        return ret

    rl_price = gp.get_rl_fa_price(
        num_dt=num_dt,
        method=params_bag["method"],
        exploring_start=params_bag["exploring_start"],
        algorithm=params_bag["algorithm"],
        softmax=params_bag["softmax"],
        epsilon=params_bag["epsilon"],
        epsilon_half_life=params_bag["epsilon_half_life"],
        lambd=params_bag["lambda"],
        num_paths=num_paths,
        batch_size=params_bag["batch_size"],
        feature_funcs=[(lambda x, i=i: rl_feature_func(
            x[0][0],
            x[0][1][-1],
            x[1],
            i
        )) for i in range(num_laguerre + 5)],
        neurons=params_bag["neurons"],
        learning_rate=params_bag["learning_rate"],
        learning_rate_decay=params_bag["learning_rate_decay"],
        adam=params_bag["adam"],
        offline=params_bag["offline"]
    )

    return {
        "Grid": grid_price,
        "LS": ls_price,
        "RL": rl_price
    }


if __name__ == '__main__':
    is_call_val = False
    spot_price_val = 80.0
    strike_val = 75.0
    expiry_val = 2.0
    lognormal_val = True
    r_val = 0.02
    sigma_val = 0.25
    num_dt_val = 10
    num_paths_val = 1000000
    num_laguerre_val = 3

    params_bag_val = {
        "method": "LSPI",
        "exploring_start": False,
        "algorithm": TDAlgorithm.ExpectedSARSA,
        "softmax": False,
        "epsilon": 0.2,
        "epsilon_half_life": 100000,
        "batch_size": 10000,
        "neurons": None,
        "learning_rate": 0.03,
        "learning_rate_decay": 10000,
        "adam": (True, 0.9, 0.99),
        "lambda": 0.8,
        "offline": True,
    }

    am_prices = get_vanilla_american_price(
        is_call=is_call_val,
        spot_price=spot_price_val,
        strike=strike_val,
        expiry=expiry_val,
        lognormal=lognormal_val,
        r=r_val,
        sigma=sigma_val,
        num_dt=num_dt_val,
        num_paths=num_paths_val,
        num_laguerre=num_laguerre_val,
        params_bag=params_bag_val
    )
    print(am_prices)
    print(params_bag_val)
    from examples.american_pricing.bs_pricing import EuropeanBSPricing

    ebsp = EuropeanBSPricing(
        is_call=is_call_val,
        spot_price=spot_price_val,
        strike=strike_val,
        expiry=expiry_val,
        r=r_val,
        sigma=sigma_val
    )
    print(ebsp.option_price)
