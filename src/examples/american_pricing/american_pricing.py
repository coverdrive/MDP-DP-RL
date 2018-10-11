from typing import Callable, Sequence, Tuple, Set, Optional, Mapping, Any
import numpy as np
from algorithms.td_algo_enum import TDAlgorithm
from algorithms.rl_func_approx.monte_carlo import MonteCarlo
from algorithms.rl_func_approx.td0 import TD0
from algorithms.rl_func_approx.rl_func_approx_base import RLFuncApproxBase
from algorithms.rl_func_approx.tdlambda import TDLambda
from src.examples.american_pricing.num_utils import get_future_price_mean_var
from processes.mdp_rep_for_rl_fa import MDPRepForRLFA
from algorithms.func_approx_spec import FuncApproxSpec
from func_approx.dnn_spec import DNNSpec
from random import choice

StateType = Tuple[int, np.ndarray]
ActionType = bool


class AmericanPricing:
    """
    In the risk-neutral measure, the underlying price x_t
    follows the Ito process: dx_t = r_t x_t dt + dispersion(t, x_t) dz_t
    spot_price is x_0
    payoff is a function from (t, (x_0, ..., x_t) to payoff (
    eg: \sum_{i=0}^t x_i / (t+1) - K)
    expiry is the time to expiry of american option (in years)
    dispersion(t, x_t) is a function from (t, x_t) to dispersion
    We define ir_t = \int_0^t r_u du, so discount D_t = e^{- ir_t}
    where r_t is the infinitesimal risk-free rate at time t
    """

    def __init__(
        self,
        spot_price: float,
        payoff: Callable[[float, np.ndarray], float],
        expiry: float,
        dispersion: Callable[[float, float], float],
        ir: Callable[[float], float]
    ) -> None:
        self.spot_price: float = spot_price
        self.payoff: Callable[[float, np.ndarray], float] = payoff
        self.expiry: float = expiry
        self.dispersion: Callable[[float, float], float] = dispersion
        self.ir: Callable[[float], float] = ir

    def get_ls_price(
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
                # plt.scatter([paths[i, t] for i in indices], y_vals, c='r')
                # plt.scatter([paths[i, t] for i in indices], estimate, c='b')
                # plt.show()

                for i, ind in enumerate(indices):
                    if payoff[ind] > estimate[i]:
                        cashflow[ind] = payoff[ind]

        return max(
            self.payoff(0, np.array([self.spot_price])),
            np.average(cashflow * np.exp(-self.ir(dt)))
        )

    def state_reward_gen(
        self,
        state: StateType,
        action: ActionType,
        num_dt: int,
        delta_t: float
    ) -> Tuple[StateType, float]:
        ind, price_arr = state
        t = ind * delta_t
        reward = (np.exp(-self.ir(t)) * self.payoff(t, price_arr)) if\
            (action and ind <= num_dt) else 0.
        m, v = get_future_price_mean_var(
            price_arr[-1],
            t,
            delta_t,
            self.ir,
            self.dispersion
        )
        next_price = np.random.normal(m, np.sqrt(v))
        price1 = np.append(price_arr, next_price)
        next_ind = (num_dt if action else ind) + 1
        return (next_ind, price1), reward

    def get_rl_fa_obj(
        self,
        num_dt: int,
        method: str,
        algorithm: TDAlgorithm,
        softmax: bool,
        epsilon: float,
        epsilon_half_life: float,
        lambd: float,
        num_episodes: int,
        feature_funcs: Sequence[Callable[[Tuple[StateType, ActionType]], float]],
        neurons: Optional[Sequence[int]],
        learning_rate: float,
        adam: Tuple[bool, float, float],
        offline: bool
    ) -> RLFuncApproxBase:
        dt = self.expiry / num_dt

        def sa_func(_: StateType) -> Set[ActionType]:
            return {True, False}

        # noinspection PyShadowingNames
        def terminal_state(
            s: StateType,
            num_dt=num_dt
        ) -> bool:
            return s[0] > num_dt

        # noinspection PyShadowingNames
        def sr_func(
            s: StateType,
            a: ActionType,
            num_dt=num_dt,
            dt=dt
        ) -> Tuple[StateType, float]:
            return self.state_reward_gen(s, a, num_dt, dt)

        def init_s() -> StateType:
            return 0, np.array([self.spot_price])

        def init_sa() -> Tuple[StateType, ActionType]:
            return init_s(), choice([True, False])

        # noinspection PyShadowingNames
        mdp_rep_obj = MDPRepForRLFA(
            state_action_func=sa_func,
            gamma=1.,
            terminal_state_func=terminal_state,
            state_reward_gen_func=sr_func,
            init_state_gen=init_s,
            init_state_action_gen=init_sa
        )

        fa_spec = FuncApproxSpec(
            state_feature_funcs=[],
            sa_feature_funcs=feature_funcs,
            dnn_spec=(None if neurons is None else (DNNSpec(
                neurons=neurons,
                hidden_activation=DNNSpec.log_squish,
                hidden_activation_deriv=DNNSpec.log_squish_deriv,
                output_activation=DNNSpec.pos_log_squish,
                output_activation_deriv=DNNSpec.pos_log_squish_deriv
            ))),
            learning_rate=learning_rate,
            adam_params=adam,
            add_unit_feature=False
        )

        if method == "MC":
            ret = MonteCarlo(
                mdp_rep_for_rl=mdp_rep_obj,
                softmax=softmax,
                epsilon=epsilon,
                epsilon_half_life=epsilon_half_life,
                num_episodes=num_episodes,
                max_steps=num_dt + 2,
                fa_spec=fa_spec
            )
        elif method == "TD0":
            ret = TD0(
                mdp_rep_for_rl=mdp_rep_obj,
                algorithm=algorithm,
                softmax=softmax,
                epsilon=epsilon,
                epsilon_half_life=epsilon_half_life,
                num_episodes=num_episodes,
                max_steps=num_dt + 2,
                fa_spec=fa_spec
            )
        else:
            ret = TDLambda(
                mdp_rep_for_rl=mdp_rep_obj,
                algorithm=algorithm,
                softmax=softmax,
                epsilon=epsilon,
                epsilon_half_life=epsilon_half_life,
                lambd=lambd,
                num_episodes=num_episodes,
                max_steps=num_dt + 2,
                fa_spec=fa_spec,
                offline=offline
            )

        return ret

    # noinspection PyShadowingNames
    @staticmethod
    def get_vanilla_american_price(
        is_call: bool,
        spot_price: float,
        strike: float,
        expiry: float,
        r: float,
        sigma: float,
        num_dt: int,
        num_paths: int,
        num_laguerre: int,
        params_bag: Mapping[str, Any]
    ) -> Mapping[str, float]:
        from numpy.polynomial.laguerre import lagval
        from examples.american_pricing.grid_pricing import GridPricing
        payoff = lambda _, x, is_call=is_call, strike=strike:\
            (1 if is_call else -1) * (x - strike)
        dispersion = lambda _, x, sigma=sigma: sigma * x
        # noinspection PyShadowingNames
        ir_func = lambda t, r=r: r * t

        x_lim = 4. * sigma * spot_price * np.sqrt(expiry)
        num_dx = 200
        dx = x_lim / num_dx

        grid_price = GridPricing(
            spot_price=spot_price,
            payoff=payoff,
            expiry=expiry,
            dispersion=dispersion,
            ir=ir_func
        ).get_price(
            num_dt=num_dt,
            dx=dx,
            num_dx=num_dx
        )

        gp = AmericanPricing(
            spot_price=spot_price,
            payoff=(lambda t, x, payoff=payoff: payoff(t, x[-1])),
            expiry=expiry,
            dispersion=dispersion,
            ir=ir_func
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
            return np.exp(-x / (strike * 2)) * \
                   lagval(x / strike, ident[i])

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
            if i < num_laguerre + 4:
                if ind < num_dt and not a:
                    if i == 0:
                        ret = 1.
                    elif i < num_laguerre + 1:
                        ret = laguerre_feature_func(x, i - 1)
                    elif i == num_laguerre + 1:
                        ret = np.sin(-ind * np.pi / (2. * num_dt) + np.pi / 2.)
                    elif i == num_laguerre + 2:
                        ret = np.log(dt * (num_dt - ind))
                    else:
                        rat = float(ind) / num_dt
                        ret = rat * rat
                else:
                    ret = 0.
            else:
                if ind <= num_dt and a:
                    ret = payoff(ind * dt, x)
                else:
                    ret = 0

            return ret

        rl_fa_obj = gp.get_rl_fa_obj(
            num_dt=num_dt,
            method=params_bag["method"],
            algorithm=params_bag["algorithm"],
            softmax=params_bag["softmax"],
            epsilon=params_bag["epsilon"],
            epsilon_half_life=params_bag["epsilon_half_life"],
            lambd=params_bag["lambda"],
            num_episodes=num_paths,
            feature_funcs=[(lambda x, i=i: rl_feature_func(
                x[0][0],
                x[0][1][-1],
                x[1],
                i
            )) for i in range(num_laguerre + 5)],
            neurons=params_bag["neurons"],
            learning_rate=params_bag["learning_rate"],
            adam=params_bag["adam"],
            offline=params_bag["offline"]
        )
        qvf = rl_fa_obj.get_qv_func_fa(None)
        init_s = (0, np.array([spot_price]))
        val_exec = qvf(init_s)(True)
        val_cont = qvf(init_s)(False)
        rl_price = max(val_exec, val_cont)

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
    r_val = 0.02
    sigma_val = 0.25
    num_dt_val = 10
    num_paths_val = 1000000
    num_laguerre_val = 3

    params_bag_val = {
        "method": "TD0",
        "algorithm": TDAlgorithm.ExpectedSARSA,
        "softmax": False,
        "epsilon": 0.2,
        "epsilon_half_life": 100000,
        "neurons": None,
        "learning_rate": 0.05,
        "adam": (True, 0.9, 0.99),
        "lambda": 0.0,
        "offline": False
    }

    am_prices = AmericanPricing.get_vanilla_american_price(
        is_call=is_call_val,
        spot_price=spot_price_val,
        strike=strike_val,
        expiry=expiry_val,
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
