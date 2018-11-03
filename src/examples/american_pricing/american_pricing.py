from typing import Callable, Sequence, Tuple, Set, Optional
import numpy as np
from algorithms.td_algo_enum import TDAlgorithm
from algorithms.rl_func_approx.monte_carlo import MonteCarlo
from algorithms.rl_func_approx.td0 import TD0
from algorithms.rl_func_approx.tdlambda import TDLambda
from algorithms.rl_func_approx.tdlambda_exact import TDLambdaExact
from algorithms.rl_func_approx.lspi import LSPI
from src.examples.american_pricing.num_utils import get_future_price_mean_var
from processes.mdp_rep_for_rl_fa import MDPRepForRLFA
from algorithms.func_approx_spec import FuncApproxSpec
from func_approx.dnn_spec import DNNSpec
from utils.gen_utils import memoize
from random import choice

StateType = Tuple[int, np.ndarray]
ActionType = bool
OneMinusEpsilon = 1. - 1e4


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

    @memoize
    def get_all_paths(self, num_paths: int, num_dt: int) -> np.ndarray:
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
        return paths

    def get_ls_price(
        self,
        num_dt: int,
        num_paths: int,
        feature_funcs: Sequence[Callable[[float, np.ndarray], float]]
    ) -> float:
        paths = self.get_all_paths(num_paths, num_dt)
        cashflow = np.array([max(self.payoff(self.expiry, paths[i, :]), 0.)
                             for i in range(num_paths)])
        dt = self.expiry / num_dt
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
            self.payoff(0., np.array([self.spot_price])),
            np.average(cashflow * np.exp(-self.ir(dt)))
        )

    def state_reward_gen(
        self,
        state: StateType,
        action: ActionType,
        num_dt: int
    ) -> Tuple[StateType, float]:
        ind, price_arr = state
        delta_t = self.expiry / num_dt
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

    def get_rl_fa_price(
        self,
        num_dt: int,
        method: str,
        exploring_start: bool,
        algorithm: TDAlgorithm,
        softmax: bool,
        epsilon: float,
        epsilon_half_life: float,
        lambd: float,
        num_paths: int,
        batch_size: int,
        feature_funcs: Sequence[Callable[[Tuple[StateType, ActionType]], float]],
        neurons: Optional[Sequence[int]],
        learning_rate: float,
        learning_rate_decay: float,
        adam: Tuple[bool, float, float],
        offline: bool
    ) -> float:
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
            num_dt=num_dt
        ) -> Tuple[StateType, float]:
            return self.state_reward_gen(s, a, num_dt)

        def init_s() -> StateType:
            return 0, np.array([self.spot_price])

        def init_sa() -> Tuple[StateType, ActionType]:
            return init_s(), choice([True, False])

        # noinspection PyShadowingNames
        mdp_rep_obj = MDPRepForRLFA(
            state_action_func=sa_func,
            gamma=OneMinusEpsilon,
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
            rl_fa_obj = MonteCarlo(
                mdp_rep_for_rl=mdp_rep_obj,
                exploring_start=exploring_start,
                softmax=softmax,
                epsilon=epsilon,
                epsilon_half_life=epsilon_half_life,
                num_episodes=num_paths,
                max_steps=num_dt + 2,
                fa_spec=fa_spec
            )
        elif method == "TD0":
            rl_fa_obj = TD0(
                mdp_rep_for_rl=mdp_rep_obj,
                exploring_start=exploring_start,
                algorithm=algorithm,
                softmax=softmax,
                epsilon=epsilon,
                epsilon_half_life=epsilon_half_life,
                num_episodes=num_paths,
                max_steps=num_dt + 2,
                fa_spec=fa_spec
            )
        elif method == "TDL":
            rl_fa_obj = TDLambda(
                mdp_rep_for_rl=mdp_rep_obj,
                exploring_start=exploring_start,
                algorithm=algorithm,
                softmax=softmax,
                epsilon=epsilon,
                epsilon_half_life=epsilon_half_life,
                lambd=lambd,
                num_episodes=num_paths,
                batch_size=batch_size,
                max_steps=num_dt + 2,
                fa_spec=fa_spec,
                offline=offline
            )
        elif method == "TDE":
            rl_fa_obj = TDLambdaExact(
                mdp_rep_for_rl=mdp_rep_obj,
                exploring_start=exploring_start,
                algorithm=algorithm,
                softmax=softmax,
                epsilon=epsilon,
                epsilon_half_life=epsilon_half_life,
                lambd=lambd,
                num_episodes=num_paths,
                batch_size=batch_size,
                max_steps=num_dt + 2,
                state_feature_funcs=[],
                sa_feature_funcs=feature_funcs,
                learning_rate=learning_rate,
                learning_rate_decay=learning_rate_decay
            )
        else:
            rl_fa_obj = LSPI(
                mdp_rep_for_rl=mdp_rep_obj,
                exploring_start=exploring_start,
                softmax=softmax,
                epsilon=epsilon,
                epsilon_half_life=epsilon_half_life,
                num_episodes=num_paths,
                batch_size=batch_size,
                max_steps=num_dt + 2,
                state_feature_funcs=[],
                sa_feature_funcs=feature_funcs
            )

        qvf = rl_fa_obj.get_qv_func_fa(None)
        # init_s = (0, np.array([self.spot_price]))
        # val_exec = qvf(init_s)(True)
        # val_cont = qvf(init_s)(False)
        # true_false_spot_max = max(val_exec, val_cont)

        all_paths = self.get_all_paths(num_paths, num_dt + 1)
        prices = np.zeros(num_paths)

        for path_num, path in enumerate(all_paths):
            steps = 0
            while steps <= num_dt:
                price_seq = path[:(steps + 1)]
                state = (steps, price_seq)
                exercise_price = np.exp(-self.ir(dt * steps)) *\
                    self.payoff(dt * steps, price_seq)
                continue_price = qvf(state)(False)
                steps += 1
                if exercise_price > continue_price:
                    prices[path_num] = exercise_price
                    steps = num_dt + 1
                    # print(state)
                    # print(exercise_price)
                    # print(continue_price)
                    # print(qvf(state)(True))

        return np.average(prices)

    def get_lspi_price(
        self,
        num_dt: int,
        num_paths: int,
        feature_funcs: Sequence[Callable[[int, np.ndarray], float]],
        batch_size: int
    ) -> float:
        features = len(feature_funcs)
        a_mat = np.zeros((features, features))
        b_vec = np.zeros(features)
        params = np.zeros(features)
        paths = self.get_all_paths(num_paths, num_dt + 1)
        dt = self.expiry / num_dt

        for path_num, path in enumerate(paths):

            for step in range(num_dt):
                t = step * dt
                disc = np.exp(self.ir(t + dt) - self.ir(t))
                phi_s = np.array([f(step, paths[:(step + 1)]) for f in feature_funcs])
                next_payoff = self.payoff(t + dt, path[:(step + 2)])
                exercise = next_payoff > params.dot(
                    [f(step + 1, path[:(step + 2)]) for f in feature_funcs]
                )
                phi_sp = 0. if exercise else np.array(
                    [f(step + 1, path[:(step + 2)]) for f in feature_funcs]
                )
                reward = next_payoff if exercise else 0.
                a_mat += np.outer(
                    phi_s,
                    phi_s - phi_sp * disc
                )
                b_vec += reward * disc * phi_s

            if (path_num + 1) % batch_size == 0:
                params = [np.linalg.inv(a_mat).dot(b_vec)]
                a_mat = np.zeros((features, features))
                b_vec = np.zeros(features)

        all_paths = self.get_all_paths(num_paths, num_dt + 1)
        prices = np.zeros(num_paths)

        for path_num, path in enumerate(all_paths):
            step = 0
            while step <= num_dt:
                t = dt * step
                price_seq = path[:(step + 1)]
                exercise_price = self.payoff(t, price_seq)
                continue_price = params.dot([f(step, price_seq) for f in
                                             feature_funcs])
                step += 1
                if exercise_price > continue_price:
                    prices[path_num] = np.exp(-self.ir(t) * exercise_price)
                    step = num_dt + 1

        return np.average(prices)


if __name__ == '__main__':
    is_call_val = False
    spot_price_val = 80.0
    strike_val = 75.0
    expiry_val = 2.0
    r_val = 0.02
    sigma_val = 0.25
    num_dt_val = 10
    num_paths_val = 100000
    num_laguerre_val = 3
    batch_size_val = 1000

    from examples.american_pricing.bs_pricing import EuropeanBSPricing
    ebsp = EuropeanBSPricing(
        is_call=is_call_val,
        spot_price=spot_price_val,
        strike=strike_val,
        expiry=expiry_val,
        r=r_val,
        sigma=sigma_val
    )
    print("European Price = %.3f" % ebsp.option_price)

    def vanilla_american_payoff(_: float, x: np.ndarray) -> float:
        if is_call_val:
            ret = max(x[-1] - strike_val, 0.)
        else:
            ret = max(strike_val - x[-1], 0.)
        return ret

    amp = AmericanPricing(
        spot_price=spot_price_val,
        payoff=lambda t, x: vanilla_american_payoff(t, x),
        expiry=expiry_val,
        dispersion=lambda _, x: sigma_val * x,
        ir=lambda t: r_val * t
    )

    ident = np.eye(num_laguerre_val)

    from numpy.polynomial.laguerre import lagval

    # noinspection PyShadowingNames
    def laguerre_feature_func(
        x: float,
        i: int
    ) -> float:
        # noinspection PyTypeChecker
        return np.exp(-x / (strike_val * 2)) * \
               lagval(x / strike_val, ident[i])

    ls_price = amp.get_ls_price(
        num_dt=num_dt_val,
        num_paths=num_paths_val,
        feature_funcs=[lambda _, x: 1.] +
                      [(lambda _, x, i=i: laguerre_feature_func(x[-1], i)) for i in
                       range(num_laguerre_val)]
    )

    print("Longstaff-Schwartz Price = %.3f" % ls_price)

    def rl_feature_func(
        t: float,
        x: float,
        i: int
    ) -> float:
        if i == 0:
            ret = 1.
        elif i < num_laguerre_val + 1:
            ret = laguerre_feature_func(x, i - 1)
        elif i == num_laguerre_val + 1:
            ret = np.sin(-t * np.pi / (2. * expiry_val) + np.pi / 2.)
        elif i == num_laguerre_val + 2:
            ret = np.log(expiry_val - t)
        else:
            rat = t / expiry_val
            ret = rat * rat
        return ret

    lspi_price = amp.get_lspi_price(
        num_dt=num_dt_val,
        num_paths=num_paths_val,
        feature_funcs=[lambda t, x, i=i: rl_feature_func(t, x[-1], i) for i in
                       range(num_laguerre_val + 4)],
        batch_size=batch_size_val
    )

    print("LSPI Price = %.3f" % lspi_price)
