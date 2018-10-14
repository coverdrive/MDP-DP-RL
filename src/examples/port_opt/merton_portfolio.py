from typing import NamedTuple, Callable, Mapping, Tuple
from examples.port_opt.port_opt import PortOpt
from algorithms.func_approx_spec import FuncApproxSpec
from func_approx.dnn_spec import DNNSpec
import numpy as np
from utils.gen_utils import memoize


class MertonPortfolio(NamedTuple):
    expiry: float  # = T
    rho: float  # = discount rate
    r: float  # = risk-free rate
    mu: np.ndarray  # = risky rate means (1-D array of length num risky assets)
    cov: np.ndarray  # = risky rate covariances (2-D square array of length num risky assets)
    epsilon: float  # = bequest parameter
    gamma: float  # = CRRA parameter

    SMALL_POS = 1e-8

    @memoize
    def get_optimal_allocation(self) -> np.ndarray:
        return np.linalg.inv(self.cov).dot(self.mu - self.r) / self.gamma

    @memoize
    def get_nu(self) -> float:
        num = (self.mu - self.r).dot(self.get_optimal_allocation())
        return self.rho / self.gamma - (1. - self.gamma) *\
            (num / (2. * self.gamma) + self.r / self.gamma)

    def get_optimal_consumption(self) -> Callable[[float], float]:
        nu = self.get_nu()

        # noinspection PyShadowingNames
        def opt_cons_func(t: float, nu=nu) -> float:
            if nu == 0:
                opt_cons = 1. / (self.expiry - t + self.epsilon)
            else:
                opt_cons = nu / (1. + (nu * self.epsilon - 1) *
                                 np.exp(-nu * (self.expiry - t)))
            return opt_cons

        return opt_cons_func

    def risky_returns_gen(self, samples: int, delta_t: float) -> np.ndarray:
        return np.random.multivariate_normal(
            mean=(self.mu - 0.5 * self.cov.dot(np.ones(len(self.mu)))) * delta_t,
            cov=self.cov * delta_t,
            size=samples
        )

    def cons_utility(self, x: float) -> float:
        gam = 1. - self.gamma
        return x ** gam / gam if gam != 0. else np.log(x)

    def beq_utility(self, x: float) -> float:
        return self.epsilon ** self.gamma * self.cons_utility(x)

    # noinspection PyShadowingNames
    def get_port_opt_obj(self, time_steps: int) -> PortOpt:
        risky_assets = len(self.mu)
        delta_t = self.expiry / time_steps
        return PortOpt(
            num_risky=risky_assets,
            riskless_returns=[self.r * delta_t] * time_steps,
            returns_gen_funcs=[lambda n, delta_t=delta_t: self.risky_returns_gen(
                n,
                delta_t
            )] * time_steps,
            cons_util_func=lambda x: self.cons_utility(x),
            beq_util_func=lambda x: self.beq_utility(x),
            discount_rate=self.rho * delta_t
        )

    # noinspection PyShadowingNames
    def get_actor_mu_spec(self, time_steps: int) -> FuncApproxSpec:
        tnu = self.get_nu()

        # noinspection PyShadowingNames
        def state_ff(state: Tuple[int, float], tnu=tnu) -> float:
            tte = self.expiry * (1. - float(state[0]) / time_steps)
            if tnu == 0:
                ret = 1. / (tte + self.epsilon)
            else:
                ret = tnu / (1. + (tnu * self.epsilon - 1.) * np.exp(-tnu * tte))
            return ret

        return FuncApproxSpec(
            state_feature_funcs=[state_ff],
            sa_feature_funcs=[lambda x, state_ff=state_ff: state_ff(x[0])],
            dnn_spec=DNNSpec(
                neurons=[],
                hidden_activation=DNNSpec.log_squish,
                hidden_activation_deriv=DNNSpec.log_squish_deriv,
                output_activation=DNNSpec.sigmoid,
                output_activation_deriv=DNNSpec.sigmoid_deriv
            )
        )

    @staticmethod
    def get_actor_nu_spec() -> FuncApproxSpec:
        return FuncApproxSpec(
            state_feature_funcs=[],
            sa_feature_funcs=[],
            dnn_spec=DNNSpec(
                neurons=[],
                hidden_activation=DNNSpec.log_squish,
                hidden_activation_deriv=DNNSpec.log_squish_deriv,
                output_activation=DNNSpec.pos_log_squish,
                output_activation_deriv=DNNSpec.pos_log_squish_deriv
            )
        )

    @staticmethod
    def get_actor_mean_spec() -> FuncApproxSpec:
        return FuncApproxSpec(
            state_feature_funcs=[],
            sa_feature_funcs=[],
            dnn_spec=None
        )

    @staticmethod
    def get_actor_variance_spec() -> FuncApproxSpec:
        return FuncApproxSpec(
            state_feature_funcs=[],
            sa_feature_funcs=[],
            dnn_spec=DNNSpec(
                neurons=[],
                hidden_activation=DNNSpec.log_squish,
                hidden_activation_deriv=DNNSpec.log_squish_deriv,
                output_activation=DNNSpec.pos_log_squish,
                output_activation_deriv=DNNSpec.pos_log_squish_deriv
            )
        )

    # noinspection PyMethodMayBeStatic,PyShadowingNames
    def get_critic_spec(self, time_steps: int) -> FuncApproxSpec:
        tnu = self.get_nu()
        gam = 1. - self.gamma

        # noinspection PyShadowingNames
        def state_ff(
            state: Tuple[int, float],
            tnu=tnu,
            gam=gam
        ) -> float:
            t = float(state[0]) * self.expiry / time_steps
            tte = self.expiry - t
            if tnu == 0:
                ret = tte + self.epsilon
            else:
                ret = (1. + (tnu * self.epsilon - 1.) * np.exp(-tnu * tte)) / tnu
            mult = state[1] ** gam / gam if gam != 0 else np.log(state[1])
            return ret ** self.gamma * mult / np.exp(self.rho * t)

        return FuncApproxSpec(
            state_feature_funcs=[state_ff],
            sa_feature_funcs=[lambda x, state_ff=state_ff: state_ff(x[0])],
            dnn_spec=None
        )

    # noinspection PyShadowingNames
    @memoize
    def get_adp_pg_policy_func(
        self,
        time_steps: int,
        reinforce: bool,
        num_state_samples: int,
        num_next_state_samples: int,
        num_action_samples: int,
        num_batches: int,
        actor_lambda: float,
        critic_lambda: float
    ) -> Callable[[Tuple[int, float]], Tuple[float, ...]]:
        adp_pg_obj = self.get_port_opt_obj(time_steps).get_adp_pg_obj(
            reinforce=reinforce,
            num_state_samples=num_state_samples,
            num_next_state_samples=num_next_state_samples,
            num_action_samples=num_action_samples,
            num_batches=num_batches,
            actor_lambda=actor_lambda,
            critic_lambda=critic_lambda,
            actor_mu_spec=self.get_actor_mu_spec(time_steps),
            actor_nu_spec=MertonPortfolio.get_actor_nu_spec(),
            actor_mean_spec=MertonPortfolio.get_actor_mean_spec(),
            actor_variance_spec=MertonPortfolio.get_actor_variance_spec(),
            critic_spec=self.get_critic_spec(time_steps)
        )
        adp_pg_obj.get_optimal_stoch_policy_func()
        pf = adp_pg_obj.pol_fa

        def ret_pol(s: Tuple[int, float]) -> Tuple[float, ...]:
            cons = pf[0].get_func_eval(s)
            alloc = [f.get_func_eval(s) for f in pf[2:2+len(self.mu)]]
            return tuple([cons] + alloc)

        return ret_pol

    # noinspection PyShadowingNames
    @memoize
    def get_pg_policy_func(
        self,
        time_steps: int,
        reinforce: bool,
        batch_size: int,
        num_batches: int,
        num_action_samples: int,
        actor_lambda: float,
        critic_lambda: float
    ) -> Callable[[Tuple[int, float]], Tuple[float, ...]]:
        pg_obj = self.get_port_opt_obj(time_steps).get_pg_obj(
            reinforce=reinforce,
            batch_size=batch_size,
            num_batches=num_batches,
            num_action_samples=num_action_samples,
            actor_lambda=actor_lambda,
            critic_lambda=critic_lambda,
            actor_mu_spec=self.get_actor_mu_spec(time_steps),
            actor_nu_spec=MertonPortfolio.get_actor_nu_spec(),
            actor_mean_spec=MertonPortfolio.get_actor_mean_spec(),
            actor_variance_spec=MertonPortfolio.get_actor_variance_spec(),
            critic_spec=self.get_critic_spec(time_steps)
        )
        pg_obj.get_optimal_stoch_policy_func()
        pf = pg_obj.pol_fa

        def ret_pol(s: Tuple[int, float]) -> Tuple[float, ...]:
            cons = pf[0].get_func_eval(s)
            alloc = [f.get_func_eval(s) for f in pf[2:2+len(self.mu)]]
            return tuple([cons] + alloc)

        return ret_pol

    @staticmethod
    def get_cons_alloc_from_policy(
        time_steps: int,
        pol: Callable[[Tuple[int, float]], Tuple[float, ...]]
    ) -> Mapping[str, np.ndarray]:
        actions = np.array([pol((t, 1.)) for t in range(time_steps)])
        cons = actions[:, 0]
        alloc = actions[:, 1:]
        return {"Consumptions": cons, "Allocations": alloc}

    def test_opt_policies_vs_merton(
        self,
        time_steps: int,
        adp_pg_policy: Callable[[Tuple[int, float]], Tuple[float, ...]],
        pg_policy: Callable[[Tuple[int, float]], Tuple[float, ...]],
        num_paths: int
    ) -> Mapping[str, float]:
        port_opt = self.get_port_opt_obj(time_steps)
        ma = self.get_optimal_allocation()
        mc = self.get_optimal_consumption()

        # noinspection PyShadowingNames
        def merton_pol(
            s: Tuple[int, float],
            ma=ma,
            mc=mc,
            time_steps=time_steps
        ) -> Tuple[float, ...]:
            sp = MertonPortfolio.SMALL_POS
            cons = max(sp, min(1. - sp, mc(self.expiry * float(s[0]) /
                                           time_steps)))
            alloc = [ma] * len(self.mu)
            return tuple(np.insert(alloc, 0, cons))

        return {
            "Merton": port_opt.test_det_policy(merton_pol, num_paths),
            "ADP PG Opt": port_opt.test_det_policy(adp_pg_policy, num_paths),
            "PG Opt": port_opt.test_det_policy(pg_policy, num_paths)
        }


if __name__ == '__main__':
    expiry_val = 0.4
    rho_val = 0.04
    r_val = 0.04
    mu_val = np.array([0.08])
    cov_val = np.array([[0.0009]])
    epsilon_val = 1e-8
    gamma_val = 0.2

    mp = MertonPortfolio(
        expiry=expiry_val,
        rho=rho_val,
        r=r_val,
        mu=mu_val,
        cov=cov_val,
        epsilon=epsilon_val,
        gamma=gamma_val
    )

    time_steps_val = 5
    reinforce_val = True
    num_state_samples_val = 500
    num_next_state_samples_val = 30
    num_action_samples_val = 200
    num_batches_val = 3000
    actor_lambda_val = 0.99
    critic_lambda_val = 0.99

    opt_alloc = mp.get_optimal_allocation()
    print(opt_alloc)
    opt_cons_func = mp.get_optimal_consumption()
    print([opt_cons_func(t * expiry_val / time_steps_val) for t in range(time_steps_val)])

    adp_pg_pol = mp.get_adp_pg_policy_func(
        time_steps=time_steps_val,
        reinforce=reinforce_val,
        num_state_samples=num_state_samples_val,
        num_next_state_samples=num_next_state_samples_val,
        num_action_samples=num_action_samples_val,
        num_batches=num_batches_val,
        actor_lambda=actor_lambda_val,
        critic_lambda=critic_lambda_val
    )
    adp_pg_cons_alloc = MertonPortfolio.get_cons_alloc_from_policy(
        time_steps_val,
        adp_pg_pol
    )
    print(adp_pg_cons_alloc)

    pg_pol = mp.get_pg_policy_func(
        time_steps=time_steps_val,
        reinforce=reinforce_val,
        batch_size=num_state_samples_val,
        num_batches=num_batches_val,
        num_action_samples=num_action_samples_val,
        actor_lambda=actor_lambda_val,
        critic_lambda=critic_lambda_val
    )
    pg_cons_alloc = MertonPortfolio.get_cons_alloc_from_policy(
        time_steps_val,
        pg_pol
    )
    print(pg_cons_alloc)

    print(opt_alloc)
    print([opt_cons_func(t * expiry_val / time_steps_val) for t in range(time_steps_val)])

    num_test_paths = 5000

    test_res = mp.test_opt_policies_vs_merton(
        time_steps_val,
        adp_pg_pol,
        pg_pol,
        num_test_paths
    )
    print(test_res)
