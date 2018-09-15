from typing import NamedTuple, Callable, Mapping, Tuple
from examples.port_opt.port_opt import PortOpt
from algorithms.func_approx_spec import FuncApproxSpec
from func_approx.dnn_spec import DNNSpec
import numpy as np


class MertonPortfolio(NamedTuple):
    expiry: float  # = T
    rho: float  # = discount rate
    r: float  # = risk-free rate
    mu: np.ndarray  # = risky rate means (1-D array of length num risky assets)
    cov: np.ndarray  # = risky rate covariances (2-D square array of length num risky assets)
    epsilon: float  # = bequest parameter
    gamma: float  # = CRRA parameter

    def get_optimal_allocation(self) -> np.ndarray:
        return np.linalg.inv(self.cov).dot(self.mu - self.r) / self.gamma

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
            mean=self.mu * delta_t,
            cov=self.cov * delta_t,
            size=samples
        )

    def cons_utility(self, x: float) -> float:
        gam = 1. - self.gamma
        return x ** gam / gam if gam != 0. else np.log(x)

    def beq_utility(self, x: float) -> float:
        return self.epsilon ** self.gamma * self.cons_utility(x)

    def get_port_opt_obj(self, time_steps: int) -> PortOpt:
        risky_assets = len(self.mu)
        delta_t = self.expiry / time_steps
        return PortOpt(
            num_risky=risky_assets,
            riskless_returns=[self.r * delta_t] * time_steps,
            returns_gen_funcs=[lambda n: self.risky_returns_gen(n, delta_t)] * time_steps,
            cons_util_func=lambda x: self.cons_utility(x),
            beq_util_func=lambda x: self.beq_utility(x),
            discount_factor=np.exp(-self.rho * delta_t)
        )

    def get_actor_mu_spec(self, time_steps: int) -> FuncApproxSpec:
        tnu = self.get_nu()

        # noinspection PyShadowingNames
        def state_ff(state: Tuple[int, float], tnu=tnu) -> float:
            tte = self.expiry * (1. - float(state[0]) / time_steps)
            if tnu == 0:
                ret = tte + self.epsilon
            else:
                ret = 1. + (tnu * self.epsilon - 1.) * np.exp(-tnu * tte)
            return 1. / ret

        return FuncApproxSpec(
            state_feature_funcs=[state_ff],
            action_feature_funcs=[],
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
            action_feature_funcs=[],
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
            action_feature_funcs=[],
            dnn_spec=None
        )

    @staticmethod
    def get_actor_variance_spec() -> FuncApproxSpec:
        return FuncApproxSpec(
            state_feature_funcs=[],
            action_feature_funcs=[],
            dnn_spec=DNNSpec(
                neurons=[],
                hidden_activation=DNNSpec.log_squish,
                hidden_activation_deriv=DNNSpec.log_squish_deriv,
                output_activation=DNNSpec.pos_log_squish,
                output_activation_deriv=DNNSpec.pos_log_squish_deriv
            )
        )

    # noinspection PyMethodMayBeStatic
    def get_critic_spec(self, time_steps: int) -> FuncApproxSpec:
        tnu = self.get_nu()

        # noinspection PyShadowingNames
        def state_ff(state: Tuple[int, float], tnu=tnu) -> float:
            t = float(state[0]) * self.expiry / time_steps
            tte = self.expiry - t
            if tnu == 0:
                ret = tte + self.epsilon
            else:
                ret = (1. + (tnu * self.epsilon - 1.) * np.exp(-tnu * tte)) / tnu
            mult = state[1] ** (1. - self.gamma) if self.gamma != 1\
                else np.log(state[1])
            return ret ** self.gamma * mult / np.exp(self.rho * t)

        return FuncApproxSpec(
            state_feature_funcs=[state_ff],
            action_feature_funcs=[],
            dnn_spec=None
        )

    # noinspection PyShadowingNames
    def get_adp_pg_optima(
        self,
        time_steps: int,
        num_state_samples: int,
        num_next_state_samples: int,
        num_action_samples: int,
        num_batches: int,
        actor_lambda: float,
        critic_lambda: float
    ) -> Mapping[str, np.ndarray]:
        adp_pg_obj = self.get_port_opt_obj(time_steps).get_adp_pg_obj(
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
        policy = adp_pg_obj.get_optimal_det_policy_func()
        actions = np.array([policy((t * self.expiry / time_steps, 1.)) for t in
                            range(time_steps)])
        cons = actions[:, 0]
        alloc = actions[:, 1:]
        return {"Consumptions": cons, "Allocations": alloc}

    # noinspection PyShadowingNames
    def get_pg_optima(
        self,
        time_steps: int,
        batch_size: int,
        num_batches: int,
        num_action_samples: int,
        actor_lambda: float,
        critic_lambda: float
    ) -> Mapping[str, np.ndarray]:
        pg_obj = self.get_port_opt_obj(time_steps).get_pg_obj(
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
        policy = pg_obj.get_optimal_det_policy_func()
        actions = np.array([policy((t * self.expiry / time_steps, 1.)) for t in
                            range(time_steps)])
        cons = actions[:, 0]
        alloc = actions[:, 1:]
        return {"Consumptions": cons, "Allocations": alloc}


if __name__ == '__main__':
    expiry_val = 1.
    rho_val = 0.02
    r_val = 0.04
    mu_val = np.array([0.08])
    cov_val = np.array([[0.0025]])
    epsilon_val = 1e-6
    gamma_val = 0.5

    mp = MertonPortfolio(
        expiry=expiry_val,
        rho=rho_val,
        r=r_val,
        mu=mu_val,
        cov=cov_val,
        epsilon=epsilon_val,
        gamma=gamma_val
    )

    time_steps_val = 10
    num_state_samples_val = 100
    num_next_state_samples_val = 30
    num_action_samples_val = 1000
    num_batches_val = 1000
    actor_lambda_val = 0.7
    critic_lambda_val = 0.7

    opt_alloc = mp.get_optimal_allocation()
    print(opt_alloc)
    opt_cons_func = mp.get_optimal_consumption()
    print([opt_cons_func(t * expiry_val / time_steps_val) for t in range(time_steps_val)])

    adp_pg_opt = mp.get_adp_pg_optima(
        time_steps=time_steps_val,
        num_state_samples=num_state_samples_val,
        num_next_state_samples=num_next_state_samples_val,
        num_action_samples=num_action_samples_val,
        num_batches=num_batches_val,
        actor_lambda=actor_lambda_val,
        critic_lambda=critic_lambda_val
    )
    print(adp_pg_opt)

    pg_opt = mp.get_pg_optima(
        time_steps=time_steps_val,
        batch_size=num_state_samples_val,
        num_batches=num_batches_val,
        num_action_samples=num_action_samples_val,
        actor_lambda=actor_lambda_val,
        critic_lambda=critic_lambda_val
    )
    print(pg_opt)
