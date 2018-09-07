from typing import NamedTuple, Callable, Sequence, Mapping
from examples.port_opt.port_opt import PortOpt
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

    def get_optimal_consumption(self) -> Callable[[float], float]:
        num = (self.mu - self.r).dot(self.get_optimal_allocation())
        nu = self.rho / self.gamma - (1. - self.gamma) * (num / (2. * self.gamma)
                                                          + self.r / self.gamma)

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
            returns_gen_funcs=[lambda n: self.risky_returns_gen(n, delta_t)] *
                              time_steps,
            cons_util_func=lambda x: self.cons_utility(x),
            beq_util_func=lambda x: self.beq_utility(x),
            discount_factor=np.exp(-self.rho * self.expiry / time_steps)
        )

    def get_adp_pg_optima(
        self,
        time_steps: int,
        num_state_samples: int,
        num_next_state_samples: int,
        num_action_samples: int,
        num_batches: int,
        actor_lambda: float,
        critic_lambda: float,
        actor_neurons: Sequence[int],
        critic_neurons: Sequence[int]
    ) -> Mapping[str, np.ndarray]:
        adp_pg_obj = self.get_port_opt_obj(time_steps).get_adp_pg_obj(
            num_state_samples=num_state_samples,
            num_next_state_samples=num_next_state_samples,
            num_action_samples=num_action_samples,
            num_batches=num_batches,
            actor_lambda=actor_lambda,
            critic_lambda=critic_lambda,
            actor_neurons=actor_neurons,
            critic_neurons=critic_neurons
        )
        policy = adp_pg_obj.get_optimal_det_policy_func()
        actions = np.array([policy((t * self.expiry / time_steps, 1.)) for t in
                            range(time_steps)])
        cons = actions[:, 0]
        alloc = actions[:, 1:]
        return {"Consumptions": cons, "Allocations": alloc}

    def get_pg_optima(
        self,
        time_steps: int,
        batch_size: int,
        num_batches: int,
        num_action_samples: int,
        actor_lambda: float,
        critic_lambda: float,
        actor_neurons: Sequence[int],
        critic_neurons: Sequence[int]
    ) -> Mapping[str, np.ndarray]:
        pg_obj = self.get_port_opt_obj(time_steps).get_pg_obj(
            batch_size=batch_size,
            num_batches=num_batches,
            num_action_samples=num_action_samples,
            actor_lambda=actor_lambda,
            critic_lambda=critic_lambda,
            actor_neurons=actor_neurons,
            critic_neurons=critic_neurons
        )
        policy = pg_obj.get_optimal_det_policy_func()
        actions = np.array([policy((t * self.expiry / time_steps, 1.)) for t in
                            range(time_steps)])
        cons = actions[:, 0]
        alloc = actions[:, 1:]
        return {"Consumptions": cons, "Allocations": alloc}


if __name__ == '__main__':
    expiry_val = 1
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
    opt_alloc = mp.get_optimal_allocation()
    print(opt_alloc)
    opt_cons_func = mp.get_optimal_consumption()
    print([opt_cons_func(t) for t in range(expiry_val)])

    time_steps_val = 1
    num_state_samples_val = 50
    num_next_state_samples_val = 20
    num_action_samples_val = 50
    num_batches_val = 100
    actor_lambda_val = 0.95
    critic_lambda_val = 0.95
    actor_neurons_val = [4]
    critic_neurons_val = [3]

    adp_pg_opt = mp.get_adp_pg_optima(
        time_steps=time_steps_val,
        num_state_samples=num_state_samples_val,
        num_next_state_samples=num_next_state_samples_val,
        num_action_samples=num_action_samples_val,
        num_batches=num_batches_val,
        actor_lambda=actor_lambda_val,
        critic_lambda=critic_lambda_val,
        actor_neurons=actor_neurons_val,
        critic_neurons=critic_neurons_val
    )
    print(adp_pg_opt)

    pg_opt = mp.get_pg_optima(
        time_steps=time_steps_val,
        batch_size=num_state_samples_val,
        num_batches=num_batches_val,
        num_action_samples=num_action_samples_val,
        actor_lambda=actor_lambda_val,
        critic_lambda=critic_lambda_val,
        actor_neurons=actor_neurons_val,
        critic_neurons=critic_neurons_val
    )
    print(pg_opt)
