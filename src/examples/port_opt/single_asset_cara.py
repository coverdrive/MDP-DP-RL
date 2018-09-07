from typing import NamedTuple, Sequence
from examples.port_opt.port_opt import PortOpt
import numpy as np


class SingleAssetCARA(NamedTuple):
    time_steps: int  # = T
    rho: float  # = discount factor
    r: float  # = risk-free rate
    mu: float  # = risky rate mean
    sigma: float  # = risky rate std dev
    gamma: float  # = CRRA parameter

    def get_optimal_allocation(self) -> Sequence[float]:
        return [(self.mu - self.r) / (self.sigma ** 2 * self.gamma) *
                (1. + self.r) ** (self.time_steps - t - 1)
                for t in range(self.time_steps)]

    def risky_returns_gen(self, samples: int) -> np.ndarray:
        return np.column_stack((np.random.normal(
            loc=self.mu,
            scale=self.sigma,
            size=samples
        ),))

    def utility(self, x: float) -> float:
        return - np.exp(-self.gamma * x) / self.gamma

    def get_port_opt_obj(self) -> PortOpt:
        return PortOpt(
            num_risky=1,
            riskless_returns=[self.r] * self.time_steps,
            returns_gen_funcs=[lambda n: self.risky_returns_gen(n)] * self.time_steps,
            cons_util_func=lambda _: 0.,
            beq_util_func=lambda x: self.utility(x),
            discount_factor=self.rho
        )

    def get_adp_pg_optima(
        self,
        num_state_samples: int,
        num_next_state_samples: int,
        num_action_samples: int,
        num_batches: int,
        actor_lambda: float,
        critic_lambda: float,
        actor_neurons: Sequence[int],
        critic_neurons: Sequence[int]
    ) -> np.ndarray:
        adp_pg_obj = self.get_port_opt_obj().get_adp_pg_obj(
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
        actions = np.array([policy((t, 1.)) for t in range(self.time_steps)])
        alloc = actions[:, 1]
        return alloc

    def get_pg_optima(
        self,
        batch_size: int,
        num_batches: int,
        num_action_samples: int,
        actor_lambda: float,
        critic_lambda: float,
        actor_neurons: Sequence[int],
        critic_neurons: Sequence[int]
    ) -> np.ndarray:
        pg_obj = self.get_port_opt_obj().get_pg_obj(
            batch_size=batch_size,
            num_batches=num_batches,
            num_action_samples=num_action_samples,
            actor_lambda=actor_lambda,
            critic_lambda=critic_lambda,
            actor_neurons=actor_neurons,
            critic_neurons=critic_neurons
        )
        policy = pg_obj.get_optimal_det_policy_func()
        actions = np.array([policy((t, 1.)) for t in range(self.time_steps)])
        alloc = actions[:, 1]
        return alloc


if __name__ == '__main__':
    time_steps_val = 5
    rho_val = 0.02
    r_val = 0.04
    mu_val = 0.08
    sigma_val = 0.05
    gamma_val = 0.5

    mp = SingleAssetCARA(
        time_steps=time_steps_val,
        rho=rho_val,
        r=r_val,
        mu=mu_val,
        sigma=sigma_val,
        gamma=gamma_val
    )
    opt_alloc = mp.get_optimal_allocation()
    print(opt_alloc)

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
        batch_size=num_state_samples_val,
        num_batches=num_batches_val,
        num_action_samples=num_action_samples_val,
        actor_lambda=actor_lambda_val,
        critic_lambda=critic_lambda_val,
        actor_neurons=actor_neurons_val,
        critic_neurons=critic_neurons_val
    )
    print(pg_opt)
