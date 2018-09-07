from typing import NamedTuple, Sequence, Tuple
from examples.port_opt.port_opt import PortOpt
import numpy as np
from algorithms.func_approx_spec import FuncApproxSpec
from func_approx.dnn_spec import DNNSpec
from algorithms.adp.adp_pg import ADPPolicyGradient
from algorithms.rl_pg.pg import PolicyGradient
from processes.mdp_rep_for_adp_pg import MDPRepForADPPG
from processes.mdp_rep_for_rl_pg import MDPRepForRLPG

StateType = Tuple[int, float]
ActionType = Tuple[float]


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

    def risky_returns(self, samples: int) -> np.ndarray:
        return np.random.normal(loc=self.mu, scale=self.sigma, size=samples)

    def utility(self, x: float) -> float:
        return - np.exp(-self.gamma * x) / self.gamma

    @staticmethod
    def score_func(
        action: ActionType,
        params: Tuple[float, float]
    ) -> Tuple[float, float]:
        mean, variance = params
        mean_score = (action[0] - mean) / variance
        variance_score = -0.5 * (1. / variance - mean_score ** 2)
        return mean_score, variance_score

    @staticmethod
    def sample_actions_gen(
        params: Tuple[float, float],
        num_samples: int
    ) -> Sequence[ActionType]:
        mean, variance = params
        return [(x,) for x in np.random.normal(
            loc=mean,
            scale=np.sqrt(variance),
            size=num_samples
        )]

    @staticmethod
    def init_state() -> StateType:
        return 0, 1.

    @staticmethod
    def critic_spec(neurons: Sequence[int]) -> FuncApproxSpec:
        return FuncApproxSpec(
            state_feature_funcs=[
                lambda s: float(s[0]),
                lambda s: s[1]
            ],
            action_feature_funcs=[],
            dnn_spec=DNNSpec(
                neurons=neurons,
                hidden_activation=DNNSpec.log_squish,
                hidden_activation_deriv=DNNSpec.log_squish_deriv,
                output_activation=DNNSpec.identity,
                output_activation_deriv=DNNSpec.identity_deriv
            )
        )

    @staticmethod
    def actor_spec(neurons: Sequence[int]) \
            -> Tuple[FuncApproxSpec, FuncApproxSpec]:
        mean = FuncApproxSpec(
            state_feature_funcs=[
                lambda s: float(s[0]),
                lambda s: s[1]
            ],
            action_feature_funcs=[],
            dnn_spec=DNNSpec(
                neurons=neurons,
                hidden_activation=DNNSpec.log_squish,
                hidden_activation_deriv=DNNSpec.log_squish_deriv,
                output_activation=DNNSpec.identity,
                output_activation_deriv=DNNSpec.identity_deriv
            )
        )
        variance = FuncApproxSpec(
            state_feature_funcs=[
                lambda s: float(s[0]),
                lambda s: s[1]
            ],
            action_feature_funcs=[],
            dnn_spec=DNNSpec(
                neurons=neurons,
                hidden_activation=DNNSpec.log_squish,
                hidden_activation_deriv=DNNSpec.log_squish_deriv,
                output_activation=DNNSpec.pos_log_squish,
                output_activation_deriv=DNNSpec.pos_log_squish_deriv
            )
        )
        return mean, variance

    def state_reward_gen(
        self,
        state: StateType,
        action: ActionType,
        num_samples: int
    ) -> Sequence[Tuple[StateType, float]]:
        # noinspection PyPep8Naming
        t, W = state
        if t == self.time_steps:
            ret = [((t, 0.), self.utility(W))] * num_samples
        else:
            next_states = [(
                t + 1,
                W * (action[0] * (1. + rr) + (1. - action[0]) * (1. + self.r))
            ) for rr in self.risky_returns(num_samples)]
            ret = [(x, 0.) for x in next_states]
        return ret

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
    ) -> Sequence[float]:
        init_state = SingleAssetCARA.init_state()
        mdp_rep_obj = MDPRepForADPPG(
            self.rho,
            lambda n: [init_state] * n,
            lambda s, a, n: self.state_reward_gen(s, a, n),
            lambda s: s[0] == self.time_steps
        )
        adp_pg_obj = ADPPolicyGradient(
            mdp_rep_for_adp_pg=mdp_rep_obj,
            num_state_samples=num_state_samples,
            num_next_state_samples=num_next_state_samples,
            num_action_samples=num_action_samples,
            num_batches=num_batches,
            max_steps=self.time_steps + 2,
            actor_lambda=actor_lambda,
            critic_lambda=critic_lambda,
            score_func=SingleAssetCARA.score_func,
            sample_actions_gen_func=SingleAssetCARA.sample_actions_gen,
            vf_fa_spec=SingleAssetCARA.critic_spec(critic_neurons),
            pol_fa_spec=SingleAssetCARA.actor_spec(actor_neurons)
        )
        policy = adp_pg_obj.get_optimal_det_policy_func()
        return [policy((t, 1.)) for t in range(self.time_steps)]

    def get_pg_optima(
        self,
        batch_size: int,
        num_batches: int,
        num_action_samples: int,
        actor_lambda: float,
        critic_lambda: float,
        actor_neurons: Sequence[int],
        critic_neurons: Sequence[int]
    ) -> Sequence[float]:
        init_state = PortOpt.init_state()
        mdp_rep_obj = MDPRepForRLPG(
            self.rho,
            lambda: init_state,
            lambda s, a: self.state_reward_gen(s, a, 1)[0],
            lambda s: s[0] == self.time_steps
        )
        pg_obj = PolicyGradient(
            mdp_rep_for_rl_pg=mdp_rep_obj,
            batch_size=batch_size,
            num_batches=num_batches,
            num_action_samples=num_action_samples,
            max_steps=self.time_steps + 2,
            actor_lambda=actor_lambda,
            critic_lambda=critic_lambda,
            score_func=SingleAssetCARA.score_func,
            sample_actions_gen_func=SingleAssetCARA.sample_actions_gen,
            fa_spec=SingleAssetCARA.critic_spec(critic_neurons),
            pol_fa_spec=SingleAssetCARA.actor_spec(actor_neurons)
        )
        policy = pg_obj.get_optimal_det_policy_func()
        return [policy((t, 1.)) for t in range(self.time_steps)]


if __name__ == '__main__':
    time_steps_val = 1
    rho_val = 1.0
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

    num_state_samples_val = 25
    num_next_state_samples_val = 20
    num_action_samples_val = 100
    num_batches_val = 1000
    actor_lambda_val = 0.95
    critic_lambda_val = 0.95
    actor_neurons_val = [4, 2]
    critic_neurons_val = [3, 3]

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
