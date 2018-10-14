from typing import NamedTuple, Sequence, Tuple
from examples.port_opt.port_opt import PortOpt
import numpy as np
from algorithms.func_approx_spec import FuncApproxSpec
from algorithms.adp.adp_pg import ADPPolicyGradient
from algorithms.rl_pg.pg import PolicyGradient
from processes.mdp_rep_for_adp_pg import MDPRepForADPPG
from processes.mdp_rep_for_rl_pg import MDPRepForRLPG
from func_approx.dnn_spec import DNNSpec

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
                (1. + self.r) ** (t + 1 - self.time_steps)
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

    # noinspection PyMethodMayBeStatic,PyShadowingNames
    def critic_spec(self, neurons: Sequence[int]) -> FuncApproxSpec:

        def feature_func(state: StateType) -> float:
            t = float(state[0])
            # noinspection PyPep8Naming
            W = state[1]
            term1 = self.rho ** (-t)
            term2 = np.exp((self.mu - self.r) ** 2 / (2 * self.sigma ** 2) * t)
            term3 = np.exp(-self.gamma * (1. + self.r) ** (self.time_steps - t) * W)
            return term1 * term2 * term3

        return FuncApproxSpec(
            state_feature_funcs=[feature_func],
            sa_feature_funcs=[lambda x, feature_func=feature_func: feature_func(x[0])],
            dnn_spec=DNNSpec(
                neurons=neurons,
                hidden_activation=DNNSpec.relu,
                hidden_activation_deriv=DNNSpec.relu_deriv,
                output_activation=DNNSpec.identity,
                output_activation_deriv=DNNSpec.identity_deriv
            )
        )

    # noinspection PyShadowingNames
    def actor_spec(self) -> Tuple[FuncApproxSpec, FuncApproxSpec]:
        ff = lambda s: (1. + self.r) ** float(s[0])
        mean = FuncApproxSpec(
            state_feature_funcs=[ff],
            sa_feature_funcs=[lambda x, ff=ff: ff(x[0])],
            dnn_spec=None
        )
        variance = FuncApproxSpec(
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
        return mean, variance

    def state_reward_gen(
        self,
        state: StateType,
        action: ActionType,
        num_samples: int
    ) -> Sequence[Tuple[StateType, float]]:
        # noinspection PyPep8Naming
        t, W = state
        next_states = [(
            t + 1,
            action[0] * (1. + rr) + (W - action[0]) * (1. + self.r)
        ) for rr in self.risky_returns(num_samples)]
        return [(
            x,
            self.rho * self.utility(x[1]) if t == self.time_steps - 1 else 0.
        ) for x in next_states]

    # noinspection PyShadowingNames
    def get_adp_pg_optima(
        self,
        reinforce: bool,
        num_state_samples: int,
        num_next_state_samples: int,
        num_action_samples: int,
        num_batches: int,
        actor_lambda: float,
        critic_lambda: float,
        critic_neurons: Sequence[int]
    ) -> Sequence[float]:
        init_state = SingleAssetCARA.init_state()
        mdp_rep_obj = MDPRepForADPPG(
            self.rho,
            lambda n: [init_state] * n,
            lambda s, a, n: self.state_reward_gen(s, a, n),
            lambda s: s[0] == self.time_steps - 1
        )
        adp_pg_obj = ADPPolicyGradient(
            mdp_rep_for_adp_pg=mdp_rep_obj,
            reinforce=reinforce,
            num_state_samples=num_state_samples,
            num_next_state_samples=num_next_state_samples,
            num_action_samples=num_action_samples,
            num_batches=num_batches,
            max_steps=self.time_steps,
            actor_lambda=actor_lambda,
            critic_lambda=critic_lambda,
            score_func=SingleAssetCARA.score_func,
            sample_actions_gen_func=SingleAssetCARA.sample_actions_gen,
            vf_fa_spec=self.critic_spec(critic_neurons),
            pol_fa_spec=self.actor_spec()
        )
        policy = adp_pg_obj.get_optimal_det_policy_func()
        return [policy((t, 1.)) for t in range(self.time_steps)]

    # noinspection PyShadowingNames
    def get_pg_optima(
        self,
        reinforce: bool,
        batch_size: int,
        num_batches: int,
        num_action_samples: int,
        actor_lambda: float,
        critic_lambda: float,
        critic_neurons: Sequence[int]
    ) -> Sequence[float]:
        init_state = PortOpt.init_state()
        mdp_rep_obj = MDPRepForRLPG(
            self.rho,
            lambda: init_state,
            lambda s, a: self.state_reward_gen(s, a, 1)[0],
            lambda s: s[0] == self.time_steps - 1
        )
        pg_obj = PolicyGradient(
            mdp_rep_for_rl_pg=mdp_rep_obj,
            reinforce=reinforce,
            batch_size=batch_size,
            num_batches=num_batches,
            num_action_samples=num_action_samples,
            max_steps=self.time_steps,
            actor_lambda=actor_lambda,
            critic_lambda=critic_lambda,
            score_func=SingleAssetCARA.score_func,
            sample_actions_gen_func=SingleAssetCARA.sample_actions_gen,
            fa_spec=self.critic_spec(critic_neurons),
            pol_fa_spec=self.actor_spec()
        )
        policy = pg_obj.get_optimal_det_policy_func()
        return [policy((t, 1.)) for t in range(self.time_steps)]


if __name__ == '__main__':
    time_steps_val = 5
    rho_val = 0.98
    r_val = 0.04
    mu_val = 0.06
    sigma_val = 0.04
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

    reinforce_val = True
    num_state_samples_val = 500  # you need at least a few hundred batch size
    num_next_state_samples_val = 50  # a few dozen next states would be good
    num_action_samples_val = 5000  # make this a few thousand
    num_batches_val = 5000  # a few thousand batches
    actor_lambda_val = 0.99
    critic_lambda_val = 0.99
    critic_neurons = []

    adp_pg_opt = mp.get_adp_pg_optima(
        reinforce=reinforce_val,
        num_state_samples=num_state_samples_val,
        num_next_state_samples=num_next_state_samples_val,
        num_action_samples=num_action_samples_val,
        num_batches=num_batches_val,
        actor_lambda=actor_lambda_val,
        critic_lambda=critic_lambda_val,
        critic_neurons=critic_neurons
    )
    print(adp_pg_opt)

    pg_opt = mp.get_pg_optima(
        reinforce=reinforce_val,
        batch_size=num_state_samples_val,
        num_batches=num_batches_val,
        num_action_samples=num_action_samples_val,
        actor_lambda=actor_lambda_val,
        critic_lambda=critic_lambda_val,
        critic_neurons=critic_neurons
    )
    print(pg_opt)
    print(opt_alloc)
