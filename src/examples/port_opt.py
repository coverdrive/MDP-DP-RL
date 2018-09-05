from typing import Tuple, Generic, Sequence, Callable
import numpy as np
from func_approx.dnn_spec import DNNSpec
from algorithms.adp.adp_pg import ADPPolicyGradient
from algorithms.rl_pg.pg import PolicyGradient
from scipy.special import digamma
from processes.mdp_rep_for_adp_pg import MDPRepForADPPG
from processes.mdp_rep_for_rl_pg import MDPRepForRLPG
from algorithms.func_approx_spec import FuncApproxSpec

StateType = Tuple[int, float]
ActionType = Tuple[float, ...]


class PortOpt:

    def __init__(
        self,
        num_risky: int,
        riskless_returns: Sequence[float],
        returns_gen_funcs: Sequence[Callable[[int], np.ndarray]],
        cons_util_func: Callable[[float], float],
        beq_util_func: Callable[[float], float],
        discount_factor: float
    ) -> None:
        if PortOpt.validate_spec(
            num_risky,
            riskless_returns,
            returns_gen_funcs,
            discount_factor
        ):
            self.num_risky: int = num_risky
            self.riskless_returns: Sequence[float] = riskless_returns
            self.epochs: int = len(riskless_returns)
            self.returns_gen_funcs: Sequence[Callable[[int], np.ndarray]]\
                = returns_gen_funcs
            self.cons_util_func: Callable[[float], float] = cons_util_func
            self.beq_util_func: Callable[[float], float] = beq_util_func
            self.discount_factor: float = discount_factor
        else:
            raise ValueError

    @staticmethod
    def validate_spec(
        num_risky_assets: int,
        riskless_returns_seq: Sequence[float],
        returns_gen: Sequence[Callable[[int], np.ndarray]],
        disc_fact: float
    ) -> bool:
        b1 = num_risky_assets >= 1
        b2 = all(x > 0 for x in riskless_returns_seq)
        b3 = len(riskless_returns_seq) == len(returns_gen)
        b4 = 0. <= disc_fact <= 1.
        return all([b1, b2, b3, b4])

    # Epoch t is from time t to time (t+1), for 0 <= t < T
    # where T = number of epochs. At time T (i.e., at end of epoch
    # (T-1)), the process ends and W_T is the quantity of bequest.
    # The order of operations in an epoch t (0 <= t < T) is:
    # 1) Observe the state (t, W_t).
    # 2) Consume C_t . W_t so wealth drops to W_t . (1 - C_t)
    # 3) Allocate W_t . (1 - C_t) to n risky assets and 1 riskless asset
    # 4) Riskless asset grows by r_t and risky assets grow stochastically
    #    with wealth growing from W_t . (1 - C_t) to W_{t+1}
    # 5) At the end of final epoch (T-1) (i.e., at time T), bequest W_T.
    #
    # U_Beq(W_T) is the utility of bequest, U_Cons(W_t) is the utility
    # State at the start of epoch t is (t, W_t)
    # Action upon observation of state (t, W_t) is (C_t, A_1, .. A_n)
    # where 0 <= C_t <= 1 is the consumption and A_i, i = 1 .. n, is the
    # allocation to risky assets. Allocation to riskless asset will be set to
    # A_0 = 1 - \sum_{i=1}^n A_i. If stochastic return of risky asset i is R_i:
    # W_{t+1} = W_t . (1 - C_t) . (A_0 . (1+r) + \sum_{i=1}^n A_i . (1 + R_i))

    @staticmethod
    def score_func(
        action: ActionType,
        params: Sequence[float]
    ) -> Sequence[float]:
        """
        :param action: is a tuple (a_0, a_1, ...., a_n) where a1
        is the consumption, and a_i, 1 <= i <= n, is the allocation
        for risky asset i.
        :param params: is a Sequence (alpha, beta, sigma^2_1, ..., sigma^2_n,
        mu_1,..., mu_n) (of length 2n + 2) where (alpha, beta) describes the
        beta distribution for the consumption 0 <= a_0 <= 1, and and
        (mu_i, sigma^2_i) describes the normal distribution for the allocation
        a_i of risky asset i, 1 <= i <= n.
        :return: nabla_{params} [log_e p(a_0, a_1, ..., a_n; alpha, beta,
        sigma^2_1, ..., sigma^2_n, mu_1, ..., mu_n)], which is a Sequence of
        length 2n+2
        """
        n = len(action) - 1
        alpha, beta = params[:2]
        variances = params[2:2+n]
        means = params[2+n:]
        cons = action[0]
        alloc = action[1:]
        # beta distrubution PDF p(cons; alpga, beta) = gamma(alpha + beta) /
        # (gamma(alpha) * gamma(beta)) * cons^{alpha-1} * (1-cons)^(beta-1)
        # score_{alpha} = d/d(alpha) (log p) = digamma(alpha + beta) -
        # digamma(alpha) + log(cons)
        # score_{beta} = d/d(beta) (log p) = digamma(alpha + beta) -
        # digamma(beta) + log(1 - cons)
        dig_ab = digamma(alpha + beta)
        alpha_score = dig_ab - digamma(alpha) + np.log(cons)
        beta_score = dig_ab - digamma(beta) + np.log(1. - cons)
        # normal distribution PDF p(alloc; mean, var) = 1/sqrt(2 pi var) .
        # e^{-(alloc - mean)^2 / (2 var)}
        # score_{mean} = d/d(mean) (log p) = (alloc - mean) / var
        # score_{var} = d/d(var) (log p) = -0.5 * (1/var - (alloc - mean)^2 / var^2)
        means_score = [(alloc[i] - means[i]) / v for i, v in
                       enumerate(variances)]
        variances_score = [-0.5 * (1. / v - means_score[i] ** 2) for i, v in
                           enumerate(variances)]
        return [alpha_score, beta_score] + variances_score + means_score

    @staticmethod
    def sample_actions_gen(
        params: Sequence[float],
        num_samples: int
    ) -> Sequence[ActionType]:
        """
        :param params: is a sequence (alpha, betam sigma^2_1, ..., sigma^2_n,
        mu_1, ..., mu_n (of length 2n+2) where (alpha, beta) describes the
        beta distribution for the consumption 0 <= a_0 <= 1, and (mu_1, sigm1^2_i)
        describes the normal distribution for the allocation a_i of risky asset i,
        1 <= i <= n.
        :param num_samples: number of samples
        :return: list (of length num_samples) of (n+1)-tuples (a_0, a_1, ...., a_n)
        """
        n = len(params) / 2 - 1
        alpha, beta = params[:2]
        variances = params[2:2+n]
        means = params[2+n:]
        cons_samples = np.random.beta(a=alpha, b=beta, size=num_samples)
        alloc_samples = [np.random.normal(
            loc=means[i],
            scale=np.sqrt(v),
            size=num_samples
        ) for i, v in enumerate(variances)]
        return [tuple(x) for x in np.vstack([cons_samples] + alloc_samples).T]

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
                hidden_activation=DNNSpec.relu,
                hidden_activation_deriv=DNNSpec.relu_deriv,
                output_activation=DNNSpec.identity,
                output_activation_deriv=DNNSpec.identity_deriv
            )
        )

    @staticmethod
    def actor_spec(neurons: Sequence[int], num_risky: int)\
            -> Sequence[FuncApproxSpec]:
        alpha_beta_vars = [FuncApproxSpec(
            state_feature_funcs=[
                lambda s: float(s[0]),
                lambda s: s[1]
            ],
            action_feature_funcs=[],
            dnn_spec=DNNSpec(
                neurons=neurons,
                hidden_activation=DNNSpec.relu,
                hidden_activation_deriv=DNNSpec.relu_deriv,
                output_activation=DNNSpec.softplus,
                output_activation_deriv=DNNSpec.softplus_deriv
            )
        ) for _ in range(num_risky + 2)]
        means = [FuncApproxSpec(
            state_feature_funcs=[
                lambda s: float(s[0]),
                lambda s: s[1]
            ],
            action_feature_funcs=[],
            dnn_spec=DNNSpec(
                neurons=neurons,
                hidden_activation=DNNSpec.relu,
                hidden_activation_deriv=DNNSpec.relu_deriv,
                output_activation=DNNSpec.identity,
                output_activation_deriv=DNNSpec.identity_deriv
            )
        ) for _ in range(num_risky)]
        return alpha_beta_vars + means

    # noinspection PyPep8Naming
    def state_reward_gen(
        self,
        state: StateType,
        action: ActionType,
        num_samples: int
    ) -> Sequence[Tuple[StateType, float]]:
        t, W = state
        if t == self.epochs:
            ret = [((t, 0.), self.beq_util_func(W))] * num_samples
        else:
            cons = action[0]
            risky_alloc = action[1:]
            riskless_alloc = 1. - sum(risky_alloc)
            alloc = np.insert(np.array(risky_alloc), 0, riskless_alloc)
            ret_samples = np.hstack((
                np.full((num_samples, 1), self.riskless_returns[t]),
                self.returns_gen_funcs[t](num_samples)
            ))
            W1 = W * (1 - cons)
            ret = [((t + 1, W1 * alloc.dot(1 + rs)), self.cons_util_func(cons))
                   for rs in ret_samples]
        return ret

    def get_adp_pg_obj(
        self,
        num_state_samples: int,
        num_next_state_samples: int,
        num_action_samples: int,
        num_batches: int,
        actor_lambda: float,
        critic_lambda: float,
        actor_neurons: Sequence[int],
        critic_neurons: Sequence[int]
    ) -> ADPPolicyGradient:
        init_state = PortOpt.init_state()
        mdp_rep_obj = MDPRepForADPPG(
            self.discount_factor,
            lambda n: [init_state] * n,
            lambda s, a, n: self.state_reward_gen(s, a, n),
            lambda s: s[0] == self.epochs
        )
        return ADPPolicyGradient(
            mdp_rep_for_adp_pg=mdp_rep_obj,
            num_state_samples=num_state_samples,
            num_next_state_samples=num_next_state_samples,
            num_action_samples=num_action_samples,
            num_batches=num_batches,
            max_steps=self.epochs + 2,
            actor_lambda=actor_lambda,
            critic_lambda=critic_lambda,
            score_func=PortOpt.score_func,
            sample_actions_gen_func=PortOpt.sample_actions_gen,
            vf_fa_spec=PortOpt.critic_spec(critic_neurons),
            pol_fa_spec=PortOpt.actor_spec(actor_neurons, self.num_risky)
        )

    def get_pg_obj(
        self,
        batch_size: int,
        num_batches: int,
        num_action_samples: int,
        actor_lambda: float,
        critic_lambda: float,
        actor_neurons: Sequence[int],
        critic_neurons: Sequence[int]
    ) -> PolicyGradient:
        init_state = PortOpt.init_state()
        mdp_rep_obj = MDPRepForRLPG(
            self.discount_factor,
            lambda: init_state,
            lambda s, a: self.state_reward_gen(s, a, 1)[0],
            lambda s: s[0] == self.epochs
        )
        return PolicyGradient(
            mdp_rep_for_rl_pg=mdp_rep_obj,
            batch_size=batch_size,
            num_batches=num_batches,
            num_action_samples=num_action_samples,
            max_steps=self.epochs + 2,
            actor_lambda=actor_lambda,
            critic_lambda=critic_lambda,
            score_func=PortOpt.score_func,
            sample_actions_gen_func=PortOpt.sample_actions_gen,
            fa_spec=PortOpt.critic_spec(critic_neurons),
            pol_fa_spec=PortOpt.actor_spec(actor_neurons, self.num_risky)
        )


if __name__ == '__main__':
    risky_assets = 1
    num_epochs = 5
    rho = 0.02
    riskfree_return = 0.01
    mu = 0.08
    sigma = 0.03
    epsilon = 1e-4
    gamma = 0.5

    # noinspection PyShadowingNames
    def risky_returns_gen(
        samples: int,
        mu=mu,
        sigma=sigma
    ) -> np.ndarray:
        return np.column_stack((np.random.normal(
            loc=mu,
            scale=sigma,
            size=samples
        ),))

    # noinspection PyShadowingNames
    def util_func(x: float, gamma=gamma) -> float:
        gam = 1. - gamma
        return x ** gam / gam

    # noinspection PyShadowingNames
    def beq_util(x: float, gamma=gamma, epsilon=epsilon) -> float:
        return epsilon ** gamma * util_func(x)

    riskfree_returns = [riskfree_return] * num_epochs
    returns_genf = [risky_returns_gen] * num_epochs
    discount = np.exp(-rho)

    portfolio_optimization = PortOpt(
        num_risky=risky_assets,
        riskless_returns=riskfree_returns,
        returns_gen_funcs=returns_genf,
        cons_util_func=util_func,
        beq_util_func=beq_util,
        discount_factor=discount
    )
