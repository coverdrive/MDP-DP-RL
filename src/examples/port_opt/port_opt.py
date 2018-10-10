from typing import Tuple, Sequence, Callable
import numpy as np
from func_approx.dnn_spec import DNNSpec
from algorithms.adp.adp_pg import ADPPolicyGradient
from algorithms.rl_pg.pg import PolicyGradient
from processes.mdp_rep_for_adp_pg import MDPRepForADPPG
from processes.mdp_rep_for_rl_pg import MDPRepForRLPG
from algorithms.func_approx_spec import FuncApproxSpec
from utils.beta_distribution import BetaDistribution

StateType = Tuple[int, float]
ActionType = Tuple[float, ...]


class PortOpt:

    SMALL_POS = 1e-8

    def __init__(
        self,
        num_risky: int,
        riskless_returns: Sequence[float],
        returns_gen_funcs: Sequence[Callable[[int], np.ndarray]],
        cons_util_func: Callable[[float], float],
        beq_util_func: Callable[[float], float],
        discount_rate: float
    ) -> None:
        if PortOpt.validate_spec(
            num_risky,
            riskless_returns,
            returns_gen_funcs,
            discount_rate
        ):
            self.num_risky: int = num_risky
            self.riskless_returns: Sequence[float] = riskless_returns
            self.epochs: int = len(riskless_returns)
            self.returns_gen_funcs: Sequence[Callable[[int], np.ndarray]]\
                = returns_gen_funcs
            self.cons_util_func: Callable[[float], float] = cons_util_func
            self.beq_util_func: Callable[[float], float] = beq_util_func
            self.discount_rate: float = discount_rate
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
    # U_Beq(W_T) is the utility of bequest, U_Cons(W_t) is the utility of
    # consmption.
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
        :param params: is a Sequence (mu, nu, mu_1, ..., mu_n,
        sigma^2_1,..., sigma^2_n) (of length 2n + 2) where (mu, nu) describes the
        beta distribution for the consumption 0 < a_0 < 1, and (mu_i, sigma^2_i)
        describes the normal distribution for the allocation a_i of risky asset
        i, 1 <= i <= n.
        :return: nabla_{params} [log_e p(a_0, a_1, ..., a_n; mu, nu,
        mu_1, ..., mu_n, sigma^2_1, ..., sigma^2_n)], which is a Sequence of
        length 2n+2
        """
        n = len(action) - 1
        mu, nu = params[:2]
        means = params[2:2+n]
        variances = params[2+n:]
        cons = action[0]
        alloc = action[1:]
        # beta distrubution PDF p(cons; alpha, beta) = gamma(alpha + beta) /
        # (gamma(alpha) * gamma(beta)) * cons^{alpha-1} * (1-cons)^(beta-1)
        # mu = alpha / (alpha + beta) = alpha / nu, nu = alpha + beta
        # alpha = mu * nu, beta = (1-mu) * nu
        # Score_mu(x) = d(log(p(x))) / d(mu) = Score_alpha * d(alpha) / d(mu)
        # + Score_beta * d(beta) / d(mu)
        # = (digamma(beta) - digamma(alpha) + log(x) - log(1 - x)) * nu
        # Score_nu(x) = d(log(p(x))) / d(nu) = Score_alpha * d(alpha) / d(nu)
        # + Score_beta * d(beta) / d(nu)
        # = (digamma(beta) - digamma(alpha) + log(x) - log(1 - x)) * mu +
        # digamma(nu) - digamma(beta) + log(1 - x)
        mu_score, nu_score = BetaDistribution(mu, nu).get_mu_nu_scores(cons)
        # normal distribution PDF p(alloc; mean, var) = 1/sqrt(2 pi var) .
        # e^{-(alloc - mean)^2 / (2 var)}
        # score_{mean} = d/d(mean) (log p) = (alloc - mean) / var
        # score_{var} = d/d(var) (log p) = -0.5 * (1/var - (alloc - mean)^2 / var^2)
        means_score = [(alloc[i] - means[i]) / v for i, v in
                       enumerate(variances)]
        variances_score = [-0.5 * (1. / v - means_score[i] ** 2) for i, v in
                           enumerate(variances)]
        return [mu_score, nu_score] + means_score + variances_score

    @staticmethod
    def sample_actions_gen(
        params: Sequence[float],
        num_samples: int
    ) -> Sequence[ActionType]:
        """
        :param params: is a sequence (mu, nu, mu_1, ..., mu_n, sigma^2_1, ...,
        sigma^2_n (of length 2n+2) where (mu, nu) describes the
        beta distribution for the consumption 0 < a_0 < 1, and (mu_1, sigma^2_i)
        describes the normal distribution for the allocation a_i of risky asset i,
        1 <= i <= n.
        :param num_samples: number of samples
        :return: list (of length num_samples) of (n+1)-tuples (a_0, a_1, ...., a_n)
        """
        n = int(len(params) / 2) - 1
        mu, nu = params[:2]
        means = params[2:2+n]
        variances = params[2+n:]
        cons_samples = BetaDistribution(mu, nu).get_samples(num_samples)
        alloc_samples = [np.random.normal(
            loc=means[i],
            scale=np.sqrt(v),
            size=num_samples
        ) for i, v in enumerate(variances)]
        return [tuple(x) for x in np.vstack([cons_samples] + alloc_samples).T]

    @staticmethod
    def init_state() -> StateType:
        return 0, 1.

    # noinspection PyPep8Naming
    def state_reward_gen(
        self,
        state: StateType,
        action: ActionType,
        num_samples: int
    ) -> Sequence[Tuple[StateType, float]]:
        t, W = state
        cons = action[0]
        risky_alloc = action[1:]
        riskless_alloc = 1. - sum(risky_alloc)
        alloc = np.insert(np.array(risky_alloc), 0, riskless_alloc)
        ret_samples = np.hstack((
            np.full((num_samples, 1), self.riskless_returns[t]),
            self.returns_gen_funcs[t](num_samples)
        ))
        epoch_end_wealth = [W * (1. - cons) * max(PortOpt.SMALL_POS,
                                                  alloc.dot(np.exp(rs)))
                            for rs in ret_samples]
        return [(
            (t + 1, eew),
            self.cons_util_func(W * cons)
            + (np.exp(-self.discount_rate) * self.beq_util_func(eew)
               if t == self.epochs - 1 else 0.)
        ) for eew in epoch_end_wealth]

    def get_adp_pg_obj(
        self,
        reinforce: bool,
        num_state_samples: int,
        num_next_state_samples: int,
        num_action_samples: int,
        num_batches: int,
        actor_lambda: float,
        critic_lambda: float,
        actor_mu_spec: FuncApproxSpec,
        actor_nu_spec: FuncApproxSpec,
        actor_mean_spec: FuncApproxSpec,
        actor_variance_spec: FuncApproxSpec,
        critic_spec: FuncApproxSpec
    ) -> ADPPolicyGradient:
        init_state = PortOpt.init_state()
        mdp_rep_obj = MDPRepForADPPG(
            np.exp(-self.discount_rate),
            lambda n: [init_state] * n,
            lambda s, a, n: self.state_reward_gen(s, a, n),
            lambda s: s[0] == self.epochs - 1
        )
        risky = self.num_risky
        policy_spec = [actor_mu_spec, actor_nu_spec] +\
                      [actor_mean_spec] * risky +\
                      [actor_variance_spec] * risky
        return ADPPolicyGradient(
            mdp_rep_for_adp_pg=mdp_rep_obj,
            reinforce=reinforce,
            num_state_samples=num_state_samples,
            num_next_state_samples=num_next_state_samples,
            num_action_samples=num_action_samples,
            num_batches=num_batches,
            max_steps=self.epochs,
            actor_lambda=actor_lambda,
            critic_lambda=critic_lambda,
            score_func=PortOpt.score_func,
            sample_actions_gen_func=PortOpt.sample_actions_gen,
            vf_fa_spec=critic_spec,
            pol_fa_spec=policy_spec
        )

    def get_pg_obj(
        self,
        reinforce: bool,
        batch_size: int,
        num_batches: int,
        num_action_samples: int,
        actor_lambda: float,
        critic_lambda: float,
        actor_mu_spec: FuncApproxSpec,
        actor_nu_spec: FuncApproxSpec,
        actor_mean_spec: FuncApproxSpec,
        actor_variance_spec: FuncApproxSpec,
        critic_spec: FuncApproxSpec
    ) -> PolicyGradient:
        init_state = PortOpt.init_state()
        mdp_rep_obj = MDPRepForRLPG(
            np.exp(self.discount_rate),
            lambda: init_state,
            lambda s, a: self.state_reward_gen(s, a, 1)[0],
            lambda s: s[0] == self.epochs - 1
        )
        risky = self.num_risky
        policy_spec = [actor_mu_spec, actor_nu_spec] +\
                      [actor_mean_spec] * risky +\
                      [actor_variance_spec] * risky
        return PolicyGradient(
            mdp_rep_for_rl_pg=mdp_rep_obj,
            reinforce=reinforce,
            batch_size=batch_size,
            num_batches=num_batches,
            num_action_samples=num_action_samples,
            max_steps=self.epochs,
            actor_lambda=actor_lambda,
            critic_lambda=critic_lambda,
            score_func=PortOpt.score_func,
            sample_actions_gen_func=PortOpt.sample_actions_gen,
            fa_spec=critic_spec,
            pol_fa_spec=policy_spec
        )

    def test_det_policy(
        self,
        det_pol: Callable[[StateType], ActionType],
        num_paths: int
    ) -> float:
        path_returns = []
        for _ in range(num_paths):
            state = self.init_state()
            path_return = 0.
            for i in range(self.epochs):
                action = det_pol(state)
                state, reward = self.state_reward_gen(state, action, 1)[0]
                path_return += reward * np.exp(-self.discount_rate * i)
            path_returns.append(path_return)
        return sum(path_returns) / len(path_returns)


if __name__ == '__main__':
    risky_assets = 1
    num_epochs = 5
    rho = 0.04
    r = 0.04
    mean = 0.08
    sigma = 0.03
    gamma = 0.2

    optimal_allocation = (mean - r) / (sigma ** 2 * gamma)
    print(optimal_allocation)

    # noinspection PyShadowingNames
    def risky_returns_gen(
        samples: int,
        mean=mean,
        sigma=sigma
    ) -> np.ndarray:
        return np.column_stack((np.random.normal(
            loc=mean,
            scale=sigma,
            size=samples
        ),))

    def util_func(_: float) -> float:
        return 0.

    # noinspection PyShadowingNames
    def beq_util(x: float, gamma=gamma) -> float:
        gam = 1. - gamma
        return x ** gam / gam if gam != 0 else np.log(x)

    riskfree_returns = [r] * num_epochs
    returns_genf = [risky_returns_gen] * num_epochs

    portfolio_optimization = PortOpt(
        num_risky=risky_assets,
        riskless_returns=riskfree_returns,
        returns_gen_funcs=returns_genf,
        cons_util_func=util_func,
        beq_util_func=beq_util,
        discount_rate=rho
    )

    reinforce_val = True
    num_state_samples_val = 500
    num_next_state_samples_val = 30
    num_action_samples_val = 50
    num_batches_val = 3000
    actor_lambda_val = 0.99
    critic_lambda_val = 0.99

    actor_mu = FuncApproxSpec(
        state_feature_funcs=[],
        sa_feature_funcs=[],
        dnn_spec=DNNSpec(
            neurons=[],
            hidden_activation=DNNSpec.log_squish,
            hidden_activation_deriv=DNNSpec.log_squish_deriv,
            output_activation=DNNSpec.sigmoid,
            output_activation_deriv=DNNSpec.sigmoid_deriv
        )
    )
    actor_nu = FuncApproxSpec(
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
    actor_mean = FuncApproxSpec(
        state_feature_funcs=[],
        sa_feature_funcs=[],
        dnn_spec=None
    )
    actor_variance = FuncApproxSpec(
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
    critic = FuncApproxSpec(
        state_feature_funcs=[],
        sa_feature_funcs=[],
        dnn_spec=None
    )

    adp_pg_obj = portfolio_optimization.get_adp_pg_obj(
        reinforce=reinforce_val,
        num_state_samples=num_state_samples_val,
        num_next_state_samples=num_next_state_samples_val,
        num_action_samples=num_action_samples_val,
        num_batches=num_batches_val,
        actor_lambda=actor_lambda_val,
        critic_lambda=critic_lambda_val,
        actor_mu_spec=actor_mu,
        actor_nu_spec=actor_nu,
        actor_mean_spec=actor_mean,
        actor_variance_spec=actor_variance,
        critic_spec=critic
    )
    policy1 = adp_pg_obj.get_optimal_det_policy_func()
    actions1 = [policy1((t, 1.)) for t in range(portfolio_optimization.epochs)]
    consumptions1, risky_allocations1 = zip(*actions1)
    print(consumptions1)
    print(risky_allocations1)

    pg_obj = portfolio_optimization.get_pg_obj(
        reinforce=reinforce_val,
        batch_size=num_state_samples_val,
        num_batches=num_batches_val,
        num_action_samples=num_action_samples_val,
        actor_lambda=actor_lambda_val,
        critic_lambda=critic_lambda_val,
        actor_mu_spec=actor_mu,
        actor_nu_spec=actor_nu,
        actor_mean_spec=actor_mean,
        actor_variance_spec=actor_variance,
        critic_spec=critic
    )
    policy2 = pg_obj.get_optimal_det_policy_func()
    actions2 = [policy2((t, 1.)) for t in range(portfolio_optimization.epochs)]
    consumptions2, risky_allocations2 = zip(*actions2)
    print(consumptions2)
    print(risky_allocations2)
