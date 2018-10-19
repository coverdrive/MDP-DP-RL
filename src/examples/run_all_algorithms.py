from typing import NamedTuple, Mapping, Tuple, TypeVar, Callable
from processes.mdp_refined import MDPRefined
from processes.mdp_rep_for_adp import MDPRepForADP
from processes.mdp_rep_for_rl_tabular import MDPRepForRLTabular
from processes.mdp_rep_for_rl_fa import MDPRepForRLFA
from algorithms.dp.dp_analytic import DPAnalytic
from algorithms.dp.dp_numeric import DPNumeric
from algorithms.adp.adp import ADP
import algorithms.rl_tabular.monte_carlo as mc_tabular
import algorithms.rl_tabular.td0 as td0_tabular
import algorithms.rl_tabular.tdlambda as tdl_tabular
import algorithms.rl_func_approx.monte_carlo as mc_fa
import algorithms.rl_func_approx.td0 as td0_fa
import algorithms.rl_func_approx.tdlambda as tdl_fa
from algorithms.func_approx_spec import FuncApproxSpec
from func_approx.func_approx_base import FuncApproxBase
from func_approx.dnn_spec import DNNSpec
from algorithms.opt_base import OptBase
from algorithms.td_algo_enum import TDAlgorithm
from itertools import groupby
from utils.gen_utils import memoize
import numpy as np
from operator import itemgetter

S = TypeVar('S')
A = TypeVar('A')


class RunAllAlgorithms(NamedTuple):

    mdp_refined: MDPRefined
    tolerance: float
    exploring_start: bool
    first_visit_mc: bool
    num_samples: int
    softmax: bool
    epsilon: float
    epsilon_half_life: float
    learning_rate: float
    learning_rate_decay: float
    lambd: float
    num_episodes: int
    batch_size: int
    max_steps: int
    tdl_fa_offline: bool
    fa_spec: FuncApproxSpec

    @memoize
    def get_mdp_rep_for_adp(self) -> MDPRepForADP:
        return self.mdp_refined.get_mdp_rep_for_adp()

    @memoize
    def get_mdp_rep_for_rl_tabular(self) -> MDPRepForRLTabular:
        return self.mdp_refined.get_mdp_rep_for_rl_tabular()

    @memoize
    def get_mdp_rep_for_rl_fa(self) -> MDPRepForRLFA:
        return self.mdp_refined.get_mdp_rep_for_rl_tabular()

    def get_all_algorithms(self) -> Mapping[str, OptBase]:
        return {
            "DP Analytic": self.get_dp_analytic(),
            "DP Numeric": self.get_dp_numeric(),
            "ADP": self.get_adp(),
            "Tabular Monte Carlo": self.get_tabular_monte_carlo(),
            "Tabular SARSA": self.get_tabular_sarsa(),
            "Tabular QLearning": self.get_tabular_qlearning(),
            "Tabular Expected SARSA": self.get_tabular_expected_sarsa(),
            "Tabular SARSA Lambda": self.get_tabular_sarsa_lambda(),
            "Tabular QLearning Lambda": self.get_tabular_qlearning_lambda(),
            "Tabular Expected SARSA Lambda": self.get_tabular_expected_sarsa_lambda(),
            "FuncApprox Monte Carlo": self.get_fa_monte_carlo(),
            "FuncApprox SARSA": self.get_fa_sarsa(),
            "FuncApprox QLearning": self.get_fa_qlearning(),
            "FuncApprox Expected SARSA": self.get_fa_expected_sarsa(),
            "FuncApprox SARSA Lambda": self.get_fa_sarsa_lambda(),
            "FuncApprox QLearning Lambda": self.get_fa_qlearning_lambda(),
            "FuncApprox Expected SARSA Lambda": self.get_fa_expected_sarsa_lambda()
        }

    def get_all_optimal_policies(self) -> Mapping[str, Callable[[S], A]]:
        return {s: a.get_optimal_det_policy_func() for s, a in
                self.get_all_algorithms().items()}

    def get_all_optimal_vfs(self) -> Mapping[str, Callable[[S], float]]:
        return {s: a.get_optimal_value_func() for s, a in
                self.get_all_algorithms().items()}

    def get_dp_analytic(self) -> DPAnalytic:
        return DPAnalytic(self.mdp_refined, self.tolerance)

    def get_dp_numeric(self) -> DPNumeric:
        return DPNumeric(self.mdp_refined, self.tolerance)

    def get_adp(self) -> ADP:
        return ADP(
            self.get_mdp_rep_for_adp(),
            self.num_samples,
            self.softmax,
            self.epsilon,
            self.epsilon_half_life,
            self.tolerance,
            self.fa_spec
        )

    def get_tabular_monte_carlo(self) -> mc_tabular.MonteCarlo:
        return mc_tabular.MonteCarlo(
            self.get_mdp_rep_for_rl_tabular(),
            self.exploring_start,
            self.first_visit_mc,
            self.softmax,
            self.epsilon,
            self.epsilon_half_life,
            self.num_episodes,
            self.max_steps
        )

    def get_tabular_sarsa(self) -> td0_tabular.TD0:
        return td0_tabular.TD0(
            self.get_mdp_rep_for_rl_tabular(),
            self.exploring_start,
            TDAlgorithm.SARSA,
            self.softmax,
            self.epsilon,
            self.epsilon_half_life,
            self.learning_rate,
            self.learning_rate_decay,
            self.num_episodes,
            self.max_steps
        )

    def get_tabular_qlearning(self) -> td0_tabular.TD0:
        return td0_tabular.TD0(
            self.get_mdp_rep_for_rl_tabular(),
            self.exploring_start,
            TDAlgorithm.QLearning,
            self.softmax,
            self.epsilon,
            self.epsilon_half_life,
            self.learning_rate,
            self.learning_rate_decay,
            self.num_episodes,
            self.max_steps
        )

    def get_tabular_expected_sarsa(self) -> td0_tabular.TD0:
        return td0_tabular.TD0(
            self.get_mdp_rep_for_rl_tabular(),
            self.exploring_start,
            TDAlgorithm.ExpectedSARSA,
            self.softmax,
            self.epsilon,
            self.epsilon_half_life,
            self.learning_rate,
            self.learning_rate_decay,
            self.num_episodes,
            self.max_steps
        )

    def get_tabular_sarsa_lambda(self) -> tdl_tabular.TDLambda:
        return tdl_tabular.TDLambda(
            self.get_mdp_rep_for_rl_tabular(),
            self.exploring_start,
            TDAlgorithm.SARSA,
            self.softmax,
            self.epsilon,
            self.epsilon_half_life,
            self.learning_rate,
            self.learning_rate_decay,
            self.lambd,
            self.num_episodes,
            self.max_steps
        )

    def get_tabular_qlearning_lambda(self) -> tdl_tabular.TDLambda:
        return tdl_tabular.TDLambda(
            self.get_mdp_rep_for_rl_tabular(),
            self.exploring_start,
            TDAlgorithm.QLearning,
            self.softmax,
            self.epsilon,
            self.epsilon_half_life,
            self.learning_rate,
            self.learning_rate_decay,
            self.lambd,
            self.num_episodes,
            self.max_steps
        )

    def get_tabular_expected_sarsa_lambda(self) -> tdl_tabular.TDLambda:
        return tdl_tabular.TDLambda(
            self.get_mdp_rep_for_rl_tabular(),
            self.exploring_start,
            TDAlgorithm.ExpectedSARSA,
            self.softmax,
            self.epsilon,
            self.epsilon_half_life,
            self.learning_rate,
            self.learning_rate_decay,
            self.lambd,
            self.num_episodes,
            self.max_steps
        )

    def get_fa_monte_carlo(self) -> mc_fa.MonteCarlo:
        return mc_fa.MonteCarlo(
            self.get_mdp_rep_for_rl_fa(),
            self.exploring_start,
            self.softmax,
            self.epsilon,
            self.epsilon_half_life,
            self.num_episodes,
            self.max_steps,
            self.fa_spec
        )

    def get_fa_sarsa(self) -> td0_fa.TD0:
        return td0_fa.TD0(
            self.get_mdp_rep_for_rl_fa(),
            self.exploring_start,
            TDAlgorithm.SARSA,
            self.softmax,
            self.epsilon,
            self.epsilon_half_life,
            self.num_episodes,
            self.max_steps,
            self.fa_spec
        )

    def get_fa_qlearning(self) -> td0_fa.TD0:
        return td0_fa.TD0(
            self.get_mdp_rep_for_rl_fa(),
            self.exploring_start,
            TDAlgorithm.QLearning,
            self.softmax,
            self.epsilon,
            self.epsilon_half_life,
            self.num_episodes,
            self.max_steps,
            self.fa_spec
        )

    def get_fa_expected_sarsa(self) -> td0_fa.TD0:
        return td0_fa.TD0(
            self.get_mdp_rep_for_rl_fa(),
            self.exploring_start,
            TDAlgorithm.ExpectedSARSA,
            self.softmax,
            self.epsilon,
            self.epsilon_half_life,
            self.num_episodes,
            self.max_steps,
            self.fa_spec
        )

    def get_fa_sarsa_lambda(self) -> tdl_fa.TDLambda:
        return tdl_fa.TDLambda(
            self.get_mdp_rep_for_rl_fa(),
            self.exploring_start,
            TDAlgorithm.SARSA,
            self.softmax,
            self.epsilon,
            self.epsilon_half_life,
            self.lambd,
            self.num_episodes,
            self.batch_size,
            self.max_steps,
            self.fa_spec,
            self.tdl_fa_offline
        )

    def get_fa_qlearning_lambda(self) -> tdl_fa.TDLambda:
        return tdl_fa.TDLambda(
            self.get_mdp_rep_for_rl_fa(),
            self.exploring_start,
            TDAlgorithm.QLearning,
            self.softmax,
            self.epsilon,
            self.epsilon_half_life,
            self.lambd,
            self.num_episodes,
            self.batch_size,
            self.max_steps,
            self.fa_spec,
            self.tdl_fa_offline
        )

    def get_fa_expected_sarsa_lambda(self) -> tdl_fa.TDLambda:
        return tdl_fa.TDLambda(
            self.get_mdp_rep_for_rl_fa(),
            self.exploring_start,
            TDAlgorithm.ExpectedSARSA,
            self.softmax,
            self.epsilon,
            self.epsilon_half_life,
            self.lambd,
            self.num_episodes,
            self.batch_size,
            self.max_steps,
            self.fa_spec,
            self.tdl_fa_offline
        )


if __name__ == '__main__':

    from examples.inv_control import InvControl

    ic = InvControl(
        demand_lambda=0.5,
        lead_time=1,
        stockout_cost=49.,
        fixed_order_cost=0.0,
        epoch_disc_factor=0.98,
        order_limit=7,
        space_limit=8,
        throwout_cost=30.,
        stockout_limit=5,
        stockout_limit_excess_cost=30.
    )
    valid = ic.validate_spec()
    mdp_ref_obj = ic.get_mdp_refined()
    this_tolerance = 1e-3
    exploring_start = False
    this_first_visit_mc = True
    num_samples = 30
    this_softmax = True
    this_epsilon = 0.05
    this_epsilon_half_life = 30
    this_learning_rate = 0.1
    this_learning_rate_decay = 1e6
    this_lambd = 0.8
    this_num_episodes = 3000
    this_batch_size = 10
    this_max_steps = 1000
    this_tdl_fa_offline = True
    state_ffs = FuncApproxBase.get_identity_feature_funcs(ic.lead_time + 1)
    sa_ffs = [(lambda x, f=f: f(x[0])) for f in state_ffs] + [lambda x: x[1]]
    this_fa_spec = FuncApproxSpec(
        state_feature_funcs=state_ffs,
        sa_feature_funcs=sa_ffs,
        dnn_spec=DNNSpec(
            neurons=[2, 4],
            hidden_activation=DNNSpec.relu,
            hidden_activation_deriv=DNNSpec.relu_deriv,
            output_activation=DNNSpec.identity,
            output_activation_deriv=DNNSpec.identity_deriv
        )
    )

    raa = RunAllAlgorithms(
        mdp_refined=mdp_ref_obj,
        tolerance=this_tolerance,
        exploring_start=exploring_start,
        first_visit_mc=this_first_visit_mc,
        num_samples=num_samples,
        softmax=this_softmax,
        epsilon=this_epsilon,
        epsilon_half_life=this_epsilon_half_life,
        learning_rate=this_learning_rate,
        learning_rate_decay=this_learning_rate_decay,
        lambd=this_lambd,
        num_episodes=this_num_episodes,
        batch_size=this_batch_size,
        max_steps=this_max_steps,
        tdl_fa_offline=this_tdl_fa_offline,
        fa_spec=this_fa_spec
    )

    def crit(x: Tuple[Tuple[int, ...], int]) -> int:
        return sum(x[0])

    for st, mo in raa.get_all_algorithms().items():
        print("Starting %s" % st)
        opt_pol_func = mo.get_optimal_det_policy_func()
        opt_pol = {s: opt_pol_func(s) for s in mdp_ref_obj.all_states}
        print(sorted(
            [(ip, np.mean([float(y) for _, y in v])) for ip, v in
             groupby(sorted(opt_pol.items(), key=crit), key=crit)],
            key=itemgetter(0)
        ))
