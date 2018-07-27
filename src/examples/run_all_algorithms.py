from typing import NamedTuple, Mapping, Tuple, TypeVar
from processes.mdp_refined import MDPRefined
from processes.det_policy import DetPolicy
from algorithms.opt_planning_anal import OptPlanningAnal
from algorithms.opt_planning_num import OptPlanningNum
from algorithms.opt_learning_mc import OptLearningMC
from algorithms.opt_learning_sarsa import OptLearningSARSA
from algorithms.opt_learning_qlearning import OptLearningQLearning
from algorithms.opt_learning_expsarsa import OptLearningExpSARSA
from algorithms.opt_learning_sarsa_lambda import OptLearningSARSALambda
from algorithms.opt_learning_qlearning_lambda import OptLearningQLearningLambda
from algorithms.opt_learning_expsarsa_lambda import OptLearningExpSARSALambda
from algorithms.opt_base import OptBase
from itertools import groupby
import numpy as np
from operator import itemgetter

S = TypeVar('S')


class RunAllAlgorithms(NamedTuple):

    mdp_refined: MDPRefined
    tolerance: float
    first_visit_mc: bool
    softmax: bool
    epsilon: float
    alpha: float
    lambd: float
    num_episodes: int
    max_steps: int

    def get_all_algorithms(self) -> Mapping[str, OptBase]:
        return {
            "Planning Anal": self.get_planning_anal(),
            "Planning Num": self.get_planning_num(),
            "Learning MC": self.get_learning_mc(),
            "Learning SARSA": self.get_learning_sarsa(),
            "Learning QLearning": self.get_learning_qlearning(),
            "Learning ExpSARSA": self.get_learning_expsarsa(),
            "Learning SARSA Lambda": self.get_learning_sarsa_lambda(),
            "Learning QLearning Lambda": self.get_learning_qlearning_lambda(),
            "Learning ExpSARSA Lambda": self.get_learning_expsarsa_lambda()
        }

    def get_all_optimal_policies(self) -> Mapping[str, DetPolicy]:
        return {s: a.get_optimal_det_policy() for s, a in
                self.get_all_algorithms().items()}

    def get_all_optimal_vf_dicts(self) -> Mapping[str, Mapping[S, float]]:
        return {s: a.get_value_func_dict(a.get_optimal_det_policy())
                for s, a in self.get_all_algorithms().items()}

    def get_planning_anal(self) -> OptPlanningAnal:
        return OptPlanningAnal(self.mdp_refined, self.tolerance)

    def get_planning_num(self) -> OptPlanningNum:
        return OptPlanningNum(self.mdp_refined, self.tolerance)

    def get_learning_mc(self) -> OptLearningMC:
        return OptLearningMC(
            self.mdp_refined,
            self.first_visit_mc,
            self.softmax,
            self.epsilon,
            self.num_episodes,
            self.max_steps
        )

    def get_learning_sarsa(self) -> OptLearningSARSA:
        return OptLearningSARSA(
            self.mdp_refined,
            self.softmax,
            self.epsilon,
            self.alpha,
            self.num_episodes,
            self.max_steps
        )

    def get_learning_qlearning(self) -> OptLearningQLearning:
        return OptLearningQLearning(
            self.mdp_refined,
            self.softmax,
            self.epsilon,
            self.alpha,
            self.num_episodes,
            self.max_steps
        )

    def get_learning_expsarsa(self) -> OptLearningExpSARSA:
        return OptLearningExpSARSA(
            self.mdp_refined,
            self.softmax,
            self.epsilon,
            self.alpha,
            self.num_episodes,
            self.max_steps
        )

    def get_learning_sarsa_lambda(self) -> OptLearningSARSALambda:
        return OptLearningSARSALambda(
            self.mdp_refined,
            self.softmax,
            self.epsilon,
            self.alpha,
            self.lambd,
            self.num_episodes,
            self.max_steps
        )

    def get_learning_qlearning_lambda(self) -> OptLearningQLearningLambda:
        return OptLearningQLearningLambda(
            self.mdp_refined,
            self.softmax,
            self.epsilon,
            self.alpha,
            self.lambd,
            self.num_episodes,
            self.max_steps
        )

    def get_learning_expsarsa_lambda(self) -> OptLearningExpSARSALambda:
        return OptLearningExpSARSALambda(
            self.mdp_refined,
            self.softmax,
            self.epsilon,
            self.alpha,
            self.lambd,
            self.num_episodes,
            self.max_steps
        )


if __name__ == '__main__':

    from examples.inv_control import InvControl

    ic = InvControl(
        demand_lambda=1.2,
        lead_time=0,
        stockout_cost=49.,
        fixed_order_cost=0.0,
        epoch_disc_factor=0.98,
        order_limit=5,
        space_limit=7,
        throwout_cost=30.,
        stockout_limit=5,
        stockout_limit_excess_cost=30.
    )
    valid = ic.validate_spec()
    mdp_ref_obj = ic.get_mdp_refined()
    this_tolerance = 1e-4
    this_first_visit_mc = True
    this_softmax = False
    this_epsilon = 0.1
    this_alpha = 0.1
    this_lambd = 0.7
    this_num_episodes = 5000
    this_max_steps = 1000

    raa = RunAllAlgorithms(
        mdp_refined=mdp_ref_obj,
        tolerance=this_tolerance,
        first_visit_mc=this_first_visit_mc,
        softmax=this_softmax,
        epsilon=this_epsilon,
        alpha=this_alpha,
        lambd=this_lambd,
        num_episodes=this_num_episodes,
        max_steps=this_max_steps
    )

    def crit(x: Tuple[Tuple[int, ...], int]) -> int:
        return sum(x[0])

    for st, mo in raa.get_all_algorithms().items():
        print("Starting %s" % st)
        opt_pol = mo.get_optimal_det_policy().get_state_to_action_map().items()
        print(sorted(
            [(ip, np.mean([float(y) for _, y in v])) for ip, v in
             groupby(sorted(opt_pol, key=crit), key=crit)],
            key=itemgetter(0)
        ))
