from typing import Mapping, Optional, Sequence, Callable, Tuple
from algorithms.rl_func_approx.rl_func_approx_base import RLFuncApproxBase
from algorithms.func_approx_spec import FuncApproxSpec
import numpy as np
from processes.mdp_rep_for_rl_fa import MDPRepForRLFA
from processes.mp_funcs import get_rv_gen_func_single
from algorithms.helper_funcs import get_soft_policy_func_from_qf
from operator import itemgetter
from utils.generic_typevars import S, A
from utils.standard_typevars import VFType, QFType, PolicyActDictType


class LSPI(RLFuncApproxBase):

    def __init__(
        self,
        mdp_rep_for_rl: MDPRepForRLFA,
        exploring_start: bool,
        softmax: bool,
        epsilon: float,
        epsilon_half_life: float,
        num_episodes: int,
        batch_size: int,
        max_steps: int,
        state_feature_funcs: Sequence[Callable[[S], float]],
        sa_feature_funcs: Sequence[Callable[[Tuple[S, A]], float]]
    ) -> None:

        super().__init__(
            mdp_rep_for_rl=mdp_rep_for_rl,
            exploring_start=exploring_start,
            softmax=softmax,
            epsilon=epsilon,
            epsilon_half_life=epsilon_half_life,
            num_episodes=num_episodes,
            max_steps=max_steps,
            fa_spec=FuncApproxSpec(
                state_feature_funcs=state_feature_funcs,
                sa_feature_funcs=sa_feature_funcs,
                dnn_spec=None,
                reglr_coeff=0.,
                learning_rate=0.,
                adam_params=(False, 0., 0.),
                add_unit_feature=True
            )
        )
        self.batch_size: int = batch_size

    def get_value_func_fa(self, polf: PolicyActDictType) -> VFType:
        ffs = self.vf_fa.feature_funcs
        features = len(ffs)
        a_mat = np.zeros((features, features))
        b_vec = np.zeros(features)

        for _ in range(self.num_episodes):
            state = self.mdp_rep.init_state_gen()
            steps = 0
            terminate = False

            while not terminate:
                action = get_rv_gen_func_single(polf(state))()
                next_state, reward = \
                    self.mdp_rep.state_reward_gen_func(state, action)
                phi_s = np.array([f(state) for f in ffs])
                phi_sp = np.array([f(next_state) for f in ffs])
                a_mat += np.outer(
                    phi_s,
                    phi_s - self.mdp_rep.gamma * phi_sp
                )
                b_vec += reward * phi_s
                steps += 1
                terminate = steps >= self.max_steps or \
                    self.mdp_rep.terminal_state_func(state)
                state = next_state

        self.vf_fa.params = [np.linalg.inv(a_mat).dot(b_vec)]

        return self.vf_fa.get_func_eval

    # noinspection PyShadowingNames
    def get_qv_func_fa(self, polf: Optional[PolicyActDictType]) -> QFType:
        ffs = self.qvf_fa.feature_funcs
        features = len(ffs)
        a_mat = np.zeros((features, features))
        b_vec = np.zeros(features)
        control = polf is None
        this_polf = polf if polf is not None else self.get_init_policy_func()

        for episode in range(self.num_episodes):
            if self.exploring_start:
                state, action = self.mdp_rep.init_state_action_gen()
            else:
                state = self.mdp_rep.init_state_gen()
                action = get_rv_gen_func_single(this_polf(state))()

            # print((episodes, max(self.qvf_fa.get_func_eval((state, a)) for a in
            #        self.mdp_rep.state_action_func(state))))
            # print(self.qvf_fa.params)

            steps = 0
            terminate = False

            while not terminate:
                next_state, reward = \
                    self.mdp_rep.state_reward_gen_func(state, action)
                phi_s = np.array([f((state, action)) for f in ffs])
                next_action = get_rv_gen_func_single(this_polf(next_state))()
                if control:
                    next_act = max(
                        [(a, self.qvf_fa.get_func_eval((next_state, a))) for a in
                         self.state_action_func(next_state)],
                        key=itemgetter(1)
                    )[0]
                else:
                    next_act = next_action
                phi_sp = np.array([f((next_state, next_act)) for f in ffs])
                a_mat += np.outer(
                    phi_s,
                    phi_s - self.mdp_rep.gamma * phi_sp
                )
                b_vec += reward * phi_s

                steps += 1
                terminate = steps >= self.max_steps or \
                    self.mdp_rep.terminal_state_func(state)
                state = next_state
                action = next_action

            if control and (episode + 1) % self.batch_size == 0:
                self.qvf_fa.params = [np.linalg.inv(a_mat).dot(b_vec)]
                # print(self.qvf_fa.params)
                this_polf = get_soft_policy_func_from_qf(
                    self.qvf_fa.get_func_eval,
                    self.state_action_func,
                    self.softmax,
                    self.epsilon_func(episode)
                )
                a_mat = np.zeros((features, features))
                b_vec = np.zeros(features)

        if not control:
            self.qvf_fa.params = [np.linalg.inv(a_mat).dot(b_vec)]

        return lambda st: lambda act, st=st: self.qvf_fa.get_func_eval((st, act))


if __name__ == '__main__':
    from processes.mdp_refined import MDPRefined
    mdp_refined_data = {
        1: {
            'a': {1: (0.3, 9.2), 2: (0.6, 4.5), 3: (0.1, 5.0)},
            'b': {2: (0.3, -0.5), 3: (0.7, 2.6)},
            'c': {1: (0.2, 4.8), 2: (0.4, -4.9), 3: (0.4, 0.0)}
        },
        2: {
            'a': {1: (0.3, 9.8), 2: (0.6, 6.7), 3: (0.1, 1.8)},
            'c': {1: (0.2, 4.8), 2: (0.4, 9.2), 3: (0.4, -8.2)}
        },
        3: {
            'a': {3: (1.0, 0.0)},
            'b': {3: (1.0, 0.0)}
        }
    }
    gamma_val = 0.9
    mdp_ref_obj1 = MDPRefined(mdp_refined_data, gamma_val)
    mdp_rep_obj = mdp_ref_obj1.get_mdp_rep_for_rl_tabular()

    exploring_start_val = False
    softmax_flag = False
    epsilon_val = 0.1
    epsilon_half_life_val = 10000
    num_episodes_val = 100000
    batch_size_val = 1000
    max_steps_val = 1000
    state_ff = [lambda s: float(s)]
    sa_ff = [
        lambda x: float(x[0]),
        lambda x: 1. if x[1] == 'a' else 0.,
        lambda x: 1. if x[1] == 'b' else 0.,
        lambda x: 1. if x[1] == 'c' else 0.,
    ]
    lspi_obj = LSPI(
        mdp_rep_obj,
        exploring_start_val,
        softmax_flag,
        epsilon_val,
        epsilon_half_life_val,
        num_episodes_val,
        batch_size_val,
        max_steps_val,
        state_ff,
        sa_ff
    )

    def policy_func(i: int) -> Mapping[str, float]:
        if i == 1:
            ret = {'a': 0.4, 'b': 0.6}
        elif i == 2:
            ret = {'a': 0.7, 'c': 0.3}
        elif i == 3:
            ret = {'b': 1.0}
        else:
            raise ValueError
        return ret

    # this_qf = lspi_obj.get_qv_func_fa(policy_func)
    this_vf = lspi_obj.get_value_func_fa(policy_func)
    print(this_vf(1))
    print(this_vf(2))
    print(this_vf(3))

    opt_det_polf = lspi_obj.get_optimal_det_policy_func()

    # noinspection PyShadowingNames
    def opt_polf(s: S, opt_det_polf=opt_det_polf) -> Mapping[A, float]:
        return {opt_det_polf(s): 1.0}

    opt_vf = lspi_obj.get_value_func_fa(opt_polf)
    print(opt_polf(1))
    print(opt_polf(2))
    print(opt_polf(3))
    print(opt_vf(1))
    print(opt_vf(2))
    print(opt_vf(3))
