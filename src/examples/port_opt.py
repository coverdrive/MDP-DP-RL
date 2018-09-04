from typing import Tuple, Generic, Sequence, Callable
import numpy as np
from func_approx.dnn_spec import DNNSpec
from algorithms.adp.adp_pg import ADPPolicyGradient
from algorithms.rl_pg.pg import PolicyGradient
from func_approx.func_approx_base import FuncApproxBase
from algorithms.func_approx_spec import FuncApproxSpec
from processes.det_policy import DetPolicy

StateType = Tuple[int, float]
ActionType = Tuple[float, ...]


class PortOpt(Generic[StateType, ActionType]):

    @staticmethod
    def validate_spec(
        num_risky_assets: int,
        riskless_returns_seq: Sequence[float],
        returns_gen: Sequence[Callable[int], np.ndarray],
        disc_fact: float
    ) -> bool:
        b1 = num_risky_assets >= 1
        b2 = all(x > 0 for x in riskless_returns_seq)
        b3 = len(riskless_returns_seq) == len(returns_gen)
        b4 = 0. <= disc_fact <= 1.
        return all([b1, b2, b3, b4])

    def __init__(
        self,
        num_risky: int,
        riskless_returns: Sequence[float],
        returns_gen_funcs: Sequence[Callable[int], np.ndarray],
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
            self.num_risky = num_risky
            self.riskless_returns = riskless_returns
            self.epochs = len(riskless_returns)
            self.returns_gen_funcs = returns_gen_funcs
            self.cons_util_func = cons_util_func
            self.beq_util_func = beq_util_func
        else:
            raise ValueError

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

    def init_state(self) -> StateType:
        return 0, 1.

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
        return 0

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
        return 0


if __name__ == '__main__':
    print(0)
