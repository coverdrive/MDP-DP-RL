from typing import Mapping, Set, Sequence, Tuple, Generic, Callable, Optional
from utils.generic_typevars import S, A
import numpy as np
from random import sample


class AdaptiveMultistageSampling(Generic[S, A]):

    def __init__(
            self,
            start_state: S,
            actions_sets: Sequence[Set[A]],
            num_samples: Sequence[int],
            state_gen_reward_funcs: Sequence[Callable[[S, A], Tuple[Callable[[], S], float]]],
            terminal_opt_val_func: Callable[[S], float],
            discount: float,
    ) -> None:
        if len(actions_sets) == len(num_samples) == len(state_gen_reward_funcs) and \
                0. <= discount <= 1. and \
                all(len(x) <= y for x, y in zip(actions_sets, num_samples)):
            self.start_state = start_state
            self.actions_sets = actions_sets
            self.num_samples = num_samples
            self.num_time_steps = len(actions_sets)
            self.state_gen_rewards_funcs = state_gen_reward_funcs
            self.terminal_opt_val_func = terminal_opt_val_func
            self.discount = discount
        else:
            raise ValueError

    def get_opt_val_and_internals(
            self,
            state: S,
            time_step: int
    ) -> Tuple[float, Optional[Mapping[A, Tuple[float, int]]]]:
        """
        This function estimates the optimal value function V*
        for a given state in a given time step. The output is
        a tuple (pair) where the first element is the estimate
        of the optimal value function V* and the second element
        is a dictionary where the keys are the actions for that
        time step and the values are a pair where the first
        element in the estimated optimal Q-value function Q*
        for that action and the second element is the number of
        samples drawn for the action (that was used in estimating
        the Q-value function Q* for that action)
        """
        if time_step == self.num_time_steps:
            ret = (self.terminal_opt_val_func(state), None)
        else:
            actions = self.actions_sets[time_step]
            state_gen_rewards = {a: self.state_gen_rewards_funcs[time_step](state, a)
                                 for a in actions}
            state_gens = {a: x for a, (x, _) in state_gen_rewards.items()}
            rewards = {a: y for a, (_, y) in state_gen_rewards.items()}
            #  sample each action once, sample each action's next state, and
            #  recursively call the next state's V* estimate
            val_sums = {a: self.get_opt_val_and_internals(state_gens[a](), time_step + 1)[0]
                        for a in actions}
            counts = {a: 1 for a in actions}
            #  loop num_samples[time_step] number of times (beyond the
            #  len(actions) samples that have already been done above
            for i in range(len(actions), self.num_samples[time_step]):
                #  determine the actions that dominate on the UCB Q* estimated value
                #  and pick one of these dominating actions at random, call it a*
                ucb_vals = {a: rewards[a] + self.discount * val_sums[a] / counts[a]
                            + np.sqrt(2 * np.log(i) / counts[a]) for a in actions}
                max_actions = {a for a, u in ucb_vals.items() if u == max(ucb_vals.values())}
                a_star = sample(max_actions, 1)[0]
                #  sample a*'s next state at random, and recursively call the next state's
                #  V* estimate
                next_state = state_gens[a_star]()
                val_sums[a_star] += self.get_opt_val_and_internals(next_state, time_step + 1)[0]
                counts[a_star] += 1

            #  return estimated V* as weighted average of the estimated Q* where weights are
            #  proportioned by the number of times an action was sampled
            ret1 = sum(counts[a] / self.num_samples[time_step] *
                       (rewards[a] + self.discount * val_sums[a] / counts[a])
                       for a in actions)
            ret2 = {a: (rewards[a] + self.discount * val_sums[a] / counts[a], counts[a])
                    for a in actions}
            ret = (ret1, ret2)

        return ret


if __name__ == '__main__':
    from scipy.stats import gamma
    from scipy.integrate import quad
    from utils.gen_utils import memoize

    init_inv: int = 80.0  # initial inventory
    steps: int = 4  # time steps
    step_samples: int = 20
    # the following are (price, gamma distribution mean) pairs, i.e., elasticity
    el: Mapping[float, float] = {10.0: 10.0, 8.0: 20.0, 5.0: 30.0}
    rvs = {p: gamma(l) for p, l in el.items()}
    terminal_vf: Callable[[S], float] = lambda s: 0.
    this_discount: float = 1.0

    # noinspection PyShadowingNames
    @memoize
    def state_gen_rew_func(state: float, action: float, rvs=rvs) -> Tuple[Callable[[], float], float]:
        # noinspection PyShadowingNames
        def rew_f(x: float, state=state, action=action, rvs=rvs) -> float:
            return rvs[action].pdf(x) * (action * min(state, x))

        mu = rvs[action].mean()
        lower = mu - 4.0 * np.sqrt(mu)
        upper = mu + 4.0 * np.sqrt(mu)
        return (
            lambda state=state, action=action, el=el: max(
                0.,
                state - np.random.gamma(el[action], scale=1.0, size=1)[0]
            ),
            quad(rew_f, lower, upper)[0]
        )


    obj = AdaptiveMultistageSampling(
        start_state=init_inv,
        actions_sets=[set(el)] * steps,
        num_samples=[step_samples] * steps,
        state_gen_reward_funcs=[state_gen_rew_func] * steps,
        terminal_opt_val_func=terminal_vf,
        discount=this_discount
    )

    res = obj.get_opt_val_and_internals(init_inv, 0)
    print(res)
