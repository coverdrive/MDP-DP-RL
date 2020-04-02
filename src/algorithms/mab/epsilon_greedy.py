from typing import Sequence, Callable, Tuple, NoReturn
from processes.mab_env import MabEnv
from algorithms.helper_funcs import get_epsilon_decay_func
from operator import itemgetter
from numpy.random import binomial, randint
from numpy import ndarray, empty
from algorithms.mab.mab_base import MABBase


class EpsilonGreedy(MABBase):

    def __init__(
        self,
        mab: MabEnv,
        time_steps: int,
        num_episodes: int,
        epsilon: float,
        epsilon_half_life: float = 1e8,
        count_init: int = 0,
        mean_init: float = 0.,
    ) -> None:
        super().__init__(
            mab=mab,
            time_steps=time_steps,
            num_episodes=num_episodes
        )
        self.epsilon_func: Callable[[int], float] = get_epsilon_decay_func(
            epsilon,
            epsilon_half_life
        )
        self.count_init: int = count_init
        self.mean_init: float = mean_init

    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        counts: Sequence[int] = [self.count_init] * self.num_arms
        means: Sequence[int] = [self.mean_init] * self.num_arms
        ep_rewards: ndarray = empty(self.time_steps)
        ep_actions: ndarray = empty(self.time_steps, dtype=int)
        for i in range(self.time_steps):
            max_action: int = max(enumerate(means), key=itemgetter(1))[0]
            epsl: float = self.epsilon_func(i)
            action: int = max_action if binomial(1, epsl, size=1)[0] == 0 else\
                randint(self.num_arms, size=1)[0]
            reward: float = self.mab_funcs[action]()
            counts[action] += 1
            means[action] += (reward - means[action]) / counts[action]
            ep_rewards[i] = reward
            ep_actions[i] = action
        return ep_rewards, ep_actions

    def plot_exp_cum_regret_curve(self, best_mean) -> NoReturn:
        import matplotlib.pyplot as plt
        x_vals = range(1, self.time_steps + 1)
        plt.plot(self.get_expected_cum_regret(best_mean), "b", label="Exp Cum Regret")
        plt.xlabel("Time Steps")
        plt.ylabel("Expected Cumulative Regret")
        plt.title("Cumulative Regret Curve")
        plt.xlim(xmin=x_vals[0], xmax=x_vals[-1])
        plt.ylim(ymin=0.0)
        # plt.xticks(x_vals)
        plt.grid(True)
        # plt.legend(loc='upper left')
        plt.show()


if __name__ == '__main__':
    mean_vars_data = [(9., 5.), (10., 2.), (0., 4.)]
    mu_star = max(mean_vars_data, key=itemgetter(0))[0]
    steps = 200
    episodes = 1000
    eps = 0.2
    eps_hl = 50
    ci = 5
    mi = mu_star * 3.

    me = MabEnv.get_gaussian_mab_env(mean_vars_data)
    eg = EpsilonGreedy(
        mab=me,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=eps,
        epsilon_half_life=eps_hl,
        count_init=ci,
        mean_init=mi
    )
    exp_cum_regret = eg.get_expected_cum_regret(mu_star)
    print(exp_cum_regret)

    exp_act_count = eg.get_expected_action_counts()
    print(exp_act_count)

    eg.plot_exp_cum_regret_curve(mu_star)





