from typing import Sequence, Tuple
from processes.mab_env import MabEnv
from operator import itemgetter
from numpy import ndarray, empty, sqrt, log
from algorithms.mab.mab_base import MABBase


class UCB1(MABBase):

    def __init__(
        self,
        mab: MabEnv,
        time_steps: int,
        num_episodes: int,
        alpha: float
    ) -> None:
        super().__init__(
            mab=mab,
            time_steps=time_steps,
            num_episodes=num_episodes
        )
        self.alpha = alpha

    def get_episode_rewards_actions(self) -> Tuple[ndarray, ndarray]:
        ep_rewards: ndarray = empty(self.time_steps)
        ep_actions: ndarray = empty(self.time_steps, dtype=int)
        for i in range(self.num_arms):
            ep_rewards[i] = self.mab_funcs[i]()
            ep_actions[i] = i
        counts: Sequence[int] = [1] * self.num_arms
        means: Sequence[int] = [ep_rewards[j] for j in range(self.num_arms)]
        for i in range(self.num_arms, self.time_steps):
            ucbs: Sequence[float] = [means[j] +
                                     sqrt(0.5 * self.alpha * log(i) / counts[j])
                                     for j in range(self.num_arms)]
            action: int = max(enumerate(ucbs), key=itemgetter(1))[0]
            reward: float = self.mab_funcs[action]()
            counts[action] += 1
            means[action] += (reward - means[action]) / counts[action]
            ep_rewards[i] = reward
            ep_actions[i] = action
        return ep_rewards, ep_actions



if __name__ == '__main__':
    mean_vars_data = [(9., 5.), (10., 2.), (0., 4.), (6., 10.), (2., 20.), (4., 1.)]
    mu_star = max(mean_vars_data, key=itemgetter(0))[0]
    steps = 200
    episodes = 1000
    this_alpha = 4.0

    me = MabEnv.get_gaussian_mab_env(mean_vars_data)
    ucb1 = UCB1(
        mab=me,
        time_steps=steps,
        num_episodes=episodes,
        alpha=this_alpha
    )
    exp_cum_regret = ucb1.get_expected_cum_regret(mu_star)
    print(exp_cum_regret)

    exp_act_count = ucb1.get_expected_action_counts()
    print(exp_act_count)

    ucb1.plot_exp_cum_regret_curve(mu_star)



