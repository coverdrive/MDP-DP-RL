from typing import NoReturn
from operator import itemgetter
from processes.mab_env import MabEnv
from algorithms.mab.epsilon_greedy import EpsilonGreedy
from algorithms.mab.ucb1 import UCB1
from algorithms.mab.ts_gaussian import ThompsonSamplingGaussian
from algorithms.mab.ts_bernoulli import ThompsonSamplingBernoulli
from numpy import arange
import matplotlib.pyplot as plt


def plot_gaussian_algorithms() -> NoReturn:
    mean_vars_data = [(0., 4.), (6., 10.), (2., 20.), (4., 1.), (9., 5.), (10., 2.)]
    mu_star = max(mean_vars_data, key=itemgetter(0))[0]
    steps = 500
    episodes = 500
    eps = 0.3
    eps_hl = 400
    ci = 5
    mi = mu_star * 3.

    ts_mi = 0.
    ts_si = 10.

    me = MabEnv.get_gaussian_mab_env(mean_vars_data)

    greedy_opt_init = EpsilonGreedy(
        mab=me,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=0.,
        epsilon_half_life=1e8,
        count_init=ci,
        mean_init=mi
    )
    eps_greedy = EpsilonGreedy(
        mab=me,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=eps,
        epsilon_half_life=1e8,
        count_init=0,
        mean_init=0.
    )
    decay_eps_greedy = EpsilonGreedy(
        mab=me,
        time_steps=steps,
        num_episodes=episodes,
        epsilon=eps,
        epsilon_half_life=eps_hl,
        count_init=0,
        mean_init=0.
    )
    ts = ThompsonSamplingGaussian(
        mab=me,
        time_steps=steps,
        num_episodes=episodes,
        init_mean=ts_mi,
        init_stdev=ts_si
    )

    plot_colors = ['r', 'b', 'g', 'k']
    labels = [
        'Greedy, Optimistic Initialization',
        '$\epsilon$-Greedy',
        'Decaying $\epsilon$-Greedy',
        'Thompson Sampling'
    ]

    exp_cum_regrets = [
        greedy_opt_init.get_expected_cum_regret(mu_star),
        eps_greedy.get_expected_cum_regret(mu_star),
        decay_eps_greedy.get_expected_cum_regret(mu_star),
        ts.get_expected_cum_regret(mu_star)
    ]

    x_vals = range(1, steps + 1)
    for i in range(len(exp_cum_regrets)):
        plt.plot(exp_cum_regrets[i], color=plot_colors[i], label=labels[i])
    plt.xlabel("Time Steps")
    plt.ylabel("Expected Cumulative Regret")
    plt.title("Cumulative Regret Curves")
    plt.xlim(xmin=x_vals[0], xmax=x_vals[-1])
    plt.ylim(ymin=0.0)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()

    exp_act_counts = [
        greedy_opt_init.get_expected_action_counts(),
        eps_greedy.get_expected_action_counts(),
        decay_eps_greedy.get_expected_action_counts(),
        ts.get_expected_action_counts()
    ]
    index = arange(len(me.arms_sampling_funcs))
    spacing = 0.3
    width = (1 - spacing) / len(exp_act_counts)

    for i in range(len(exp_act_counts)):
        plt.bar(
            index - (1 - spacing) / 2 + i * width,
            exp_act_counts[i],
            width,
            color=plot_colors[i],
            label=labels[i]
        )
    plt.xlabel("Arms")
    plt.ylabel("Expected Counts of Arms")
    plt.title("Arms Counts Plot")
    plt.xticks(
        index - 0.3,
        ["$\mu$=%.1f,$\sigma$=%.1f" % (m, s) for m, s in mean_vars_data]
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_gaussian_algorithms()
