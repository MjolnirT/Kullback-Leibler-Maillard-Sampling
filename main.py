import numpy as np
import random
from utility import plot_regret
from BernoulliTSJeffreysPrior import BernoulliTSJeffreysPrior
from KL_MS import KL_MS


def simulate(reward_probabilities, n_rounds, algorithm):
    n_arms = len(reward_probabilities)
    rewards = []
    selected_arms = []
    best_reward = []

    for t in range(n_rounds):
        chosen_arm = None
        if t < n_arms:
            chosen_arm = t

        if t >= n_arms:
            chosen_arm = algorithm.select_arm()

        reward = 1 if random.random() < reward_probabilities[chosen_arm] else 0
        algorithm.update(chosen_arm, reward)

        selected_arms.append(chosen_arm)
        rewards.append(reward)
        best_reward.append(max(reward_probabilities))

    return selected_arms, rewards, best_reward


if __name__ == '__main__':
    reward_probabilities = [0.1, 0.2, 0.3, 0.4, 0.9]
    n_rounds = 10000
    n_arms = len(reward_probabilities)

    algorithm = BernoulliTSJeffreysPrior(n_arms)
    selected_arms, rewards, best_reward = simulate(reward_probabilities, n_rounds, algorithm)
    regret = np.array(best_reward) - np.array(rewards)
    plot_regret(regret, 'Bernoulli Thompson Sampling with Jeffreys Prior')

    # print(f'Selected arms: {selected_arms}')
    # print(f'Rewards: {rewards}')
    # print(f'Regret: {regret}')
    # print(f'Total reward: {sum(rewards)}')


    algorithm = KL_MS(n_arms)
    _, rewards, best_reward = simulate(reward_probabilities, n_rounds, algorithm)
    regret = np.array(best_reward) - np.array(rewards)
    plot_regret(regret, 'KL-MS')

