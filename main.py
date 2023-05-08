import numpy as np
from utility import plot_regret, plot_regrets
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

        # reward = 1 if random.random() < reward_probabilities[chosen_arm] else 0
        reward = reward_probabilities[chosen_arm]
        algorithm.update(chosen_arm, reward)

        selected_arms.append(chosen_arm)
        rewards.append(reward)
        best_reward.append(max(reward_probabilities))
        # best_reward.append(1 if random.random() < max(reward_probabilities) else 0)

    return selected_arms, rewards, best_reward


if __name__ == '__main__':
    # reward_probabilities = [0.1]*5 + [0.5]*10 + [0.8]*10 + [0.97]*10 + [0.98]*5 +[0.999]
    reward_probabilities = [0.1]*5 + [0.5]*10 + [0.6]
    print(f'reward_probabilities: {reward_probabilities}')
    n_rounds = 1000
    n_arms = len(reward_probabilities)

    algorithm = BernoulliTSJeffreysPrior(n_arms)
    selected_arms, rewards, best_reward = simulate(reward_probabilities, n_rounds, algorithm)
    TS_regret = np.array(best_reward) - np.array(rewards)
    # print(f'selected_arms: {selected_arms}')
    # plot_regret(TS_regret, 'Bernoulli Thompson Sampling with Jeffreys Prior')

    # print(f'Selected arms: {selected_arms}')
    # print(f'Rewards: {rewards}')
    # print(f'Regret: {regret}')
    # print(f'Total reward: {sum(rewards)}')


    algorithm = KL_MS(n_arms)
    _, rewards, best_reward = simulate(reward_probabilities, n_rounds, algorithm)
    MS_regret = np.array(best_reward) - np.array(rewards)
    # print(f'selected_arms: {selected_arms}')
    # plot_regret(MS_regret, 'KL-MS')

    plot_regrets([TS_regret, MS_regret], 'Bernoulli Thompson Sampling with Jeffreys Prior and KL-MS')
