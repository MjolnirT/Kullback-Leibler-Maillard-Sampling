import numpy as np
from KL_MS_JeffreysPrior import KL_MS_JeffreysPrior
from utility import plot_regrets
from BernoulliTSJeffreysPrior import BernoulliTS
from KL_MS import KL_MS


def simulate(reward_probabilities, n_rounds, algorithm):
    n_arms = len(reward_probabilities)
    rewards = []
    selected_arms = []
    best_reward = np.max(reward_probabilities) * np.ones(n_rounds)

    for t in range(n_rounds):
        chosen_arm = None
        # In the first n_arms rounds, play each arm once
        if t < n_arms:
            chosen_arm = t

        # After the first n_arms rounds, use the algorithm to select an arm
        if t >= n_arms:
            chosen_arm = algorithm.select_arm()

        # receive reward from the chosen arm, use the expected reward as the reward
        reward = reward_probabilities[chosen_arm]
        algorithm.update(chosen_arm, reward)

        # record the results
        selected_arms.append(chosen_arm)
        rewards.append(reward)

    return selected_arms, rewards, best_reward


if __name__ == '__main__':
    reward_probabilities = [0.1] * 5 + [0.5] * 10 + [0.8] * 10 + [0.97] * 10 + [0.98] * 5 + [0.999]
    # reward_probabilities = [0.1]*5 + [0.5]*10 + [0.6]
    print(f'reward_probabilities: {reward_probabilities}')
    n_rounds = 1000
    n_arms = len(reward_probabilities)

    algorithm = BernoulliTS(n_arms, n_rounds)
    _, rewards, best_reward = simulate(reward_probabilities, n_rounds, algorithm)
    B_TS_regret = np.array(best_reward) - np.array(rewards)

    algorithm = KL_MS(n_arms, explore_weight=1.0)
    _, rewards, best_reward = simulate(reward_probabilities, n_rounds, algorithm)
    MS_regret = np.array(best_reward) - np.array(rewards)
    prob_arm = [f"{num:.4f}" for num in algorithm.prob_arm]
    print(f"Probability of arms: {prob_arm}")

    algorithm = KL_MS_JeffreysPrior(n_arms, explore_weight=1.0)
    _, rewards, best_reward = simulate(reward_probabilities, n_rounds, algorithm)
    MS_Jeff_regret = np.array(best_reward) - np.array(rewards)
    prob_arm = [f"{num:.4f}" for num in algorithm.prob_arm]
    print(f"Probability of arms: {prob_arm}")

    label = ['Bernoulli TS' ,'KL-MS', 'KL-MS with Jeffreys Prior']
    plot_regrets([B_TS_regret, MS_regret, MS_Jeff_regret], 'Regret Comparison', label)
