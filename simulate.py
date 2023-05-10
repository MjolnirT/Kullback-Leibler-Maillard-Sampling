import numpy as np


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
