import numpy as np


def simulate(reward_probabilities, n_rounds, algorithm, output_all_arm_prob=False):
    n_arms = len(reward_probabilities)
    rewards = []
    selected_arms = []
    best_reward = np.max(reward_probabilities) * np.ones(n_rounds)
    arm_probs = np.zeros(shape=[n_rounds, n_arms])

    for t in range(n_rounds):
        chosen_arm = None
        # In the first n_arms rounds, play each arm once
        if t < n_arms:
            chosen_arm = t

        # After the first n_arms rounds, use the algorithm to select an arm
        if t >= n_arms:
            chosen_arm = algorithm.select_arm()

        # sample a reward from a Bernoulli distribution
        # reward = np.random.binomial(1, reward_probabilities[chosen_arm])
        reward = reward_probabilities[chosen_arm]
        algorithm.update(chosen_arm, reward)

        # record the results
        selected_arms.append(chosen_arm)
        rewards.append(reward)

        # record the probability of each arm
        if output_all_arm_prob:
            arm_probs[t] = algorithm.get_arm_prob()

    arm_probs[-1] = algorithm.get_arm_prob()

    return selected_arms, rewards, best_reward, arm_probs
