import numpy as np
import random


class BernoulliTSJeffreysPrior:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.prior_alpha = 0.5
        self.prior_beta = 0.5
        self.param = np.random.beta(size=n_arms, a=0.5, b=0.5)
        self.alpha = np.full(shape=n_arms, fill_value=0.5)
        self.beta = np.full(shape=n_arms, fill_value=0.5)
        self.assign_prob = np.full(shape=n_arms, fill_value=1 / n_arms)
        self.S = np.zeros(shape=n_arms)
        self.N_arms = np.zeros(shape=n_arms)
        self.t = 0

    def select_arm(self):
        theta_samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n_arms)]
        chosen_arm = np.argmax(theta_samples)
        return chosen_arm

    def update(self, chosen_arm, reward):
        self.t += 1
        self.S[chosen_arm] += reward
        self.N_arms[chosen_arm] += 1
        self.alpha[chosen_arm] = self.prior_alpha + self.S[chosen_arm]
        self.beta[chosen_arm] = self.prior_beta + self.N_arms[chosen_arm] - self.S[chosen_arm]


def simulate(reward_probabilities, n_rounds, algorithm):
    n_arms = len(reward_probabilities)
    rewards = []
    selected_arms = []
    best_reward = max(reward_probabilities) * n_rounds

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

    return selected_arms, rewards, best_reward


if __name__ == '__main__':
    reward_probabilities = [0.1, 0.2, 0.3, 0.4, 0.5]
    n_rounds = 1000
    n_arms = len(reward_probabilities)

    algorithm = BernoulliTSJeffreysPrior(n_arms)
    selected_arms, rewards, best_reward = simulate(reward_probabilities, n_rounds, algorithm)
    regret = best_reward - sum(rewards)

    print(f'Selected arms: {selected_arms}')
    print(f'Rewards: {rewards}')
    print(f'Total reward: {sum(rewards)}')
    print(f'Regret: {regret}')
