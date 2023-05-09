import numpy as np


class Base:
    def __init__(self, n_arms, n_rounds, explore_weight=1):
        self.T = n_rounds
        self.n_arms = n_arms
        self.t = 0
        self.explore_weight = explore_weight

        # initialize the records
        self.rewards = np.zeros(shape=[n_arms, n_rounds])
        self.N_arms = np.zeros(shape=n_arms).astype(int)
        self.means = np.zeros(shape=n_arms)

        # initialize the probability of each arm as the uniform distribution
        self.prob_arm = None

    def select_arm(self):
        return

    def update(self, chosen_arm, reward):
        self.rewards[chosen_arm, self.t] = reward
        self.N_arms[chosen_arm] += 1
        self.means[chosen_arm] = np.mean(self.rewards[chosen_arm, :self.N_arms[chosen_arm]])
        self.t += 1
    def update_arm_prob(self):
        return

