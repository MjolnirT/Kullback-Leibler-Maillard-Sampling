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

        self.name = None

    def select_arm(self):
        return

    def update(self, chosen_arm, reward):
        self.rewards[chosen_arm, self.N_arms[chosen_arm]] = reward
        self.N_arms[chosen_arm] += 1
        self.means[chosen_arm] = self.rewards[chosen_arm].sum() / self.N_arms[chosen_arm]
        self.t += 1

    def get_arm_prob(self):
        return self.prob_arm

    def set_arm_prob(self, prob_arm):
        self.prob_arm = prob_arm

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name


class Uniform(Base):
    def __init__(self, n_arms, n_rounds, explore_weight=1):
        super().__init__(n_arms, n_rounds, explore_weight)
        self.name = 'Uniform'

    def select_arm(self):
        return np.random.randint(0, self.n_arms)
