import numpy as np
from KL_MS import update_prob_arms


class KL_MS_JeffreysPrior:
    def __init__(self, n_arms, explore_weight=1):
        self.n_arms = n_arms
        self.means = np.full(shape=n_arms, fill_value=0.5)
        self.N_arms = np.zeros(shape=n_arms)
        self.t = 0
        self.explore_weight = explore_weight

        # initialize the probability of each arm as the uniform distribution
        self.prob_arm = np.full(shape=n_arms, fill_value=1 / n_arms)

    def select_arm(self):
        # choose an arm according to the probability of each arm
        if np.random.random() < self.explore_weight:
            chosen_arm = int(np.random.choice(range(self.n_arms), p=self.prob_arm))
        else:
            chosen_arm = int(np.argmax(self.prob_arm))
        return chosen_arm

    def update(self, chosen_arm, reward):
        self.t += 1
        S = self.means[chosen_arm] * self.N_arms[chosen_arm] + reward
        self.means[chosen_arm] = (0.5 + S) / (self.N_arms[chosen_arm] + 1 + 1)
        self.N_arms[chosen_arm] += 1

        # update the probability of each arm after the first n_arms rounds
        if self.t >= self.n_arms:
            self.prob_arm = update_prob_arms(self.means, self.N_arms)
