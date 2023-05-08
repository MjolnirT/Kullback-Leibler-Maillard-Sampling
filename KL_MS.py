import numpy as np
from utility import binomial_KL_divergence


def update_prob_arms(means, N_arms):
    prob_arm = np.zeros(shape=len(means))
    best_arm = np.argmax(means)

    for i in range(len(means)):
        prob_arm[i] = binomial_KL_divergence(means[i], means[best_arm])
    prob_arm = np.exp(-N_arms * prob_arm)
    prob_arm = prob_arm / np.sum(prob_arm)
    return prob_arm


class KL_MS:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.means = np.zeros(shape=n_arms)
        self.N_arms = np.zeros(shape=n_arms)
        self.t = 0
        self.prob_arm = np.full(shape=n_arms, fill_value=1 / n_arms)

    def select_arm(self):
        chosen_arm = np.random.choice(range(self.n_arms), p=self.prob_arm)
        return chosen_arm

    def update(self, chosen_arm, reward):
        self.t += 1
        self.means[chosen_arm] = (self.means[chosen_arm] * self.N_arms[chosen_arm] + reward) / (self.N_arms[chosen_arm] + 1)
        if self.t >= self.n_arms:
            self.prob_arm = update_prob_arms(self.means, self.N_arms)
        self.N_arms[chosen_arm] += 1

