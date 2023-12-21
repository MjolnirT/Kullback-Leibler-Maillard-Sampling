import numpy as np
from utility import binomial_KL_divergence
from Base import Base


def calculate_prob_arms(means, N_arms):
    # calculate the probability of each arm
    prob_arm = np.zeros(shape=len(means))
    best_arm = np.argmax(means)

    for i in range(len(means)):
        prob_arm[i] = binomial_KL_divergence(means[i], means[best_arm])
    prob_arm = np.exp(-N_arms * prob_arm)
    prob_arm = prob_arm / np.sum(prob_arm)
    return prob_arm


class KLMS(Base):
    def __init__(self, n_arms, T_timespan, explore_weight=1):
        super().__init__(n_arms, T_timespan, explore_weight)
        self.prob_arm = np.full(shape=n_arms, fill_value=1 / n_arms)

    def select_arm(self):
        # choose an arm according to the probability of each arm
        if np.random.random() <= self.explore_weight:
            chosen_arm = int(np.random.choice(range(self.n_arms), p=self.prob_arm))
        else:
            chosen_arm = int(np.argmax(self.prob_arm))
        return chosen_arm

    def update(self, chosen_arm, reward):
        super().update(chosen_arm, reward)
        if self.t >= self.n_arms:
            self.prob_arm = calculate_prob_arms(self.means, self.N_arms)


class KLMSJefferysPrior(Base):
    def __init__(self, n_arms, T_timespan, explore_weight=1):
        super().__init__(n_arms, T_timespan, explore_weight)
        self.prob_arm = np.full(shape=n_arms, fill_value=1 / n_arms)

        # initialize parameters for the beta distribution
        self.prior_alpha = 0.5
        self.prior_beta = 0.5
        self.alpha = np.full(shape=n_arms, fill_value=0.5)
        self.beta = np.full(shape=n_arms, fill_value=0.5)

        # initialize the KT estimator for the mean of each arm
        self.KTEstimator = np.zeros(shape=n_arms)

    def select_arm(self):
        # choose an arm according to the probability of each arm
        if np.random.random() <= self.explore_weight:
            chosen_arm = int(np.random.choice(range(self.n_arms), p=self.prob_arm))
        else:
            chosen_arm = int(np.argmax(self.prob_arm))
        return chosen_arm

    def update(self, chosen_arm, reward):
        super().update(chosen_arm, reward)

        self.alpha = self.prior_alpha + self.rewards.sum(axis=1)
        self.beta = self.prior_beta + self.N_arms - self.rewards.sum(axis=1)
        self.KTEstimator = self.alpha / (self.alpha + self.beta)
        if self.t >= self.n_arms:
            self.prob_arm = calculate_prob_arms(self.KTEstimator, self.N_arms)
