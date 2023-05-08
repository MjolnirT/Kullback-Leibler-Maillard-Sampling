import numpy as np


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
        chosen_arm = int(np.argmax(theta_samples))
        return chosen_arm

    def update(self, chosen_arm, reward):
        self.t += 1
        self.S[chosen_arm] += reward
        self.N_arms[chosen_arm] += 1
        self.alpha[chosen_arm] = self.prior_alpha + self.S[chosen_arm]
        self.beta[chosen_arm] = self.prior_beta + self.N_arms[chosen_arm] - self.S[chosen_arm]
