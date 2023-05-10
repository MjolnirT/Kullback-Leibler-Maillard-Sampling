import numpy as np
from Base import Base


class BernoulliTS(Base):
    def __init__(self, n_arms, n_rounds, explore_weight=1):
        super().__init__(n_arms, n_rounds, explore_weight)

        # initialize parameters for the beta distribution
        self.prior_alpha = 0.5
        self.prior_beta = 0.5
        self.alpha = np.full(shape=n_arms, fill_value=0.5)
        self.beta = np.full(shape=n_arms, fill_value=0.5)

        # initialize the cumulative S(r) for each arm
        self.S = np.zeros(shape=n_arms)

        # initialize the probability of each arm for offline evaluation
        self.prob_arm = np.full(shape=n_arms, fill_value=1 / n_arms)

    def select_arm(self):
        theta_samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n_arms)]
        chosen_arm = int(np.argmax(theta_samples))
        return chosen_arm

    def update(self, chosen_arm, reward):
        super().update(chosen_arm, reward)
        self.S[chosen_arm] += reward
        self.alpha[chosen_arm] = self.prior_alpha + self.S[chosen_arm]
        self.beta[chosen_arm] = self.prior_beta + self.N_arms[chosen_arm] - self.S[chosen_arm]

    def get_arm_prob(self):
        # running a Monte Carlo simulation to get the probability of each arm
        simulation_rounds = 1000
        self.prob_arm = np.zeros(shape=self.n_arms)
        for i in range(simulation_rounds):
            theta_samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n_arms)]
            self.prob_arm[np.argmax(theta_samples)] += 1 / simulation_rounds
        return self.prob_arm