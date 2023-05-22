import numpy as np
from Base import Base
from scipy import stats
from utility import log_remove_inf


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


class simuBernoulliTS(BernoulliTS):
    def get_arm_prob(self):
        # running a Monte Carlo simulation to calculate the integral based on
        # "Simple Bayesian Algorithms for Best Arm Identification" by Daniel Russo
        simulation_points = 20
        log_pdf_sample = np.zeros(shape=[self.n_arms, simulation_points])
        log_cdf_sample = np.zeros(shape=[self.n_arms, simulation_points])
        sample = np.linspace(0, 1, simulation_points+2)[1:-1].reshape(1,-1)
        for i in range(self.n_arms):

            log_pdf_sample[i] = log_remove_inf(stats.beta.pdf(sample, self.alpha[i], self.beta[i]))
            log_cdf_sample[i] = log_remove_inf(stats.beta.cdf(sample, self.alpha[i], self.beta[i]))
        log_F = np.sum(log_cdf_sample, axis=0)
        log_ratio = log_pdf_sample - log_cdf_sample + log_F
        ratio = np.exp(log_ratio)
        arm_prod = np.sum(ratio, axis=1)
        arm_prod = arm_prod / np.sum(arm_prod)
        return arm_prod
