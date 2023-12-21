import numpy as np
from Base import Base
from scipy import stats


class BernoulliTS(Base):
    def __init__(self, n_arms, T_timespan, explore_weight=1, simulation_rounds=1000):
        super().__init__(n_arms, T_timespan, explore_weight)

        # initialize parameters for the beta distribution
        self.prior_alpha = 0.5
        self.prior_beta = 0.5
        self.alpha = np.full(shape=n_arms, fill_value=0.5)
        self.beta = np.full(shape=n_arms, fill_value=0.5)

        # initialize the cumulative S(r) for each arm
        self.S = np.zeros(shape=n_arms)

        # initialize the probability of each arm for offline evaluation
        self.prob_arm = np.full(shape=n_arms, fill_value=1 / n_arms)
        self.simulation_rounds = simulation_rounds

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
        theta_samples = np.random.beta(self.alpha, self.beta, size=(self.simulation_rounds, self.n_arms))
        arm_counts = np.argmax(theta_samples, axis=1)
        counts = np.bincount(arm_counts, minlength=self.n_arms)
        self.prob_arm = counts / self.simulation_rounds
        return self.prob_arm


class simuBernoulliTS(BernoulliTS):
    def __init__(self, n_arms, T_timespan, explore_weight=1,
                 simulation_rounds=None, split_points=20):
        super().__init__(n_arms, T_timespan, explore_weight, simulation_rounds)
        self.split_points = split_points

    def get_arm_prob(self):
        # running a Monte Carlo simulation to calculate the integral based on
        # "Simple Bayesian Algorithms for Best Arm Identification" by Daniel Russo
        sample = np.linspace(0, 1, self.split_points + 2)[1:-1].reshape(-1, 1)
        sample = np.repeat(sample, self.n_arms, axis=1)
        with np.errstate(divide='ignore'):
            log_pdf_sample = np.log(stats.beta.pdf(sample, self.alpha, self.beta))
            log_cdf_sample = np.log(stats.beta.cdf(sample, self.alpha, self.beta))
            assert log_pdf_sample.shape == (self.split_points, self.n_arms)
            assert log_cdf_sample.shape == (self.split_points, self.n_arms)
            for k in range(self.n_arms):
                log_ratio = log_cdf_sample[:, np.arange(log_cdf_sample.shape[1]) != k].sum(axis=1)
                self.prob_arm[k] = np.exp(log_ratio + log_pdf_sample[:, k]).sum()
        self.prob_arm = self.prob_arm / self.split_points
        return self.prob_arm
