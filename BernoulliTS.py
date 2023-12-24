import numpy as np
from Base import Base
from scipy import stats
import torch


class BernoulliTS(Base):
    def __init__(self, n_arms, T_timespan, device, explore_weight=1, simulation_rounds=1000):
        super().__init__(n_arms, T_timespan, explore_weight)

        self.device = device
        # initialize parameters for the beta distribution
        self.prior_alpha = torch.tensor(0.5, dtype=torch.float, device=self.device)
        self.prior_beta = torch.tensor(0.5, dtype=torch.float, device=self.device)
        self.alpha = torch.full((n_arms,), fill_value=0.5, dtype=torch.float, device=self.device)
        self.beta = torch.full((n_arms,), fill_value=0.5, dtype=torch.float, device=self.device)

        # initialize the cumulative S(r) for each arm
        self.S = torch.zeros(n_arms, dtype=torch.float, device=self.device)

        # initialize the probability of each arm for offline evaluation
        self.prob_arm = torch.full((n_arms,), fill_value=1 / n_arms, dtype=torch.float, device=self.device)
        self.simulation_rounds = torch.tensor(simulation_rounds, dtype=torch.int, device=self.device)

    def select_arm(self):
        beta_dist = torch.distributions.beta.Beta(self.alpha, self.beta)
        theta_samples = beta_dist.sample()
        chosen_arm = torch.argmax(theta_samples).item()
        return chosen_arm

    def update(self, chosen_arm, reward):
        super().update(chosen_arm, reward)
        self.S[chosen_arm] += reward
        self.alpha[chosen_arm] = self.prior_alpha + self.S[chosen_arm]
        self.beta[chosen_arm] = self.prior_beta + self.N_arms[chosen_arm] - self.S[chosen_arm]

    def get_arm_prob(self):
        # running a Monte Carlo simulation to get the probability of each arm
        beta_dist = torch.distributions.beta.Beta(self.alpha, self.beta)
        theta_samples = beta_dist.sample((self.simulation_rounds,))
        arm_counts = torch.argmax(theta_samples, dim=1)
        counts = torch.bincount(arm_counts, minlength=self.n_arms)
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
