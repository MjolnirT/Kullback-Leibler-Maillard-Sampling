import numpy as np
import torch


class Base:
    def __init__(self, n_arms, T_timespan, explore_weight=1, device='cpu'):
        self.T = torch.tensor(T_timespan, dtype=torch.int, device=device)
        self.n_arms = torch.tensor(n_arms, dtype=torch.int, device=device)
        self.t = torch.tensor(0, dtype=torch.int, device=device)
        self.explore_weight = torch.tensor(explore_weight, dtype=torch.float, device=device)

        self.device = device

        # initialize the records
        self.rewards = torch.zeros(n_arms, T_timespan, dtype=torch.float, device=self.device)
        self.N_arms = torch.zeros(n_arms, dtype=torch.int, device=self.device)
        self.means = torch.zeros(n_arms, dtype=torch.float, device=self.device)

        # initialize the probability of each arm as the uniform distribution
        self.prob_arm = torch.NoneType
        self.name = torch.NoneType

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
    def __init__(self, n_arms, T_timespan, explore_weight=1, device='cpu'):
        super().__init__(n_arms, T_timespan, explore_weight)
        self.name = 'Uniform'
        self.prob_arm = torch.full((n_arms,), 1 / n_arms, dtype=torch.float, device=device)

    def select_arm(self):
        return torch.randint(0, self.n_arms, (1,)).item()
