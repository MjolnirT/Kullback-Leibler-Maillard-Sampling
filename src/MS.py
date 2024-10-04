import numpy as np
from .utility_functions.utility import gaussian_KL_divergence
from .Base import Base
from .simulation import simulate_one_alg


def calculate_prob_arms(means, variance, N_arms):
    # calculate the probability of each arm
    prob_arm = np.zeros(shape=len(means))
    best_arm = np.argmax(means)

    for i in range(len(means)):
        prob_arm[i] = gaussian_KL_divergence(means[i], variance, means[best_arm], variance)
    prob_arm = np.exp(-N_arms * prob_arm)
    prob_arm = prob_arm / np.sum(prob_arm)
    return prob_arm


class MS(Base):
    def __init__(self, n_arms, T_timespan, variance, explore_weight=1):
        super().__init__(n_arms, T_timespan, explore_weight)
        self.variance = variance
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
            self.prob_arm = calculate_prob_arms(self.means, self.variance, self.N_arms)


class MSPlus(Base):
    def __init__(self, n_arms, T_timespan, B, C, D, variance, explore_weight=1):
        super().__init__(n_arms, T_timespan, explore_weight)
        self.prob_arm = np.full(shape=n_arms, fill_value=1 / n_arms)
        self.B = B
        self.C = C
        self.D = D
        self.variance = variance

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
            KL = gaussian_KL_divergence(self.means, self.variance, self.means[self.means.argmax()], self.variance)
            self.prob_arm = np.exp(- self.N_arms * KL + np.log(1 + self.D * KL))
            self.prob_arm[self.means.argmax()] = \
                self.B * (1 + self.C * np.log(1 + np.log(self.t / self.N_arms[self.means.argmax()])))
            self.prob_arm = self.prob_arm / np.sum(self.prob_arm)

def SearchOptConfig(reward, n_arms, n_rounds):
    B = np.exp(np.linspace(1, 10, 10))
    C = np.exp(-np.linspace(1, 10, 4))
    D = np.exp(-np.linspace(1, 10, 4))
    configs = np.array(np.meshgrid(B, C, D)).T.reshape(-1, 3)

    best_regret = n_rounds * (max(reward) - min(reward))
    best_config = None
    for idx, config in enumerate(configs):

        model = MSPlus(n_arms, n_rounds, *config, 1 / 4)
        _, rewards, best_reward, _ = simulate_one_alg(reward, n_arms, n_rounds, model)
        regret = np.array(best_reward) - np.array(rewards)

        if regret[-1] < best_regret:
            best_regret = regret[-1]
            best_config = config

    print(f'best regret: {best_regret}')
    print(f'best config: {best_config}')

    return best_config