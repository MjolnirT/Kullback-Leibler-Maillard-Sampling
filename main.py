import numpy as np
from utility import plot_regrets, message
from BernoulliTS import BernoulliTS
from BernoulliKLMS import KLMS, KLMSJefferysPrior


def simulate(reward_probabilities, n_rounds, algorithm):
    n_arms = len(reward_probabilities)
    rewards = []
    selected_arms = []
    best_reward = np.max(reward_probabilities) * np.ones(n_rounds)

    for t in range(n_rounds):
        chosen_arm = None
        # In the first n_arms rounds, play each arm once
        if t < n_arms:
            chosen_arm = t

        # After the first n_arms rounds, use the algorithm to select an arm
        if t >= n_arms:
            chosen_arm = algorithm.select_arm()

        # receive reward from the chosen arm, use the expected reward as the reward
        reward = reward_probabilities[chosen_arm]
        algorithm.update(chosen_arm, reward)

        # record the results
        selected_arms.append(chosen_arm)
        rewards.append(reward)

    return selected_arms, rewards, best_reward


if __name__ == '__main__':
    reward_probabilities = [0.1] * 5 + [0.5] * 10 + [0.8] * 10 + [0.97] * 10 + [0.98] * 5 + [0.999]
    # reward_probabilities = [0.1]*5 + [0.5]*10 + [0.6]

    print_flag = False
    message(f'reward_probabilities: {reward_probabilities}', print_flag=True)
    n_rounds = 1000
    n_arms = len(reward_probabilities)

    message('BernoulliTS', print_flag=print_flag)
    algorithm = BernoulliTS(n_arms, n_rounds)
    _, rewards, best_reward = simulate(reward_probabilities, n_rounds, algorithm)
    B_TS_regret = np.array(best_reward) - np.array(rewards)

    message('KL_MS', print_flag=print_flag)
    algorithm = KLMS(n_arms, explore_weight=1.0, n_rounds=n_rounds)
    _, rewards, best_reward = simulate(reward_probabilities, n_rounds, algorithm)
    MS_regret = np.array(best_reward) - np.array(rewards)
    message(f"Probability of arms: {algorithm.prob_arm}", print_flag=print_flag)

    message('KL_MS_JeffreysPrior', print_flag=print_flag)
    algorithm = KLMSJefferysPrior(n_arms, explore_weight=1.0, n_rounds=n_rounds)
    _, rewards, best_reward = simulate(reward_probabilities, n_rounds, algorithm)
    MS_Jeff_regret_2 = np.array(best_reward) - np.array(rewards)

    label = ['Bernoulli TS', 'KLMS', 'KLMS with Jeffreys Prior']
    plot_regrets([B_TS_regret, MS_regret, MS_Jeff_regret_2], 'Regret Comparison', label)
