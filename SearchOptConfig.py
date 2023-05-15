import numpy as np
from MS import MSPlus
from simulation import simulate_one_alg


def SearchOptConfig(reward, n_arms, n_rounds):
    B = np.exp(np.linspace(1, 10, 10))
    C = np.exp(-np.linspace(1, 10, 4))
    D = np.exp(-np.linspace(1, 10, 4))
    configs = np.array(np.meshgrid(B, C, D)).T.reshape(-1, 3)

    best_regret = n_rounds * (max(reward) - min(reward))
    best_config = None
    for idx, config in enumerate(configs):
        # if idx % 100 == 0:
        #     print(f'idx: {idx} / {len(configs)}')

        model = MSPlus(n_arms, n_rounds, *config, 1 / 4)
        _, rewards, best_reward, _ = simulate_one_alg(reward, n_arms, n_rounds, model)
        regret = np.array(best_reward) - np.array(rewards)

        if regret[-1] < best_regret:
            best_regret = regret[-1]
            best_config = config

    print(f'best regret: {best_regret}')
    print(f'best config: {best_config}')

    return best_config


env_reward = [0.2] + [0.25]
# reward_probabilities = [0.8] + [0.9]

n_arms = len(env_reward)
n_rounds = 100
SearchOptConfig(env_reward, n_arms, n_rounds)
