import numpy as np
from MS import MSPlus
from simulate import simulate


def SearchOptConfig(reward_probabilities, n_rounds):
    B = np.exp(np.linspace(1, 10, 10))
    C = np.exp(-np.linspace(1, 10, 4))
    D = np.exp(-np.linspace(1, 10, 4))
    configs = np.array(np.meshgrid(B, C, D)).T.reshape(-1, 3)

    n_arms = len(reward_probabilities)

    best_regret = n_rounds * (max(reward_probabilities) - min(reward_probabilities))
    best_config = None
    for idx, config in enumerate(configs):
        if idx % 100 == 0:
            print(f'idx: {idx} / {len(configs)}')

        model = MSPlus(n_arms, n_rounds, *config, 1/4)
        _, rewards, best_reward = simulate(reward_probabilities, n_rounds, model)
        regret = np.array(best_reward) - np.array(rewards)

        if regret[-1] < best_regret:
            best_regret = regret[-1]
            best_config = config

    print(f'best regret: {best_regret}')
    print(f'best config: {best_config}')

    return best_config


reward_probabilities = [0.1] * 5 + [0.5] * 10 + [0.8] * 10 + [0.97] * 10 + [0.98] * 5 + [0.999]
SearchOptConfig(reward_probabilities, 100)
