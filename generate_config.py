import json

from SearchOptConfig import SearchOptConfig
from utility import message
from utility_io import get_filename

if __name__ == '__main__':
    print_flag = True
    # env_reward = [0.2, 0.25]
    # test_case = 1

    env_reward = [0.8, 0.9]
    test_case = 2

    # env_reward = np.linspace(0.1, 0.9, 9)
    # test_case = 3

    # env_reward = [0.8, 0.9]
    # test_case = 4

    n_simulations = 50
    T_timespan = 2000
    n_arms = len(env_reward)

    config = {"environment": {"reward": env_reward,
                              "test case": test_case,
                              "n_simulations": n_simulations,
                              "base": {"T_timespan": T_timespan,
                                       "n_arms": n_arms}}}
    message(f'reward_probabilities: {env_reward}', print_flag=True)

    # set algorithms and their parameters
    included_alg = ['BernoulliTS', 'KL-MS', 'KL-MS+JefferysPrior', 'MS', 'MS+', 'simuBernoulliTS']
    MC_simulation_round = 10000
    variance = float(1 / 4)

    algorithms = {}
    for idx, alg in enumerate(included_alg):
        algorithms[idx] = {'name': alg,
                           'model': alg,
                           'params': {}}
        if alg == 'BernoulliTS':
            algorithms[idx]['params']["simulation_rounds"] = MC_simulation_round
        if alg == 'MS':
            algorithms[idx]['params']["variance"] = variance
        if alg == 'MS+':
            algorithms[idx]['params']["variance"] = variance

    config["algorithms"] = algorithms
    config_json = json.dumps(config, indent=4)

    filename = get_filename(T_timespan, n_simulations, test_case,
                            MC_simulation_round, is_configuration=True)

    with open('config/' + filename, 'w') as file:
        file.write(config_json)

    message(f'config file {filename} is generated.', print_flag=True)
