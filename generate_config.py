import json
import os
from src.utility_functions.utility import message
from src.utility_functions.utility_io import get_filename, check_folder_exist
from src.global_config import LOG_FLAG, CONFIG_DIR
# Configuration parameters

# test case 1: [0.2, 0.25]
# test case 2: [0.8, 0.9]
TEST_CASE = 2
ENV_REWARD = [0.2, 0.25] if TEST_CASE == 1 else [0.8, 0.9]


N_SIMULATIONS = 2000
T_TIMESPAN = 10000
INCLUDED_ALG = ['BernoulliTS', 'KL-MS', 'KL-MS+JefferysPrior', 'MS', 'MS+', 'simuBernoulliTS']
MC_SIMULATION_ROUND = 1000
VARIANCE = 0.25

def generate_config():

    check_folder_exist(CONFIG_DIR)
    n_arms = len(ENV_REWARD)
    config = {
        "environment": {
            "reward": ENV_REWARD,
            "test case": TEST_CASE,
            "n_simulations": N_SIMULATIONS,
            "base": {
                "T_timespan": T_TIMESPAN,
                "n_arms": n_arms
            }
        }
    }
    message(f'reward_probabilities: {ENV_REWARD}', print_flag=LOG_FLAG)

    algorithms = {
        idx: {
            'name': alg,
            'model': alg,
            'params': {
                "MC_simulation_round": MC_SIMULATION_ROUND if alg == 'BernoulliTS' else None,
                "variance": VARIANCE if alg in ['MS', 'MS+'] else None
            }
        }
        for idx, alg in enumerate(INCLUDED_ALG)
    }

    # Remove None values from params
    for alg in algorithms.values():
        alg['params'] = {k: v for k, v in alg['params'].items() if v is not None}

    config["algorithms"] = algorithms
    config_json = json.dumps(config, indent=4)

    config_filename = get_filename(T_TIMESPAN, N_SIMULATIONS, TEST_CASE, MC_SIMULATION_ROUND, is_configuration=True)
    config_filepath = os.path.join(CONFIG_DIR, config_filename)
    with open(config_filepath, 'w') as file:
        file.write(config_json)

    message(f'config file {config_filename} is generated.', print_flag=LOG_FLAG)

if __name__ == '__main__':
    generate_config()
