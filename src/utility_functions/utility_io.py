import json
import os
from ..BernoulliKLMS import KLMS, KLMSJefferysPrior
from ..BernoulliTS import BernoulliTS, simuBernoulliTS
from ..MS import MS, MSPlus, SearchOptConfig
from .utility import message


# generate the simulation_filename for the simulation results
# input: the parameters for the simulation
# output: the simulation_filename for the simulation results
def get_filename(T_timespan, n_simulations, test_case, simulations_per_round,
                 is_simulation=False, is_evaluation=False, is_configuration=False):
    if is_simulation:
        filename = 'simulation'
    elif is_evaluation:
        filename = 'evaluation'
    elif is_configuration:
        filename = 'config'
    else:
        raise ValueError("Invalid filename type")

    filename += f'_T_{T_timespan}_s_{n_simulations}_test{test_case}_MC_{simulations_per_round}'
    filename += '.json' if is_configuration else '.pkl'

    return filename


# parse the algorithm configuration
# input: one is the algorithm dictionary, the other is the environment dictionary
# output: a dictionary with the algorithm name as key, and the algorithm class and parameters as value
def parse_algorithm(alg_dict, environment):
    alg_dict_out = {}
    alg_param = environment["base"].copy()
    alg_param.update(alg_dict['params'])

    if alg_dict['model'] == 'BernoulliTS':
        alg_dict_out[alg_dict['name']] = {'model': BernoulliTS, 'params': alg_param}
    elif alg_dict['model'] == 'KL-MS':
        alg_dict_out[alg_dict['name']] = {'model': KLMS, 'params': alg_param}
    elif alg_dict['model'] == 'KL-MS+JefferysPrior':
        alg_dict_out[alg_dict['name']] = {'model': KLMSJefferysPrior, 'params': alg_param}
    elif alg_dict['model'] == 'MS':
        alg_dict_out[alg_dict['name']] = {'model': MS, 'params': alg_param}
    elif alg_dict['model'] == 'MS+':
        opt_config = SearchOptConfig(environment["reward"], n_arms=alg_param["n_arms"], n_rounds=100)
        alg_param.update({"B": opt_config[0], "C": opt_config[1], "D": opt_config[2]})
        alg_dict_out[alg_dict['name']] = {'model': MSPlus, 'params': alg_param}
    elif alg_dict['model'] == 'simuBernoulliTS':
        alg_dict_out[alg_dict['name']] = {'model': simuBernoulliTS, 'params': alg_param}

    return alg_dict_out


# read the configuration file
# input: a CONFIG json file
# output: two dictionaries, one for environment, one for algorithms
def read_algorithms(filename, print_flag=None):
    message(f'read configuration file: {filename}', print_flag=True)
    algorithms = {}
    with open(filename, 'r') as file:
        config_json = file.read()
    config = json.loads(config_json)
    environment = config["environment"]

    for key, value in config.items():
        message(f'{key}: {value}', print_flag=print_flag)
        if key != 'environment':
            for alg_idx, alg in value.items():
                message(f'{alg_idx}: {alg}', print_flag=print_flag)
                algorithms.update(parse_algorithm(alg, environment))

    return environment, algorithms

def check_folder_exist(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        message(f'Folder {folder_name} created.')
    else:
        message(f'Folder {folder_name} already exists.')
    return folder_name