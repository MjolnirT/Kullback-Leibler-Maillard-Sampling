import json
import torch
from Base import Uniform, BatchUniform
from BernoulliKLMS import KLMS, KLMSJefferysPrior
from BernoulliTS import BernoulliTS, simuBernoulliTS
from MS import MS, MSPlus
from SearchOptConfig import SearchOptConfig
from utility import message


# generate the filename for the simulation results
# input: the parameters for the simulation
# output: the filename for the simulation results
def get_filename(T_timespan, n_simulations, test_case, simulations_per_round,
                 is_simulation=False, is_evaluation=False, is_configuration=False):
    filename = None
    if is_simulation:
        filename = 'simulation'
    if is_evaluation:
        filename = 'evaluation'
    if is_configuration:
        filename = 'config'

    filename = filename + '_T_' + str(T_timespan) + \
               '_s_' + str(n_simulations) + \
               '_test' + str(test_case) + \
               '_MC_' + str(simulations_per_round)
    if is_configuration:
        filename = filename + '.json'
    else:
        filename = filename + '.pkl'

    return filename


# parse the algorithm configuration
# input: one is the algorithm dictionary, the other is the environment dictionary
# output: a dictionary with the algorithm name as key, and the algorithm class and parameters as value
def parse_algorithm(alg_dict, environment, device):
    alg_dict_out = {}
    alg_param = environment["base"].copy()
    alg_param.update(alg_dict['params'])
    alg_param['device'] = device

    if alg_dict['model'] == 'Uniform':
        alg_dict_out[alg_dict['name']] = {'model': Uniform,
                                          'params': alg_param}
    if alg_dict['model'] == 'BernoulliTS':
        alg_dict_out[alg_dict['name']] = {'model': BernoulliTS,
                                          'params': alg_param}
    if alg_dict['model'] == 'KL-MS':
        alg_dict_out[alg_dict['name']] = {'model': KLMS,
                                          'params': alg_param}
    if alg_dict['model'] == 'KL-MS+JefferysPrior':
        alg_dict_out[alg_dict['name']] = {'model': KLMSJefferysPrior,
                                          'params': alg_param}
    if alg_dict['model'] == 'MS':
        alg_dict_out[alg_dict['name']] = {'model': MS,
                                          'params': alg_param}
    if alg_dict['model'] == 'MS+':
        reward = torch.tensor(environment["reward"], dtype=torch.float, device=device)
        n_arms = torch.tensor(environment["base"]["n_arms"], dtype=torch.int, device=device)
        opt_config = SearchOptConfig(reward, n_arms, n_rounds=100, device=device)
        alg_param.update({"B": opt_config[0],
                          "C": opt_config[1],
                          "D": opt_config[2]})
        alg_dict_out[alg_dict['name']] = {'model': MSPlus,
                                          'params': alg_param}
    if alg_dict['model'] == 'simuBernoulliTS':
        alg_dict_out[alg_dict['name']] = {'model': simuBernoulliTS,
                                          'params': alg_param}
        
    return alg_dict_out


def parse_batch_algorithm(alg_dict, environment, device, batch_size):
    alg_dict_out = {}
    alg_param = environment["base"].copy()
    alg_param.update(alg_dict['params'])
    alg_param['device'] = device
    alg_param['batch_size'] = batch_size

    if alg_dict['model'] == 'Uniform':
        alg_dict_out[alg_dict['name']] = {'model': BatchUniform,
                                          'params': alg_param}
    # if alg_dict['model'] == 'BernoulliTS':
    #     alg_dict_out[alg_dict['name']] = {'model': BernoulliTS,
    #                                       'params': alg_param}
        
    return alg_dict_out


# read the configuration file
# input: a CONFIG json file
# output: two dictionaries, one for environment, one for algorithms
def read_algorithms(filename, batch_size=None, print_flag=None, device=None):
    message(f'Using {device} in regular mode', print_flag)
    message(f'read configuration file: {filename}', print_flag=True)
    algorithms = {}
    with open(filename, 'r') as file:
        config_json = file.read()
    config = json.loads(config_json)
    environment = config["environment"]
    for key, value in config.items():
        message(f'{key}: {value}', print_flag=print_flag)
        if key == 'algorithms':
            for alg_idx, alg in value.items():
                message(f'{alg_idx}: {alg}', print_flag=print_flag)
                algorithms.update(parse_algorithm(alg, environment, device))

    environment = config["environment"]
    return environment, algorithms


def read_batch_algorithms(filename, batch_size, print_flag=None, device=None):
    message(f'Using {device} in batch mode', print_flag)
    message(f'read configuration file: {filename}', print_flag=True)
    algorithms = {}
    with open(filename, 'r') as file:
        config_json = file.read()
    config = json.loads(config_json)
    environment = config["environment"]
    for key, value in config.items():
        message(f'{key}: {value}', print_flag=print_flag)
        if key == 'algorithms':
            for alg_idx, alg in value.items():
                added_alg = parse_batch_algorithm(alg, environment, device, batch_size)
                algorithms.update(added_alg)
                message(f'{added_alg}', print_flag=print_flag)

    environment = config["environment"]
    return environment, algorithms

