import json
import pickle
import time
import numpy as np
import os
from ..Base import Uniform
from .utility import plot_hist_overlapped, message, plot_lines
from .utility_io import get_filename
from ..global_config import DATA_DIR, PLOT_DIR, CONFIG_DIR, LOG_FLAG


def generate_eval_plots(filename, env_reward, algorithms_name,
                        n_simulations, n_arms, n_algorithms, T_timespan,
                        output_dir,
                        ref_alg=None, exclude_alg=None, is_print=True):
    with open(filename, 'rb') as file:
        eval_result = pickle.load(file)
    file.close()

    eval_reward = np.zeros(shape=[n_simulations, n_algorithms, T_timespan])
    select_arms = np.zeros(shape=[n_simulations, n_algorithms, T_timespan], dtype=int)
    for i in range(n_simulations):
        select_arms[i], eval_reward[i] = eval_result[i]
    eval_reward = np.cumsum(eval_reward, axis=2)

    oracle = (np.ones(shape=n_arms) / n_arms).dot(env_reward)

    message(f"Start plotting", is_print)
    experiment_param = ' | mu=' + str(env_reward) + ' | simulations=' + str(n_simulations)
    plot_filepath = os.path.join(output_dir, 'eval_reward_line.png')
    plot_lines(eval_reward,
               ci=0.95,
               x_label='time step',
               y_label='cumulative reward',
               title='Cumulative Reward Comparison' + experiment_param,
               label=algorithms_name,
               ref_alg=ref_alg,
               add_ci=True,
               save_path=plot_filepath,
               figure_size=(8, 6),
               font_size=18,
               exclude_alg=exclude_alg)

    # eval_reward_last = eval_reward[:, :, -1]
    eval_reward_last = eval_reward[:, :, -1] / T_timespan
    plot_filepath = os.path.join(output_dir, 'eval_reward_hist.png')
    plot_hist_overlapped(eval_reward_last,
                         x_label='average reward',
                         y_label='frequency',
                         title='Average Reward Distribution' + experiment_param,
                         label=algorithms_name,
                         add_density=True,
                         add_mean=False,
                         oracle=oracle,
                         save_path=plot_filepath,
                         figure_size=(8, 6),
                         font_size=21,
                         exclude_alg=exclude_alg)
    MSE = np.mean((eval_reward_last - oracle) ** 2, axis=0)
    mean = np.mean(eval_reward_last, axis=0)
    message(f"MSE: {MSE}", is_print)
    message(f"mean: {mean}", is_print)

    message('-------------------', is_print)
    for alg_idx in range(n_algorithms):
        if algorithms_name[alg_idx] in exclude_alg:
            continue
        generate_metric(eval_reward_last[:, alg_idx], oracle, is_print, algorithms_name[alg_idx])
        message('-------------------', is_print)


def generate_metric(reward, oracle, is_print, algorithm_name):
    finite_reward = reward[np.isfinite(reward)]
    MSE = np.mean((finite_reward - oracle) ** 2)
    mean = np.mean(finite_reward)
    message(f"{algorithm_name} MSE: {MSE}", is_print)
    message(f"{algorithm_name} mean: {mean}", is_print)
    message(f'{algorithm_name} Bias: {mean - oracle}', is_print)
    message(f"{algorithm_name} Drop {len(reward) - len(finite_reward)} NaN values", is_print)
    return MSE, mean


if __name__ == '__main__':
    start_time = time.time()

    config_filename = 'figure1.json'
    config_filepath = os.path.join(CONFIG_DIR, config_filename)
    message(f"Read configuration from {config_filepath}.", LOG_FLAG)
    with open(config_filepath, 'r') as f:
        config = json.load(f)

    MC_simulation_round = "1000"

    environment = config["environment"]
    env_reward = environment["reward"]
    test_case = environment['test case']
    n_simulations = environment['n_simulations']

    T_timespan = environment["base"]['T_timespan']
    n_arms = environment["base"]['n_arms']

    algorithms = config["algorithms"]
    n_algorithms = len(algorithms)
    algorithms_name = [algorithms[key]["name"] for key in algorithms]
    exclude_alg = ['KL-MS+JefferysPrior', 'MS', 'MS+', 'simuBernoulliTS']

    eval_algorithm = {
        'Uniform': {
            'model': Uniform,
            'params': {"n_arms": n_arms, "T_timespan": T_timespan}
        }
    }
    eval_algorithms_name = next(iter(eval_algorithm))

    evaluation_filename = get_filename(T_timespan, n_simulations, test_case,
                            MC_simulation_round,
                            is_evaluation=True)
    evaluation_filepath = os.path.join(DATA_DIR, evaluation_filename)
    message(f'Read simulation results from {evaluation_filepath}.', LOG_FLAG)
    generate_eval_plots(evaluation_filepath, env_reward, algorithms_name,
                        n_simulations, n_arms, n_algorithms, T_timespan, PLOT_DIR,
                        ref_alg=algorithms_name, exclude_alg=exclude_alg, is_print=LOG_FLAG)
    message(f'Time elapsed: {time.time() - start_time:.2f}s', LOG_FLAG)
