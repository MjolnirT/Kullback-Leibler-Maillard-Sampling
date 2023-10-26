import json
import pickle
import time

import numpy as np

from Base import Uniform
from utility import plot_hist_overlapped, message, plot_lines
from utility_io import get_filename


def generate_eval_plots(filename, env_reward, algorithms_name,
                        n_simulations, n_arms, n_algorithms, T_timespan,
                        ref_alg=None, exclude_alg=None, is_print=True):
    with open(filename, 'rb') as file:
        eval_result = pickle.load(file)
    file.close()

    eval_reward = np.zeros(shape=[n_simulations, n_algorithms, T_timespan])
    select_arms = np.zeros(shape=[n_simulations, n_algorithms, T_timespan], dtype=int)
    for i in range(n_simulations):
        select_arms[i], eval_reward[i] = eval_result[i]
    eval_reward = np.cumsum(eval_reward, axis=2)

    # oracle = (np.ones(shape=n_arms) / n_arms).dot(env_reward) * T_timespan
    oracle = (np.ones(shape=n_arms) / n_arms).dot(env_reward)

    message(f"Start plotting", is_print)
    experiment_param = ' | mu=' + str(env_reward) + ' | simulations=' + str(n_simulations)
    plot_lines(eval_reward,
               ci=0.95,
               x_label='time step',
               y_label='cumulative reward',
               # title='Cumulative Reward Comparison' + experiment_param,
               label=algorithms_name,
               ref_alg=ref_alg,
               add_ci=True,
               save_path='./figures/eval_reward_line.png',
               figure_size=(8, 6),
               font_size=18,
               exclude_alg=exclude_alg)

    # eval_reward_last = eval_reward[:, :, -1]
    eval_reward_last = eval_reward[:, :, -1] / T_timespan

    plot_hist_overlapped(eval_reward_last,
                         x_label='average reward',
                         y_label='frequency',
                         # title='Cumulative Reward Distribution' + experiment_param,
                         label=algorithms_name,
                         add_density=True,
                         add_mean=False,
                         oracle=oracle,
                         save_path='./figures/eval_reward_hist.png',
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
    is_print = True

    path = 'config/'
    config_file = path + 'test2_T_10K_MC_1K.json'
    message(f"Read configuration from {config_file}.", is_print)
    with open(config_file, 'r') as f:
        config = json.load(f)
    f.close()
    simulations_per_round = "1000"
    split_points = "NA"
    is_interpolation = False

    environment = config["environment"]
    env_reward = environment["reward"]
    test_case = environment['test case']
    n_simulations = environment['n_simulations']

    T_timespan = environment["base"]['n_rounds']
    n_arms = environment["base"]['n_arms']

    n_algorithms = len(config["algorithms"])
    algorithms_name = [config["algorithms"][key]["name"] for key in config["algorithms"]]
    exclude_alg = ['KL-MS+JefferysPrior', 'MS', 'MS+', 'BernoulliTS+RiemannApprox']

    eval_algorithm = {'Uniform':
                          {'model': Uniform,
                           'params': {"n_arms": n_arms, "n_rounds": T_timespan}}}
    eval_algorithms_name = list(eval_algorithm.keys())[0]

    filename = get_filename(T_timespan, n_simulations, test_case,
                            simulations_per_round, split_points, is_interpolation,
                            is_evaluation=True)
    message(f'Read simulation results from {filename}.', is_print)
    generate_eval_plots(filename, env_reward, algorithms_name,
                        n_simulations, n_arms, n_algorithms, T_timespan,
                        ref_alg=algorithms_name, exclude_alg=exclude_alg, is_print=is_print)
    message(f'Time elapsed: {time.time() - start_time:.2f}s', is_print)
